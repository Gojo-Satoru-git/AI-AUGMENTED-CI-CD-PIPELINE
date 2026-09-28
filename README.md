# AI-Augmented CI/CD Pipeline — Test-Triage Agent

Team 18 · Dept. of Information Technology, MIT Anna University
Praveen Kumar C · Jaiprakash S · Dharshini T · Guide: Dr. P. Anandha Kumar

Phase 2, workstreams **S1 (data foundation)**, **S2 (triage model)**,
**S3 (stateful trust tiers)** and **S4 (root-cause localization)**.
Full roadmap and scope decisions: [`PHASE2_PLAN.md`](PHASE2_PLAN.md).

---

## What this component does

Takes a distributed trace from a failed CI run and answers the question a
human on-call would have to answer: **is this worth retrying, or is something
actually broken?** It classifies the trace into five operational classes and
returns a policy decision plus the spans that justify it.

| Class | Meaning | Pipeline action |
|---|---|---|
| `PASS` | no failure signature | allow merge |
| `INFRA_FLAKE` | transient network/process fault | **retry** (within budget) |
| `DATA_FAULT` | corruption or loss — deterministic | escalate, never retry |
| `ENV_FAULT` | host-level failure | quarantine the node |
| `REAL_BUG` | genuine defect signature | block merge, alert developer |

The `INFRA_FLAKE` / `DATA_FAULT` split is the one that pays: both look like
failures, and retrying a corruption fault burns CI minutes and fails again.

---

## What changed from Phase 1, and why

Phase 1 trained on `data/raw/failure_trace.csv`. Each row there is `TaskID`
followed by exactly **2155 numbers**, and the CSV header is identical, in
order, to `eventId.json` — so every row is an **event-count vector**, not a
sequence. The old pipeline split that row on commas and fed the *counts* into
`nn.Embedding` as vocabulary IDs, then truncated to the first 100 columns.

That is why the reported **88.18%** validation accuracy was frozen across
epochs 3, 4 and 5: deciding from column 0 (`getFileInfo` count) alone scores
**≈88.2%** on the same balanced 59,634-row set. The network had converged to a
constant rule on one feature.

Phase 2 rebuilds the input from the raw TraceBench tables:

* **Real sequences.** Per task, ordered `OpName|status` tokens reconstructed
  from `event.csv`.
* **Clock alignment.** An event's `StartTime` is on *its own host's* clock, so
  raw sorting is meaningless across hosts. Offsets are recovered from the
  trace's own parent–child edges (`edge.csv` records the father's clock reading
  when it spawned each child) and propagated down the call tree. **100% of the
  81,777 traces align fully**, and the aligned order recovers the true root
  operation (`fs -copyFromLocal`) that as-recorded file order drops.
* **Honest labels.** Six classes from TraceBench's own fault taxonomy across
  all 363 scenario folders. `COM_Sin_Bug` contains *real HDFS defects*, which
  is what gives `REAL_BUG` a real definition — Phase 1 had no code-bug data at
  all yet reported "Code Bug" as its default class.
* **Corrected policy.** Phase 1 mapped every non-anomalous trace to
  `"Code Bug" → BLOCK_MERGE`, so healthy traces were merge-blocked. See
  `src/policy_engine/policy.py`.

Two deliberate anti-leakage choices, both tested:

1. **Grouped split.** Train/val/test are drawn over whole *scenario folders*,
   never over traces. Traces from one scenario share a fault, a cluster size
   and a workload.
2. **`title` is not a model input.** The per-task Title (`fs -mkdir`) records
   the client workload, and folder names encode workload alongside fault, so
   feeding it lets the model identify the scenario instead of the failure.

---

## Layout

```
src/data/build_sequences.py   TraceBench -> clock-aligned labelled sequences
src/data/dataset.py           vocab, grouped split, tensorisation, bucketing
src/features/triage_lstm.py   attention-pooled BiLSTM (packed sequences)
src/models/train.py           seeded training, early stopping, config logging
src/models/evaluate.py        per-class metrics + four baselines
src/models/predict.py         serving wrapper, returns evidence spans
src/policy_engine/policy.py   class + trust tier -> action (pure, unit-tested)
src/policy_engine/database.py trust tiers + S3 promotion/demotion state machine
src/policy_engine/rego/       triage_policy.rego - Rego mirror of policy.py
src/policy_engine/main.py     FastAPI triage + postmortem service
src/rca/localize.py           S4: per-op PASS-latency baseline, suspect-span ranking
scripts/run_rca_eval.py       builds the RCA baseline, top-k localisation vs. baselines
scripts/replay_trust_trajectory.py  replays the test set through S3, T0->T3, then a demotion demo
tests/test_triage.py          taxonomy, alignment, policy, leakage, serving
tests/test_trust.py           S3: Wilson bound, promotion, demotion
tests/test_rca.py             S4: suspect ranking, postmortem generation
tests/test_rego_policy.py     policy.py <-> triage_policy.rego equivalence (needs opa CLI)
```

---

## Running it

```bash
pip install -r requirements.txt

# 1. Build the corpus (resumable; ~2 min over all 363 scenarios)
python src/data/build_sequences.py \
    --tracebench "<path>/HDFS_v3_TraceBench/tracebench" \
    --out-dir data/processed --per-class 20000
python src/data/build_sequences.py \
    --tracebench "<path>/HDFS_v3_TraceBench/tracebench" \
    --out-dir data/processed --merge

# 2. Train
python src/models/train.py --epochs 20 --batch-size 256 \
    --max-len 256 --emb-dim 48 --hidden 64

# 3. Evaluate against baselines
python src/models/evaluate.py

# 4. Build the RCA baseline + evaluate root-cause localisation (S4)
python scripts/run_rca_eval.py

# 5. Replay the trust-tier state machine over the test set (S3 demo)
python scripts/replay_trust_trajectory.py

# 6. Serve
python run_api.py          # http://localhost:8000/docs

# 7. Test
pytest -q
```

`data/` is gitignored — rebuild the corpus with step 1 rather than committing
41 MB of derived data.

---

## API

`POST /api/v1/triage` — accepts either a pre-tokenised `trace_sequence` or raw
`spans` (the shape the Jaeger adapter will emit in S5), and returns the class,
probabilities, the policy action, the rule that fired, and the top attention
spans as evidence.

`POST /api/v1/triage/ground-truth` closes the loop: once CI knows what the
failure really was, that runs the S3 trust-tier state machine
(`src/policy_engine/database.py`) and returns whatever promotion or demotion
it triggered. `GET /api/v1/agent/scorecard` reports the agent's measured
accuracy and current tier.

Trust tiers are **agent autonomy levels** — T0 observe-only, T1 recommend,
T2 act-with-rollback, T3 act-autonomously — not developer seniority, which is
what Phase 1 stored. An agent below an action's tier can only recommend it.
Promotion requires the **Wilson lower-bound** of accuracy (not the raw
accuracy) to clear each tier's threshold over enough evidence; a REAL_BUG the
agent misclassified *and* acted on autonomously demotes it one tier
immediately, bypassing that statistic entirely. See design decisions 7–8 in
`CLAUDE.md`.

`POST /api/v1/postmortem` — accepts the same `spans`/`trace_sequence` shape
as `/triage`, and returns the top-N suspect spans ranked against a
PASS-class latency baseline (`src/rca/localize.py`), plus a ready-to-paste
Markdown postmortem. Requires `scripts/run_rca_eval.py` to have been run
once to build the baseline.
