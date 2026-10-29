# The minimal live project - step by step

**Goal:** a small system that actually *runs*, breaks on purpose, and is watched by the
agent, so the professors see the whole loop - not a model scoring a dataset:

```
minishop (4 services)  ->  traces  ->  Jaeger adapter  ->  triage model  ->  release gate
        ^                                                                          |
        |  inject faults on a schedule                         decision (+ dry-run action)
        |                                                                          v
        +---------------  true outcome fed back  ->  trust tier moves  ->  Grafana / metrics
```

**Time:** about 45 minutes for steps 1-5 (everything runs on your laptop with no Docker);
add 1 hour for step 6 (Jaeger + Grafana). Rehearse step 8 twice.

**Why a toy app and not TrainTicket?** Every fault here is injected by us, so every trace is
labelled *exactly* (see "What is and isn't convincing" below), an experiment takes
seconds instead of an afternoon, and it needs no cluster. The same code path serves a
bigger system: replace `src/live/minishop.py` with your traced app and keep the rest.

---

## Step 0 - check the machine (2 min)

```bash
cd AI-CI-CD-Pipeline
./venv/Scripts/python.exe -m pytest -q          # expect: ~180 passed, 12 skipped
```

Use **`127.0.0.1`, not `localhost`**, in any URL you give a script. On Windows `localhost` tries
IPv6 first and costs ~2 s per call (measured: 2,111 ms vs 26 ms) - it makes a live demo
look broken.

## Step 1 - look at the system before any model (5 min)

```bash
python scripts/live_peek.py
```

Prints one trace per fault. This is the slide to start the demo on. Point at:

| fault | what the trace shows | class |
|---|---|---|
| `flaky_net` | `db.insert` fails twice (IOEXC), third attempt succeeds | INFRA_FLAKE - a retry would have worked |
| `slow_db` | same shape, db spans 2-50x longer | INFRA_FLAKE |
| `db_corruption` | `db.insert` fails *once* with a checksum error, **not retried** | DATA_FAULT - deterministic, retrying is pointless |
| `host_down` | three refused connections in a row, request fails | ENV_FAULT - the node is the problem |
| `code_bug` | `orders.validate` raises before any db call | REAL_BUG |
| `silent_bug` | **identical to a healthy trace** | REAL_BUG by label - invisible to any trace-based model |

The distinctions (retry succeeds vs. fails identically vs. never reaches the db) are exactly
why the policy treats these classes differently.

## Step 2 - collect labelled experiments (3 min)

```bash
python scripts/live_collect.py
```

Runs 37 independent experiments x 50 traces (about 1,850 traces) into `live_data/` with a
`manifest.json`. Each trace is labelled by what *actually happened to that request*: a request
that never touched the faulty path is PASS even inside a faulty experiment.

Why 37 experiments and not 5: the train/val/test split is **by experiment**, and a fault
class has sub-types (slow db vs flaky network). With too few runs a split can hold a sub-type
the model never trained on - we measured the LSTM scoring **0.0 F1** on INFRA_FLAKE
that way. More independent runs fixed it.

## Step 3 - build the corpus and train (5 min)

```bash
python scripts/jaeger_to_corpus.py --manifest live_data/manifest.json \
       --out data/processed/corpus_live.jsonl.gz

python src/models/train.py --corpus data/processed/corpus_live.jsonl.gz \
       --out src/models/saved_live --reports reports/live \
       --feature-schema jaeger_adapter_v1 --weight-power 0.5 \
       --epochs 40 --patience 8 --batch-size 32 --lr 0.005 --hidden 32 --emb-dim 24 --max-len 32
```

`--feature-schema jaeger_adapter_v1` records that the model was trained on exactly the
features the adapter computes at serving time, so live traces are trusted automatically
(no train/serve skew). Don't drop it. `--out` and `--reports` point at *new* folders so the
shipped HDFS model and its reports are never overwritten.

## Step 4 - evaluate honestly (2 min)

```bash
python src/models/evaluate.py --corpus data/processed/corpus_live.jsonl.gz \
       --ckpt src/models/saved_live/triage_lstm.pth --vocab src/models/saved_live/vocab.json \
       --reports reports/live
```

Expect (seeded; yours should match closely): test macro-F1 ~1.00 on ~300 held-out traces from
experiments the model never saw. **Say the next sentence yourself before anyone asks:**
the baselines also score high - scalar-GBM 1.00, bag-of-ops logistic regression ~0.97 - so
this toy is too easy to rank models on. What this step shows is that the collect -> train ->
serve route works end to end, not that the LSTM is good. The credible accuracy number for the
project remains the HDFS one: macro-F1 0.746 on real TraceBench traces.

## Step 5 - run the closed loop (1 min; use `--pause 0.1` to narrate)

```bash
python scripts/live_run.py                 # ~25 s
python scripts/live_run.py --pause 0.1     # paced for a room (~1 min)
```

Walk through the output top to bottom (390 requests; the schedule is steady -> slow db ->
flaky network -> corruption -> dead host -> code bug -> *unseen* bug -> *silent* bug):

1. **Earns trust.** Starts at T0 (observe only). Promoted at ~step 15 (T1), ~45 (T2), ~97 (T3),
   each time because the Wilson lower bound of *its own accuracy since the last change* cleared
   the bar. Nobody set the tier.
2. **Acts only when allowed.** Below T2 it answers `PAUSE_FOR_MANUAL_REVIEW` even when right.
3. **Unseen bug (`novel_bug`).** It still says REAL_BUG at ~0.93, although the failing operation
   was never in training. Be honest about why: the bug aborts early, producing a *short trace*,
   which is the shape it learned for REAL_BUG - the same confound found on HDFS. Don't claim
   it "understood" the bug.
4. **Silent bug.** The defect leaves no trace signature, so the agent says PASS and merges
   (`unsafe-merge`). Ground truth arrives, and the trust engine takes autonomy away: T3 -> T2 -> T1
   within two bad merges, then T1 -> T0 as failures continue. When the system goes back to steady
   state it climbs T0 -> T1 again, but only after 15 *fresh* correct decisions.
5. **Why that last point matters** (a real bug we found *because of this loop*): the first version
   promoted the agent straight back after every demotion, by re-using the old 200-decision
   history that had earned the tier - 53 tier flips in 50 decisions. Evidence is now counted only
   since the last tier change. Tell the professors this; it is evidence of testing, not weakness.

Result is saved to `reports/live/live_run.json`.

## Step 6 - add Jaeger and Grafana (about 1 hour; needs Docker Desktop running)

Docker is installed on this machine and the daemon was running when checked; the compose
file validates (`docker compose config`). **The stack itself has never been started** - the images
(Jaeger, HotROD, Prometheus, Grafana, roughly 0.5-1 GB) were not pulled, so treat this step as
untested and budget time for it.

```bash
# 1. pin the two `latest` image tags in infra/docker-compose.yml first (see the TODOs)
docker compose -f infra/docker-compose.yml -f infra/docker-compose.live.yml up --build

# 2. in another terminal: traces go to Jaeger AND through the agent via the API
python scripts/live_run.py --api http://127.0.0.1:8000 --agent-id triage_agent \
       --otlp http://127.0.0.1:4318/v1/traces --pause 0.1
```

Then show: Jaeger UI `http://localhost:16686` (service `frontend` - the same four-service trees
from step 1, live), Grafana `http://localhost:3000` (admin/admin -> "AI release gate": tier gauge
climbing then falling), Prometheus `http://localhost:9090/targets` (`ai-gate` UP).

If Jaeger rejects the push, the run prints `Jaeger push: FAILED (...)` and **carries on** - the
agent and Grafana still work, only the Jaeger UI would be empty. The OTLP payload is checked
structurally by `tests/test_live.py` but has never been sent to a real Jaeger.
Check `docker compose logs jaeger`; the usual causes are the tag (use a version with OTLP
enabled, v1.35+) and the port (4318).

## Step 7 - show it gating a pull request (30 min; see `USER_TODO.md` 1.1)

`python scripts/ci_gate.py --jaeger samples/jaeger_hdfs_data_fault.json --tier 3` prints the PR
comment locally with no GitHub at all. To show it on a real PR, push to GitHub, merge the
workflows to `main`, open a PR. Don't present `ai-gate` as a *required* check until the agent
has earned T2.

## Step 8 - the 10-minute demo script

| min | do | say |
|---|---|---|
| 0-1 | slide: the Phase 1 story (88.18% was a one-column rule) | why every number here comes with a baseline |
| 1-3 | `live_peek.py` | what a trace is; why retry-vs-escalate differs by fault |
| 3-4 | `live_collect.py` output + `evaluate.py` result | exact labels; and the "too easy" caveat, unprompted |
| 4-8 | `live_run.py --pause 0.1` | trust is earned, acts only when allowed, loses it on a missed bug |
| 8-9 | Grafana tier gauge (or `reports/live/live_run.json`) | the same thing as a dashboard |
| 9-10 | slide: limits (below) | what is verified and what is not |

**Fallbacks:** no Docker -> skip step 6 (steps 1-5 need only Python). Model missing -> rerun
steps 2-3 (about 6 min). Everything failing -> show `reports/live/live_run.json` and
`reports/trust_trajectory.png`.

## What is and isn't convincing

**Convincing:** a running system with four services; traces with real retry/backoff structure;
labels that are exact; held-out *experiments* (not traces); an agent that earns, uses and loses
autonomy from its own measured record; a bug found in our own trust engine by running it.

**Not convincing - say so first:**
- It is a toy: every fault is injected by us and the problem is easy (scalar baselines hit 100%).
  The live loop shows the *system*, not the model's quality.
- `silent_bug` is invisible by construction; `novel_bug` is caught by trace *shape*, not insight.
- Model quality on real traces is the HDFS result (macro-F1 0.746; ~40% of healthy traces flagged).
- Jaeger push, Grafana, Docker and GitHub workflows are written but were not run on this machine.
- Transfer to another real system needs its own labelled experiments (steps 2-4 are that recipe).

## Likely questions, with honest answers

- *"Why is accuracy ~100%?"* The faults are simple and injected; scalar baselines get 100% too.
  The point of this system is the closed loop. The real accuracy is the HDFS number.
- *"Why an LSTM if a baseline is as good?"* On both HDFS and here, order-free baselines match it:
  order carries little signal for this classification. The sequence pipeline is still needed for
  root-cause localisation. We report that finding rather than hide it.
- *"Does it work on a different system?"* Not without retraining - measured: 20% on stock exports
  of its own training traces. That is why the gate abstains out-of-distribution and why
  `jaeger_to_corpus.py` exists.
- *"How do you know the trust tiers work?"* Tests, a 10,513-decision replay, the chaos harness,
  and this loop - which found and fixed a flapping bug in them.
- *"What stops it doing something dangerous?"* It decides; a separate actuator executes, dry-run by
  default, with names from traces validated before reaching a command line; below T2 it can only
  recommend; out-of-distribution traces and low confidence go to a human.
