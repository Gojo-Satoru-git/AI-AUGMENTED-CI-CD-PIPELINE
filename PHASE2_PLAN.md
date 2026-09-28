# Phase 2 Plan — AI-Augmented CI/CD Pipeline (Team 18)

Audited 02 Sep 2026 against commit `46a0666` (+9 uncommitted files).

## 1. Phase 1 status

2 of 7 architecture blocks started (~15% of zeroth-review scope). 0 passing tests, 0 reported metrics.

### BLOCKING FINDING — the training data is not sequential

Every row of `data/raw/failure_trace.csv` and `normal_trace.csv` is `TaskID` + exactly **2155**
numeric values. The CSV header is identical, in order, to `eventId.json`, so column *i* is
**the number of times event type *i* occurred** in that task. Values are counts (0-48).

`preprocess_data.py` splits that row on commas and feeds the *counts* into `nn.Embedding`
as vocabulary token IDs, then truncates to the first 100 columns — discarding 95.4% of the
feature space. `vocab_size=49` is literally "max count + 1".

Consequence: the reported **88.18%** val accuracy (frozen identically across epochs 3/4/5)
equals the trivial rule "decide from column 0 (`getFileInfo` count) alone", which scores
**~88.2%** on the same balanced 59,634-row set. The network converged to a constant
decision on the first column. Do not present 88% as a result.

### Other defects

| Sev | Defect |
|---|---|
| BLOCK | Inverted policy semantics: label 1 = HDFS anomaly, label 0 = healthy; `main.py` maps `P>0.80 -> Flaky` else `Code Bug -> BLOCK_MERGE`. Healthy traces get merge-blocked. No code-bug class exists in training data. |
| BLOCK | No evaluation at all — no held-out set, no P/R/F1/PR-AUC, no confusion matrix persisted. |
| HIGH | Trust model is a static 3-row SQLite lookup, not the stateful T0-T3 promotion model promised. No OPA/Rego. |
| HIGH | RCA `FALLBACK_DATA_DIR` hardcoded to one scenario (~345 TaskIDs); draws a spring layout, never localises a root cause. |
| HIGH | `tests/test_preprocess_and_api.py` imports `src.models.main` (does not exist -> `src.policy_engine.main`). Suite has never passed. |
| MED | `vocab.json` is 0 bytes; `venv/` (Windows) committed; `requirements.txt` unpinned; no README; 2.4 GB `data/` gitignored with no fetch script. |

## 2. Locked decisions

1. **Model** — rebuild on real sequences from TraceBench `event.csv`; keep count-vector model as a baseline.
2. **Scope** — Triage, Policy Engine, Postmortem/RCA deep; Security, Observability, Chaos demonstrated.
3. **Telemetry** — train/evaluate on HDFS TraceBench; demo live on TrainTicket + Jaeger.

## 3. Dataset — 363 scenarios already on disk, only 3 in the repo

`dataset/HDFS_v3_TraceBench/tracebench/` holds 363 labelled scenario folders, 17 injected
faults, >370,000 traces. `COM_Sin_Bug` contains **real system bugs** — a genuine code-bug class.

| Class | Families | Folders | Pipeline meaning |
|---|---|---|---|
| PASS | NM_CL, NM_DN | 88 | Green build |
| INFRA_FLAKE | AN_Net_slowHDFS/slowDN/disconnectDN, AN_Proc_killDN/suspendDN | 147 | Retry eligible |
| DATA_FAULT | AN_Data_corruptBlk/corruptMeta/cutBlk/cutMeta/lossBlk/lossMeta | 85 | Escalate, never retry |
| ENV_FAULT | AN_Sys_panicDN/deadDN/readOnlyDN | 25 | Quarantine node |
| REAL_BUG | COM_Sin_Bug | 3 | Block merge, alert dev |
| COMPOUND | COM_Sin_{Sys,Proc,Net,Data}, COM_Mul_AA | 15 | Held out — generalisation |

Imbalance: sample at trace level, split by **whole folder** (no scenario in two splits),
report per-class metrics.

## 4. Workstreams

- **S0 Repo hygiene** (all, 3d) — drop `venv/`, pin requirements, fix the broken test import,
  `scripts/fetch_data.py`, README, commit the 9 dirty files.
  *Exit:* fresh clone -> install -> fetch -> `pytest` green on another machine.

- **S1 Data foundation** (Praveen, 2w) — `src/data/build_sequences.py`: per TaskID, sort
  `event.csv` by `StartTime`, emit ordered `OpName` tokens with status parsed from
  `Description` (`getFileInfo:OK`, `RPC:addBlock:ERR`). Add span duration, DAG depth from
  `edge.csv`, host/agent; task features from `trace.csv` (`Title`, latency, NumReports,
  NumEdges). Label by folder. Group-split by folder. Real `vocab.json`.
  *Exit:* versioned variable-length labelled corpus + justified `MAX_SEQ_LEN` (95th pct).

- **S2 Triage model** (Praveen, 2w) — bidirectional LSTM with `pack_padded_sequence`,
  5-way softmax, class-weighted loss, numeric features concatenated. `train.py` + `evaluate.py`
  writing per-class P/R/F1, macro-F1, PR-AUC, confusion matrix to `reports/`. Saliency over spans.
  Baselines: majority, column-0 rule (88.2%), LR/GBM on full 2155 counts, bag-of-ops.
  *Exit:* beats all four on macro-F1, numbers committed. A negative result is publishable — report it.

- **S3 Policy Engine** (Jaiprakash, 2w) — tiers become **agent** autonomy (T0 observe,
  T1 recommend, T2 act-with-rollback, T3 autonomous), not developer seniority. Tables:
  `decisions`, `tier_history`. Promotion on Wilson lower bound of correctness over last N;
  immediate demotion on a false negative reaching production. Decision matrix in **Rego**,
  evaluated by OPA. Fix the inversion: REAL_BUG blocks, INFRA_FLAKE retries within budget,
  DATA_FAULT/ENV_FAULT escalate, PASS proceeds.
  *Exit:* agent provably climbs T0 -> T3 over a replayed stream; demotion demonstrated.

- **S4 Postmortem & RCA** (Praveen + Jaiprakash, 2w) — remove the hardcoded folder; score
  nodes against the normal-scenario baseline for their operation (latency z-score, error flag,
  subtree impact); return a ranked top-k suspect list; aggregate per host to localise the faulty
  instance. Generate a Markdown postmortem PR body.
  *Exit:* top-1/3/5 localisation accuracy per fault family vs random-node and slowest-node baselines.

- **S5 Observability / live stack** (Dharshini, 3w, starts week 1) — `infra/docker-compose.yml`
  with TrainTicket (fallback: Online Boutique) + Jaeger + Prometheus + Grafana; `infra/k8s/`;
  OTel collector. `src/observability/jaeger_adapter.py` emits **the same schema S1 produces** —
  design it against S1's format from day one. Canary health score. Grafana dashboard.
  *Exit:* click in TrainTicket -> trace in Jaeger -> well-formed sequence at the triage API.

- **S6 Security Gate** (Jaiprakash, 1.5w) — pip-audit/Trivy -> SARIF; CodeQL/Semgrep call graph;
  `src/security/reachability.py` returns reachable/unreachable/unknown; verdict feeds the same OPA policy.
  *Exit:* ~10 real CVEs, flagged vs actually reachable, one false-positive-reduction figure.

- **S7 Chaos** (Dharshini, 1.5w) — Chaos Mesh/Litmus (fallback `tc netem` + container kills).
  Experiments mirroring TraceBench fault families so live faults are comparable to trained classes.
  Log every experiment as a labelled decision into S3's `decisions` table.
  *Exit:* chaos provokes a wrong decision and the trust engine demotes the agent.

- **S8 Actuator + CI** (all, 1w) — `.github/workflows/ai-gate.yml`; actuator honouring
  allow / retry-with-budget / constrain(canary) / deny; canary rollback on S5's score;
  PR status check carrying class, confidence, tier and the rule that fired. Replace
  `simulate_triage.py` with an integration test. Decision engine never executes; actuator never decides.
  *Exit:* a flaky test opens a PR, the pipeline retries autonomously, reasoning posted to the PR.

## 5. Sequencing (14 weeks)

S1 must finish before S2 starts. S5 runs in parallel from week 1 (Docker/K8s is wall-clock time).
Suggested review points: R1 ~W3, R2 ~W8, R3 ~W12, Final ~W14 — adjust to actual review dates.

## 6. Review evidence

- **R1** — the count-vector finding, the 88.18% ~ 88.2% baseline collision, new 5-class corpus + split stats.
- **R2** — per-class P/R/F1 + confusion matrix vs four baselines; trust trajectory T0 -> T3.
- **R3** — top-k localisation accuracy; a generated postmortem PR body; live Jaeger trace at the API.
- **Final** — chaos -> misclassification -> automatic demotion; end-to-end PR gate on a real failing test.

## 6b. Status update and scope calls (post-S2)

S1–S4 are now complete. S5–S8 were deliberately re-scoped down from the original plan rather
than attempted at full weight — the original scope (K8s + TrainTicket + Prometheus/Grafana,
CodeQL/Semgrep, Chaos Mesh/LitmusChaos) is real infrastructure-engineering effort largely
orthogonal to this project's actual contribution (the triage model + trust engine + RCA), and
high-risk to demo live. Each substitute below is the same *idea* at a fraction of the
infrastructure cost — not a silent cut, stated here explicitly.

| Workstream | Original scope | What was actually built / planned instead | Why |
|---|---|---|---|
| S3 Trust engine | OPA server, Redis/Postgres | Wilson-lower-bound promotion + immediate demotion in SQLite (`src/policy_engine/database.py`); a literal `.rego` mirror of `policy.py` for the "Rego" deliverable, checked for equivalence by `tests/test_rego_policy.py` (skipped without the `opa` CLI — not installed here, see that file's docstring) | The statistics are the actual contribution; a live OPA server adds ops surface for no additional review credit |
| S4 RCA | (as planned) | Implemented per spec, minus the subtree-impact term (edge lists aren't preserved in the built corpus — see `src/rca/localize.py`'s note); z-score + error-status ranking, evaluated vs. random and slowest-span baselines | On-plan; the one honest gap is documented in-code |
| S5 Observability | TrainTicket + Jaeger + Prometheus + Grafana + K8s | Not yet built. Recommended: a Jaeger/OTel-schema adapter (spans -> the same dict `predict.py` already accepts), demoed by replaying exported trace JSON — skip TrainTicket/K8s/Prometheus/Grafana entirely | The schema-conversion idea is the real deliverable; standing up a microservices+K8s stack is the single highest demo-failure risk for the least review credit |
| S6 Security gate | CodeQL/Semgrep call-graph | Not yet built. Recommended: `pip-audit` for CVE listing + a small Python `ast`-based import-graph reachability check | Gets a real "N CVEs, flagged vs. reachable" result without integrating a heavyweight external tool |
| S7 Chaos | Chaos Mesh / LitmusChaos on K8s | Not yet built. Recommended: a plain Python fault-injection script feeding synthetic decisions into `database.py`'s demotion path (the mechanism `scripts/replay_trust_trajectory.py`'s Phase 2 already demonstrates against real ground truth) | Chaos Mesh needs a real cluster to mean anything; the narrative ("chaos -> wrong decision -> demotion") is already provable without one |
| S8 Actuator + CI | GitHub Actions + Kubernetes | Not yet built. Recommended: one `ai-gate.yml` calling `/api/v1/triage` via curl and posting the decision as a PR comment on a toy repo | Real and screenshot-able without deploying anything |

## 7. Risks

- Sequence build too slow -> start with ~60 stratified folders, cache to Parquet.
- Order may not matter -> negative result is publishable; shift weight to S3/S4.
- TrainTicket may not fit the hardware -> decide by W3, Online Boutique is a fine substitute.
- Scope creep across 7 boxes with 3 people -> hold the deep/demonstrated line at S6/S7.
- REAL_BUG has only 3 folders -> per-class CIs; COMPOUND stays held out, not training padding.
