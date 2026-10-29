# What is left for you

Everything that could be built and checked from a laptop with no cloud account, no
cluster and no GitHub repo is done. This file lists what is **not**, and why only
you can do it. Read section 0 first: it separates what was *verified* from what
was only *written*.

## 0. Verified vs. only written

| Piece | State | Evidence |
|---|---|---|
| Triage model, trust engine, RCA, policy, Rego mirror | **verified** | 150+ tests, reports/ |
| Jaeger adapter, drift guard, canary, `/metrics`, poller | **verified** against stub servers and a real local `run_api.py`; **also verified against a real Jaeger 1.76.0 (OTLP push from the live shop, then poller to agent API)** | `tests/test_observability.py`, `tests/test_integration.py` |
| Security gate | **verified** on this environment: 27 flagged → 0 reachable, with a positive control | `reports/security_gate.json`, `tests/test_security.py` |
| Chaos harness | **verified** offline on synthetic injections; **no real fault was injected** into a live system | `reports/chaos_report.md` |
| Actuator | **verified** with faked GitHub HTTP; **never sent a real request** | `tests/test_actuator.py` |
| GitHub workflows | YAML parses and has the right structure; **never executed on GitHub** | `.github/workflows/` |
| `Dockerfile`, `infra/docker-compose.yml`, Grafana dashboard | **written only — no Docker run was possible** where this was built | `infra/` |
| Train-on-your-own-system route | **verified** end to end on synthetic traces | `tests/test_live_corpus.py` |

## 1. Do these, in this order

### 1.1 Push to GitHub and watch the pipeline run once  (~1 h)

1. Create the repo and push. `data/` is git-ignored on purpose (3 GB). **Do commit**
   `src/models/saved/triage_lstm.pth`, `vocab.json`, `op_latency_baseline.json` and
   `reports/security_gate.json` — the CI gate and the Docker image need them.
   Run `git status` before the first commit and check nothing secret is staged.
2. `ai-gate.yml` uses the `workflow_run` trigger, which **only fires if the workflow
   file is on the default branch**. Merge the workflows to `main` first, then open a
   test PR.
3. Open a PR. Expect: `ci` runs → `ai-gate` runs when it finishes → a PR comment and
   an `ai-gate` commit status appear.
4. If `pip install` fails on the runner, the usual cause is the `torch==2.13.0` pin not
   existing for the runner's platform. Adjust the pin in `requirements*.txt`
   (this is the likeliest first-run failure; it was not testable here).
5. **Do not mark `ai-gate` as a required check yet.** With no persistent trust state the
   gate runs at `AI_GATE_TIER` (default 0 = recommend only), and a T0 agent can only
   ever answer "pause for a human" — a required check would fail every PR. Make it
   required only after you deploy the API (1.2) and the agent has earned T2.
6. Optional repo *variables* (Settings → Secrets and variables → Actions → Variables):
   `TRIAGE_API_URL`, `AI_GATE_TIER`, `QUARANTINE_CMD` (e.g. `kubectl cordon {node}`),
   `ROLLBACK_CMD` (e.g. `kubectl rollout undo deployment/{deployment}`).
   Node/deployment names are validated before they reach a command line.

### 1.2 Bring up the stack  (~1 h)

```bash
docker compose -f infra/docker-compose.yml up --build
```

* Jaeger is pinned to 1.76.0 (tested); still pin HotROD in `infra/docker-compose.yml` to
  whatever `docker pull` resolves, so the demo does not change under you.
* Check: API `localhost:8000/docs`, Jaeger `:16686`, HotROD `:8080`, Prometheus `:9090`
  (Status → Targets: `ai-gate` should be UP), Grafana `:3000` (admin/admin; the
  "AI release gate" dashboard is pre-provisioned — if panels are empty, run a few
  `/api/v1/gate` calls first).
* Click around HotROD, then:
  `python scripts/jaeger_poller.py --jaeger http://localhost:16686 --service frontend --once`
* **Expect every HotROD trace to come back `ood-abstain`.** That is correct behaviour:
  the shipped model has never seen those operation names. It is not a bug; show it.

### 1.3 Decide the model strategy for a live demo  (the biggest decision)

The shipped model understands **HDFS TraceBench traces only** (see
`reports/adapter_fidelity.json`: 74% on corpus records, 63% on schema-attested
exports, **20% on stock exports of the same traces**). A live TrainTicket/HotROD
demo therefore has two honest options:

* **A. Demo the live plumbing, with abstention.** Everything (Jaeger → agent → gate →
  PR comment → Grafana) runs; the classification step honestly says "not my system".
  Use `samples/jaeger_hdfs_*.json` to show the in-distribution path.
* **B. Train a model on your live system's traces.** Real work, but it is the only way
  to a working classifier there. **A ready-made, runnable version of this is the minimal live
  project in [`LIVE_PROJECT.md`](LIVE_PROJECT.md)** (a 4-service toy app with injectable faults;
  steps 1-5 need only Python and take ~45 min). Use it as the live demo, and use the recipe below
  only if you want the same thing on a real application such as TrainTicket:
  1. Deploy TrainTicket (or another app) with Jaeger.
  2. Generate steady traffic (any load script).
  3. Run **labelled experiments**: steady state → `PASS`; kill/slow/partition a service →
     `INFRA_FLAKE`; corrupt data → `DATA_FAULT`; break a node → `ENV_FAULT`. Do each
     **several independent times** (≥ 3 scenarios per class, tens of traces each) —
     the train/val/test split is by scenario, so repeats of one run don't count.
  4. Export each experiment's traces from Jaeger
     (`curl "$JAEGER/api/traces?service=X&start=…&end=…&limit=1000" > exports/run1.json`)
     and write `experiments/manifest.json` (format: `src/data/jaeger_corpus.py`).
  5. `python scripts/jaeger_to_corpus.py --manifest experiments/manifest.json`
  6. Train and serve — commands are in `scripts/jaeger_to_corpus.py`'s docstring.
     A model trained this way is trusted for adapter-made traces automatically
     (train/serve feature parity) and still abstains on unseen vocabulary.
  7. Report its metrics with the same discipline as `reports/evaluation.json`
     (per class, grouped split, baselines). Expect it to look different from 0.746.

### 1.4 Real chaos engineering  (needs a live system)

`scripts/run_chaos.py` is the offline half. The live half — Chaos Mesh / LitmusChaos on
Kubernetes, or simply `docker kill`, `docker pause`, `tc netem` on a compose stack —
is yours. Its natural role is step 3 of 1.3 (producing *labelled* faulty traces) and
then closing the loop: feed the live faults through `/api/v1/gate`, report the true
outcome to `/api/v1/triage/ground-truth`, and screenshot the Grafana tier gauge
reacting. That screenshot is the review evidence for the "chaos → wrong decision →
demotion" claim on a real system; the offline version is in `reports/chaos_report.md`.

### 1.5 Deploy the API with persistent trust state

The trust tier lives in SQLite. In Docker it is on the `/state` volume
(`TRUST_DB_PATH`); `TRUST_RETENTION=1` is set so an agent whose accuracy collapses is
demoted. Real ground truth has to be fed back (`/api/v1/triage/ground-truth`) for
tiers to move — wire that into whatever knows the true outcome of a run.

### 1.6 Clean up the security findings (real, small)

`reports/security_gate.json`: 27 vulnerabilities, none reachable from the service.
They are still worth fixing: `python -m pip install -U pip` (12 findings in pip
25.0.1) and update the Jupyter packages. pip-audit gives neither severity nor the
vulnerable function; add those per advisory in a JSON file (see
`samples/advisories_example.json`) and pass `--advisories` to let the gate narrow
"imported" down to "actually called".

### 1.7 Update the review material

The Review 2 report/deck predate S5–S8 and the findings below. I did not regenerate
them. The points worth adding: the adapter-fidelity table, the corrected attention
numbers, the chaos table (with its control row), and the security 27→0 result.

**One correction is mandatory:** Figure 3 (trust trajectory, zoom panel) in the Review 2
report and deck was drawn before the hysteresis fix (`CLAUDE.md` decision 16). Its zoom panel
shows the agent demoted and immediately re-promoted - which the text calls "immediate
demotions" - because the old engine recycled old evidence. Replace it with the regenerated
`reports/trust_trajectory.png`; the headline T0→T3 climb is unchanged.

## 2. Limitations to state plainly in the review

* **Domain.** Valid only for traces in the training schema. The drift guard checks
  *vocabulary* and *feature schema*; it cannot tell you the model is right inside its
  domain. In-domain accuracy is 76.9% (macro-F1 0.746).
* **False alarms.** PASS recall is ~0.60: roughly 40% of healthy traces are flagged.
  The chaos report's `control` row shows it; every other row must beat it.
* **REAL_BUG confound.** REAL_BUG is recognised partly by being a short trace. On
  stock exports this made 90% of traces come back REAL_BUG — i.e. a wrongly *blocked*
  merge. The schema guard exists because of this.
* **Chaos is synthetic.** Injected faults copy signatures seen in real fault traces.
  `silent_regression` has no trace signature by construction.
* **Security analysis is static.** Dynamic imports and optional dependencies that a
  package loads on its own are invisible to it.
* **Untested on real infrastructure:** workflows, Docker, Jaeger, GitHub API calls.

## 3. Command cheat sheet

```bash
pytest -q                                   # everything (some tests skip without the 3 GB corpus)
python scripts/demo_review2.py              # Review 2 demo
python scripts/demo_final.py                # S5-S8 demo
python scripts/run_security_gate.py         # CVE reachability (needs reports/pip_audit_raw.json or network)
python scripts/run_chaos.py [--sweep]       # chaos experiments
python scripts/ci_gate.py --jaeger samples/jaeger_hdfs_data_fault.json --tier 3   # dry run
python scripts/adapter_fidelity.py          # what the Jaeger route costs
python scripts/attention_stats.py           # attention concentration
python scripts/bootstrap.py --check         # artifact status; --tracebench <path> rebuilds
```
