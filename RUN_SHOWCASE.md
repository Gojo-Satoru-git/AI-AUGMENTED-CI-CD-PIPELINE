# Run the showcase

## A. The live dashboard (the demo to show)

```bash
python scripts/serve_ui.py
```

Opens http://127.0.0.1:8765, a control room for a running system. The strip across the top is the whole
pipeline (collect, adapter, drift guard, triage, RCA, trust, security, canary, gate, actuator, metrics) with
each stage's live status; click a stage to jump to its tab.

| Tab | Module shown | Try this |
|---|---|---|
| Live operations | traffic, triage, trust tiers, actuator | fault buttons, **Run scripted demo**, **Chaos monkey**, the security-scan toggle |
| Release gate & canary | canary health, release gate, PR comment | deploy a "new version": healthy v2 is promoted, slow db / flaky network is rolled back |
| Root cause | RCA suspect spans, auto postmortem | inject a fault, open the tab: ranked suspect spans for the latest incident |
| Security | CVE reachability | 27 flagged, 0 reachable; re-run the analysis; toggle a simulated reachable CVE and watch the gate block |
| Real data (HDFS) | the HDFS-trained model, drift guard, baselines | run real fault traces; the "foreign app" trace makes the gate abstain |
| Chaos & trust | chaos sweep results, trust-engine replay | the unsafe-action and demotion evidence |
| Observability | Prometheus /metrics | the same numbers Grafana would scrape |

Suggested walk-through: Live (let it reach T2/T3, press flaky network, host down, then SILENT bug and watch it
lose autonomy) -> Root cause -> Release gate (rollback a bad canary) -> Security (toggle the CVE) ->
Real data (foreign trace abstains; baselines) -> Chaos.

Optional, to also see the traces in a real Jaeger UI:

```bash
docker run -d --name jaeger -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:1.76.0
python scripts/serve_ui.py --otlp http://127.0.0.1:4318/v1/traces
# Jaeger UI: http://127.0.0.1:16686, service "frontend".  Clean up: docker rm -f jaeger
```

## B. The one-shot terminal version

From the project folder, with the project's virtual environment active:

```bash
python scripts/showcase.py
```

About one minute. Add `--pause` to advance section by section with Enter while presenting.
The first run builds the live model if it is missing (~2 extra minutes).

It prints five sections:

1. **The system**: a 4-service shop; what a healthy, flaky, host-down and silent-bug trace look like.
2. **Credible numbers**: HDFS held-out macro-F1 0.746 against four baselines (Phase 1's 88.18% was a one-column rule).
3. **The live model**: and the caveat that the toy faults are easy.
4. **The closed loop**: 390 requests with faults injected on a schedule. The agent climbs T0 to T3 on evidence,
   is demoted when a defect with no trace signature slips through, and re-earns trust.
5. **Scorecard and limits**, stated by us before anyone asks.

Output is saved to `showcase_output/`: `transcript.txt`, `summary.md`, `live_trajectory.png` (put this in the deck).

Optional, to see the same traces in a real Jaeger UI:

```bash
docker run -d --name jaeger -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:1.76.0
python scripts/showcase.py --otlp http://127.0.0.1:4318/v1/traces
# open http://127.0.0.1:16686, pick service "frontend"
docker rm -f jaeger
```

Exact numbers vary by a few requests between runs (real sleeps and timing); the shape does not.
