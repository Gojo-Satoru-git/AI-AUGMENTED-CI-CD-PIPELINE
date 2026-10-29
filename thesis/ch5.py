import json
from pathlib import Path

from blocks import *

HERE = Path(__file__).resolve().parent
LIVE = json.loads((HERE / "live_state.json").read_text(encoding="utf-8"))
SCHEDULE = [("none", 40), ("slow_db", 30), ("none", 20), ("flaky_net", 30), ("none", 20),
            ("db_corruption", 30), ("none", 20), ("host_down", 30), ("none", 20), ("code_bug", 30),
            ("none", 20), ("novel_bug", 30), ("none", 20), ("silent_bug", 30), ("none", 20)]
NAMES = {"none": "Steady (healthy)", "slow_db": "Slow database", "flaky_net": "Flaky network",
         "db_corruption": "Data corruption", "host_down": "Host down", "code_bug": "Code bug",
         "novel_bug": "Unseen bug", "silent_bug": "Silent bug"}
ORDER = ["none", "slow_db", "flaky_net", "db_corruption", "host_down", "code_bug", "novel_bug", "silent_bug"]
TRUTH = {"none": "PASS", "slow_db": "INFRA_FLAKE", "flaky_net": "INFRA_FLAKE", "db_corruption": "DATA_FAULT",
         "host_down": "ENV_FAULT", "code_bug": "REAL_BUG", "novel_bug": "REAL_BUG", "silent_bug": "REAL_BUG"}
agg, i = {k: [0, 0] for k in ORDER}, 0
for kind, n in SCHEDULE:
    seg = LIVE["series"][i:i + n]
    agg[kind][0] += n
    agg[kind][1] += sum(s[2] for s in seg)
    i += n
LIVE_ROWS = [[NAMES[k], TRUTH[k], str(agg[k][0]), str(agg[k][1]), f"{agg[k][1] / agg[k][0]:.0%}"] for k in ORDER]
tot_n = sum(v[0] for v in agg.values())
tot_c = sum(v[1] for v in agg.values())
LIVE_ROWS.append(["All phases", "", str(tot_n), str(tot_c), f"{tot_c / tot_n:.1%}"])
EV_ROWS = [[str(e["step"]), ("Promoted" if e["action"] == "promoted" else "Demoted"), f"T{e['from']} to T{e['to']}",
            {"promoted": "Wilson lower bound reached the threshold on fresh evidence",
             "demoted": "Missed real defect acted on autonomously" if "REAL_BUG" in e["reason"]
             else "Retention floor failed (accuracy collapse)"}[e["action"]]] for e in LIVE["events"]]
LIVE_ACC = f"{tot_c / tot_n:.1%}"

BLOCKS = [
    CH("RESULTS AND DISCUSSION"),
    H2("System Requirements"),
    H3("Software Requirements"),
    P("The software is written in Python and every reported number was produced with the versions in "
      "{T:software}. The model is implemented in PyTorch {R:pytorch}, the baselines use scikit-learn "
      "{R:sklearn}, the service uses FastAPI, and Jaeger {R:jaeger} runs in a container when a real trace "
      "back end is wanted. A Rego mirror of the policy for the Open Policy Agent {R:opa} is included and "
      "checked against the Python policy by a test, but that test was skipped here because the opa "
      "executable was not installed. Metrics are exposed in the Prometheus text format {R:prom}."),
    TAB("software", "Software requirements of the system",
        ["Software", "Version", "Purpose"],
        [["Python", "3.13", "Implementation language"],
         ["PyTorch (CPU build)", "2.13.0", "Triage model, training"],
         ["NumPy / pandas", "2.5.1 / 3.0.5", "Data handling"],
         ["scikit-learn", "1.9.0", "Baselines, metrics"],
         ["FastAPI / uvicorn / pydantic", "0.141.1 / 0.52.1 / 2.13.4", "Service and dashboard"],
         ["NetworkX", "3.6.1", "Call-graph handling"],
         ["pytest / pip-audit", "9.1.1 / 2.10.1", "Tests; vulnerability report"],
         ["Jaeger all-in-one (Docker)", "1.76.0", "Real trace back end (optional)"]],
        [5.6, 4.6, 5.9]),
    H3("Hardware Requirements"),
    P("All training and measurement were done on one Windows 11 laptop without a GPU. Training the HDFS "
      "model took 1,427 seconds on the CPU. The corpus build over the 3.0 GB data set is resumable and "
      "takes a few minutes. The dashboard and the live loop run comfortably on the same machine alongside a "
      "Jaeger container."),

    H2("Experimental Setup and Evaluation Methodology"),
    P("The HDFS corpus is split by scenario folder into training, validation and test sets as shown in "
      "{T:split}, so that a fault seen in training is never seen in testing. The test set therefore measures "
      "generalisation to unseen experiments. Macro-averaged F1 is reported because the classes are "
      "unbalanced, and a second figure that excludes REAL_BUG is reported for a reason given in "
      "Section 5.3. Every new claim is compared with at least one baseline that uses the same inputs "
      "or fewer. The five baselines are a majority-class rule (B1), a gradient-boosted model {R:friedman} "
      "on task-level scalars (B2), logistic regression on bag-of-operations counts, which is the "
      "representation of Phase 1 (B3), a support-vector machine on TF-IDF-weighted operation counts (B4), and "
      "gradient boosting on scalars plus operation counts (B5). B5 uses everything the BiLSTM uses except "
      "order."),
    TAB("split", "Scenario-grouped split of the HDFS corpus",
        ["Split", "Scenarios", "Traces", "PASS", "INFRA_FLAKE", "DATA_FAULT", "ENV_FAULT", "REAL_BUG"],
        [["Train", "234", "40,967", "13,064", "10,112", "13,924", "2,573", "1,294"],
         ["Validation", "57", "10,287", "3,096", "2,273", "3,068", "691", "1,159"],
         ["Test", "57", "10,513", "2,964", "2,319", "3,068", "832", "1,330"]],
        [1.8, 1.6, 1.7, 1.5, 2.7, 2.5, 2.2, 2.1]),

    H2("Results for the Triage Model"),
    P("{T:baselines} and {F:baselines} compare the model with the baselines on the 10,513 test traces. "
      "The attention BiLSTM reaches a macro-F1 of 0.7464 and an accuracy of 0.7690, which is 0.043 above the "
      "strongest baseline, B5, and far above the bag-of-operations representation that Phase 1 used "
      "(0.5423). Restricted to the four classes other than REAL_BUG, however, the BiLSTM (0.6829) and B5 "
      "(0.6838) are tied, and the next two paragraphs explain why that is the figure to quote."),
    TAB("baselines", "Macro-F1 and accuracy of the model and the baselines on the test set",
        ["System", "Macro-F1", "Excl. REAL_BUG", "Accuracy"],
        [["Attention BiLSTM, class-weight exponent 0.5 (proposed)", "0.7464", "0.6829", "0.7690"],
         ["Attention BiLSTM, class-weight exponent 1.0", "0.7297", "0.6621", "0.7314"],
         ["B5 scalars + bag-of-operations, gradient boosting", "0.7037", "0.6838", "0.7407"],
         ["B2 task scalars, gradient boosting", "0.6670", "0.6588", "0.7115"],
         ["B4 bag-of-operations TF-IDF, SVM", "0.5789", "0.5598", "0.5936"],
         ["B3 bag-of-operations counts, logistic regression", "0.5423", "0.5174", "0.5409"],
         ["B1 majority class", "0.0904", "0.1129", "0.2918"]],
        [8.2, 2.2, 3.0, 2.0]),
    FIG("baselines", "Macro-F1 of the proposed model and the five baselines on the held-out test set",
        "fig_baselines.png", 13.8),
    P("{T:perclass} and {F:cm} give the per-class results. DATA_FAULT is found well (recall 0.913) and "
      "INFRA_FLAKE reasonably (recall 0.792). PASS has a recall of only 0.603, so about 40% of healthy "
      "traces are flagged as faults; this is the weakest operational number of the system and the chaos "
      "experiments of Section 5.8 reproduce it. ENV_FAULT is the weak class: precision 0.611 and recall 0.397. "
      "Softening the class weights from an exponent of 1.0 to 0.5 raised its F1 from 0.397 to 0.481 by "
      "trading recall (0.709 to 0.397) for precision (0.276 to 0.611), and which of the two matters more is "
      "a policy decision, since a false quarantine wastes capacity and a missed one leaves a bad node in "
      "rotation. Of the test traces, 139 (1.3%) were data faults predicted as INFRA_FLAKE and would have "
      "been retried, and no REAL_BUG was missed."),
    TAB("perclass", "Per-class results of the triage model on the test set",
        ["Class", "Precision", "Recall", "F1", "Support"],
        [["PASS", "0.7935", "0.6029", "0.6852", "2,964"],
         ["INFRA_FLAKE", "0.7238", "0.7922", "0.7564", "2,319"],
         ["DATA_FAULT", "0.7267", "0.9126", "0.8091", "3,068"],
         ["ENV_FAULT", "0.6111", "0.3966", "0.4810", "832"],
         ["REAL_BUG", "1.0000", "1.0000", "1.0000", "1,330"]],
        [4.0, 2.8, 2.8, 2.8, 2.8]),
    FIG("cm", "Confusion matrix of the triage model on the 10,513 test traces", "fig_cm.png", 9.6),
    P("**REAL_BUG's score of 1.000 is a workload confound.** The three REAL_BUG scenarios run a small "
      "metadata workload with a median of 7 spans on 2 hosts, whereas every other class has 88 to 134 spans on "
      "9 to 12 hosts. Trace length alone separates them, so this is scenario recognition and not bug "
      "detection, and it inflates macro-F1 by about seven points. Only three such scenarios exist, one per "
      "split, so resplitting cannot repair it, and the figure excluding REAL_BUG is the honest headline."),
    P("**Order carries little usable signal.** The order-free B5 matches the BiLSTM on the other four classes, "
      "and an ablation that shuffles the tokens of every trace reached a validation macro-F1 of 0.7552 at "
      "epoch 3, against 0.7429 for the ordered model at the same epoch and 0.7476 at its best. The sequence "
      "pipeline is nevertheless not wasted, because root-cause localisation needs the aligned call tree, but "
      "the classifier itself does not need order."),
    P("**Compound faults mostly fail.** The 20,010 held-out traces with several simultaneous faults, which "
      "the model never trained on, were classified PASS in 15,563 cases (77.8%). A multi-label output is the "
      "natural remedy and is listed as future work."),
    P("**Attention is moderately concentrated, and two attempts to spread it were rejected.** On 600 test "
      "traces of at least ten spans the mean weight of the top span was 0.54, 8.3% of traces put more than "
      "90% of the weight on one span, and the normalised entropy was 0.48. An entropy bonus of 0.05 produced "
      "uniform attention (entropy 1.000) and a lower macro-F1 of 0.7408, and a hinge penalty gave 0.90 and "
      "0.7439. Both made the evidence spans uninformative and were not adopted, so the evidence spans are "
      "reported as a hint only, consistent with the caution in {R:jain}."),

    H2("Results for the Trust Engine"),
    P("The trust engine was replayed over the 10,513 test decisions in order, with the true label supplied "
      "after each. {F:replay} shows the outcome. The agent reached T1 at decision 14, T2 at decision 906 and "
      "T3 at decision 1,130. The late promotions are expected: with an overall accuracy of 77%, a Wilson "
      "lower bound of 0.85 and then 0.93 is reached only through stretches of consistently correct decisions. "
      "A synthetic fault-injection segment was then appended, and the agent was demoted twice in "
      "succession and ended at T1."),
    FIG("replay", "Replay of the trust engine over the test stream, and the demotions after fault injection",
        "fig_trust_replay.png", 14.2),
    P("The live closed loop exposed a defect that unit tests had not. After a demotion, the next correct "
      "decision re-ran the promotion test over the same recent history that had just earned the tier, so "
      "the lower bound was high again and the tier was bought straight back: 53 tier changes in 50 "
      "decisions. The remedy, now in the engine, is that promotion and retention count only evidence "
      "gathered since the last tier change. The committed replay is unchanged by the fix (same promotions at "
      "the same decisions), while its injected-fault segment now ends at a stable tier instead of "
      "oscillating."),

    H2("Results for Root-Cause Localisation"),
    P("There is no data set that marks the faulty span, so localisation is evaluated against a proxy: the "
      "first span whose own status is an error. {T:rca} and {F:rca} give the fraction of 10,411 fault traces "
      "for which that span is among the top k suspects, against random and slowest-span baselines. The "
      "method is clearly better than both for ENV_FAULT, INFRA_FLAKE and compound faults, with a top-3 rate "
      "of 94.6%, 73.4% and 91.0%. It is weak for DATA_FAULT: its top-3 rate of 47.6% is only a little above "
      "random (38.1%), because a corruption error is not slow and therefore gives the latency baseline "
      "nothing to find. These numbers measure agreement with the proxy and are not validated against "
      "independently labelled root causes."),
    TAB("rca", "Top-k localisation rate against the proxy label",
        ["Fault class", "Top-1", "Top-3", "Top-5", "Random top-3", "Slowest-span top-3"],
        [["INFRA_FLAKE", "0.593", "0.734", "0.995", "0.323", "0.506"],
         ["ENV_FAULT", "0.817", "0.946", "0.983", "0.081", "0.250"],
         ["DATA_FAULT", "0.262", "0.476", "0.778", "0.381", "0.150"],
         ["COMPOUND", "0.739", "0.910", "0.947", "0.089", "0.087"]],
        [3.2, 1.9, 1.9, 1.9, 3.0, 4.2]),
    FIG("rca", "Top-3 localisation rate of the proposed method and two baselines", "fig_rca.png", 12.8),

    H2("Results for the Jaeger Adapter and the Drift Guard"),
    P("The model was trained on corpus records, not on Jaeger exports, so the cost of the export route was "
      "measured on 400 held-out traces ({T:fidelity}, {F:fidelity}). On corpus records the accuracy is 0.74. "
      "On Jaeger exports that carry the training feature schema it is 0.63. On stock exports of the same "
      "traces it falls to 0.20, and 358 of the 400 stock traces (90%) come back as REAL_BUG, because the "
      "small thread and host counts of a stock export resemble the short REAL_BUG workload. This is "
      "the same confound as in Section 5.3 with a measured downstream cost, and it is the reason the drift "
      "guard requires both a known vocabulary and an attested feature schema. A trace from an unrelated "
      "routing application has an unknown-operation rate of 1.0 and is routed to a human."),
    TAB("fidelity", "Accuracy of the model on three forms of the same 400 traces",
        ["Input form", "Accuracy", "Macro-F1", "Agreement with corpus path"],
        [["Corpus records (training format)", "0.74", "0.6913", "1.00"],
         ["Jaeger export with the training schema", "0.63", "0.5565", "0.74"],
         ["Stock Jaeger export", "0.20", "0.1277", "0.20"]],
        [6.6, 2.4, 2.6, 4.5]),
    FIG("fidelity", "Accuracy of the model on corpus records and on two forms of Jaeger export",
        "fig_fidelity.png", 12.8),
    P("The Jaeger path was also checked against a real back end. The live traces were pushed over OTLP/HTTP to a "
      "Jaeger 1.76.0 container and appeared in its interface under the services frontend, orders, inventory "
      "and db, with seven-span traces. A poller then read twelve traces back from Jaeger and sent them to the "
      "agent API, and all twelve were judged in-distribution."),

    H2("Results for the Security Gate"),
    P("The pip-audit report of the project's environment lists 27 vulnerabilities: 12 in pip, 6 in tornado, "
      "4 in jupyterlab, 3 in urllib3, and one each in jupyter-server and notebook. The reachability analysis found that none of "
      "the 27 packages is imported, directly or through the dependencies of the service, from the entry "
      "points of the running service ({T:security}). The verdict is therefore ALLOW and the 27 findings "
      "do not block a release. A positive-control test shows that a flagged package that is imported is "
      "reported as reachable, and a simulated reachable vulnerability makes the gate answer BLOCK, which "
      "overrides any triage result. The analysis is static: a dynamic import or a package that loads an "
      "optional dependency on its own is invisible to it. The findings are still worth fixing by upgrading "
      "the affected packages."),
    TAB("security", "Result of the vulnerability reachability analysis",
        ["Measure", "Value"],
        [["Vulnerabilities flagged by pip-audit", "27"],
         ["Installed but not imported by the service", "27"],
         ["Reachable from the service entry points", "0"],
         ["Findings removed from the blocking decision", "100%"],
         ["Gate verdict", "ALLOW (no reachable vulnerability)"]],
        [8.0, 6.5]),

    H2("Results for the Chaos Experiments"),
    P("Six kinds of fault, built from signatures seen in real TraceBench fault traces, were injected into 30 "
      "healthy held-out traces each, and a control of 30 untouched traces was added. The agent started at T3 "
      "with retention demotion enabled. {T:chaos} and {F:chaos} show the outcome. Every injected fault except "
      "the silent regression was flagged as a fault. The class was right for 97% of slow-datanode "
      "injections but only 50% of network disconnects, half of which were called DATA_FAULT. The control "
      "shows that 43% of healthy traces are flagged, the false-alarm floor that every fault row must beat. The "
      "silent regression has no trace signature by construction: none of its 30 traces was classified "
      "REAL_BUG, and two were acted on autonomously as unsafe merges. Those two immediately demoted the agent "
      "from T3 to T2 and from T2 to T1, and a retention failure on the data-node-kill experiments took it to "
      "T0: three demotions in all."),
    TAB("chaos", "Outcome of the chaos experiments (30 injections per row)",
        ["Injected fault", "Should be", "Flagged", "Class correct", "Unsafe actions"],
        [["slow_datanode", "INFRA_FLAKE", "100%", "97%", "0"],
         ["network_disconnect", "INFRA_FLAKE", "100%", "50%", "0"],
         ["datanode_kill", "INFRA_FLAKE", "100%", "83%", "0"],
         ["block_corruption", "DATA_FAULT", "100%", "70%", "0"],
         ["readonly_datanode", "ENV_FAULT", "100%", "80%", "0"],
         ["silent_regression", "REAL_BUG", "40%", "0%", "2"],
         ["control (healthy)", "PASS", "43%", "57%", "0"]],
        [4.4, 3.2, 2.4, 3.0, 3.1]),
    FIG("chaos", "Flagged rate and correct-class rate for each injected fault", "fig_chaos.png", 13.6),

    H2("Results for the Live System"),
    P("The live system was built to show the loop, not to benchmark the model. A model trained on the "
      "1,850 minishop traces reaches a macro-F1 of 1.000 on the held-out scenarios, but so does the scalar "
      "gradient-boosting baseline (1.000), and the bag-of-operations baselines reach 0.975 and 0.884. The "
      "injected faults are simple, so this number says nothing about model quality, and the credible accuracy "
      "figure of the project remains the HDFS result of Section 5.3."),
    P("{F:livetraj} shows a run of the scripted schedule, which serves 390 requests in fifteen phases covering eight kinds of traffic. "
      f"The agent decided correctly on {tot_c} of {tot_n} requests ({LIVE_ACC}) and made 2 unsafe autonomous "
      "actions. {T:livephase} breaks the result down by phase, and {T:liveevents} lists the trust-tier "
      "changes. The agent starts at T0, is promoted at requests 15, 45 and 97, and reaches T3. Every fault "
      "type before the silent bug is classified correctly in every request. The silent bug, a real defect "
      "whose trace is identical to a healthy one, is classified correctly in 3 of 30 requests. Two of the "
      "first four requests of that phase were merged autonomously, and each demoted the agent, first from T3 "
      "to T2 and then from T2 to T1. After the phase ended, the accuracy retention check took it to T0 and, once healthy "
      "traffic had rebuilt 15 correct decisions, it was promoted to T1 again. The run therefore shows the "
      "behaviour the design intends: autonomy earned from evidence, lost on a missed defect, and not "
      "regained without new evidence. Repeated runs differ by a few requests, because the services sleep "
      "for real, and gave between 92.0% and 92.9% correct decisions."),
    FIG("livetraj", "Trust tier of the agent during the scripted live run", "fig_live_traj.png", 15.4),
    TAB("livephase", "Agent accuracy by phase of the scripted live run",
        ["Phase", "True class", "Requests", "Correct", "Rate"],
        LIVE_ROWS, [4.2, 3.2, 2.6, 2.6, 2.4]),
    TAB("liveevents", "Trust-tier changes during the scripted live run",
        ["Request", "Change", "Tiers", "Reason"],
        EV_ROWS, [2.0, 2.4, 2.8, 8.9]),
    P("The dashboard exposes the same loop interactively. {F:uilive} shows its live view at the end of the "
      "scripted run, {F:uigate} shows a canary release of a slow-database version being rolled back by the "
      "gate, and {F:uihdfs} shows the HDFS model refusing to judge the trace of an unrelated application. "
      "A healthy new version is promoted by the canary check, a slow database raises the 95th-percentile "
      "latency 16.7 to 20.9 times in repeated runs, and a flaky network raised the error rate to 42.5% "
      "against 0%. When the agent is at a tier too low to act, the gate still reports its recommendation and "
      "states that it is advisory."),
    FIG("uilive", "Live view of the dashboard at the end of the scripted run (393 requests served, three more than the 390 of Table 5.9)", "ui_live.png", 15.4),
    FIG("uigate", "Canary release of a slow-database version rolled back by the release gate", "ui_gate.png", 14.0),
    FIG("uihdfs", "The HDFS model abstaining on a trace from an unrelated application", "ui_hdfs.png", 13.0),

    H2("Verification and Engineering Findings"),
    P("The behaviour described above is protected by an automated suite of 183 passing tests, with 12 "
      "skipped because the opa executable is not installed or because of one legacy marker. The tests cover "
      "the class taxonomy and policy, the leakage checks of the split, clock alignment, the trust engine, "
      "root-cause ranking, the Jaeger adapter, the gate precedence, the security analysis with a positive "
      "control, the chaos harness, the actuator against a faked HTTP service, the live corpus and the "
      "dashboard endpoints. Several of the most useful results were not features but defects that "
      "measurement exposed. {T:findings} lists them with the correction made."),
    TAB("findings", "Defects exposed by measurement and the corrections made",
        ["Finding", "Cause", "Correction"],
        [["Phase 1 accuracy of 88.18% frozen over three epochs", "Rows were event-count vectors used as token ids; a one-column rule scores 88.2%",
          "Input rebuilt from sequences; per-class metrics and baselines"],
         ["Token features shifted out of line", "Operation names containing spaces broke the space-separated token stream",
          "Names normalised; covered by a test"],
         ["Stock Jaeger exports judged with 20% accuracy", "Missing trace-level features resemble the short REAL_BUG workload",
          "Feature-schema condition added to the drift guard"],
         ["Live test accuracy of 1.0 with classes missing", "Per-class scenario assignment let PASS overwrite fault classes",
          "Scenario assigned once by dominant fault class; HDFS split verified identical"],
         ["53 tier changes in 50 decisions", "A promotion re-used the evidence that had just failed to hold the tier",
          "Only evidence since the last tier change counts"],
         ["Some host agents fell into an unknown bucket", "Agent-type lookup was case-sensitive",
          "Lookup made case-insensitive"],
         ["A demonstration reported an unknown-operation rate of 0.0", "The text was fixed in code; the measured value was 0.47",
          "The measured value is printed"],
         ["Two seconds lost on every local call on Windows", "The name localhost resolved through IPv6 first; 2,111 ms against 26 ms",
          "All scripts use 127.0.0.1"],
         ["Evidence spans lost their meaning", "Attention regularisation spread weight almost uniformly",
          "Both attempts rejected; spans reported as a hint"]],
        [4.6, 6.0, 5.5]),

    H2("Discussion and Threats to Validity"),
    P("{T:summary} collects the principal measurements together with the context in which each applies. "
      "They belong to different tasks and are not combined into one score."),
    TAB("summary", "Principal measurements and the context in which they apply",
        ["Measurement", "Value", "Context and caveat"],
        [["HDFS test macro-F1", "0.7464", "57 held-out scenarios; 0.6829 excluding REAL_BUG, tied with B5"],
         ["REAL_BUG recall", "1.000", "Workload confound: trace length separates the class"],
         ["PASS recall", "0.603", "About 40% of healthy traces are flagged"],
         ["Stock Jaeger export accuracy", "0.20", "90% called REAL_BUG; drift guard abstains"],
         ["Root-cause top-3 (ENV / DATA)", "0.946 / 0.476", "Against a proxy label, not true root causes"],
         ["Reachable vulnerabilities", "0 of 27", "Static analysis of imports only"],
         ["Silent regression, class correct", "0%", "Two unsafe merges; trust fell T3 to T0"],
         ["Live closed loop, correct decisions", LIVE_ACC, "Toy system; 2 unsafe actions; T0 to T3 to T1"],
         ["Live model macro-F1", "1.000", "Scalar baseline also 1.000: a pipeline demonstration"],
         ["Automated tests", "183 passed", "12 skipped (no opa executable; one legacy marker)"]],
        [4.6, 2.6, 8.9]),
    P("The results are subject to the following threats to validity."),
    BUL("**Domain.** The shipped model is valid only for HDFS traces in the training schema. The drift guard "
        "prevents a wrong answer outside it but cannot show that the model is right inside it."),
    BUL("**Labels.** Class labels come from TraceBench's scenario taxonomy, so the model learns the scenarios "
        "that the taxonomy names. A single split and a single seed were evaluated; no confidence interval "
        "over seeds is given."),
    BUL("**Confidence.** The confidence floor of 0.60 is an uncalibrated threshold on a softmax output "
        "{R:guo}."),
    BUL("**Synthetic evidence.** The chaos faults copy signatures from real traces and the live faults are "
        "simple; both stress the pipeline but neither replaces the HDFS result."),
    BUL("**Unexecuted infrastructure.** The two GitHub workflow files and the full container stack with "
        "Prometheus and Grafana were written and checked for structure but never run. Only Jaeger was run "
        "against a real server, and the actuator has only ever run in dry-run mode or against a faked HTTP "
        "service."),
]
