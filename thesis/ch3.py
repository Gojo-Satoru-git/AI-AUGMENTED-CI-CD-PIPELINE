from blocks import *

BLOCKS = [
    CH("SYSTEM DESIGN"),
    H2("Overall Architecture of the Proposed System"),
    P("The system is organised in five layers, shown in {F:arch}. Trace sources feed an adapter layer that "
      "normalises them and checks that the model is entitled to judge them. The decision layer holds the "
      "triage model, the root-cause localiser and the trust engine. The release-gate layer adds two "
      "independent signals, a canary comparison and a vulnerability check, and combines everything by a fixed "
      "precedence. The last layer executes the decision in dry-run mode, exports metrics and feeds the true "
      "outcome back to the trust engine."),
    FIG("arch", "Overall architecture of the proposed system", "fig_arch.png", 14.2),
    P("Two design rules run through the layers. First, deciding and executing are separate: the gate and the "
      "policy only produce a decision, and a separate actuator, which contains no policy, carries it out. "
      "Second, the system prefers to abstain. A trace outside the training domain, a canary with too little "
      "evidence and a confidence below 0.60 each route to a human instead of to an action."),

    H2("Dataset Description"),
    P("The system uses two sets of traces and two supporting records. They are described separately because "
      "no component merges them. {T:datasets} summarises them."),
    TAB("datasets", "Data used by the components of the system",
        ["Data", "Used by", "Content", "Size and labels", "Evaluation use"],
        [["TraceBench HDFS v3", "Triage, trust replay, RCA, chaos", "Operation events and parent edges from HDFS under fault injection",
          "363 scenario folders; 81,777 traces; 3.0 GB; class from folder family", "Held-out scenarios; per-class metrics"],
         ["Minishop live traces", "Live loop, canary, dashboard", "Jaeger-format traces of a 4-service application",
          "37 experiments; 1,850 traces; exact labels from the injector", "Pipeline demonstration only"],
         ["HDFS Jaeger samples", "Adapter, drift guard, gate", "Five exported traces, four from HDFS and one from an unrelated application",
          "One trace each", "Qualitative"],
         ["pip-audit report", "Security gate", "Vulnerabilities in the installed packages", "27 findings", "Reachability analysis"]],
        [2.6, 2.6, 4.0, 3.9, 3.0]),
    H3("TraceBench HDFS"),
    P("TraceBench {R:tracebench} holds 363 scenario folders. Each folder is one experiment on a Hadoop "
      "distributed file system and contains the operation events of its traces and the edges between a "
      "parent operation and its children. Folder names encode the injected fault family. The mapping used "
      "here is given in {T:classes}. Scenarios that inject several faults at once (COMPOUND) are held out "
      "entirely and used only as a generalisation probe, 20,010 traces in all. The remaining 61,767 traces "
      "form the corpus, split by scenario into 40,967 training, 10,287 validation and 10,513 test traces."),
    TAB("classes", "Class taxonomy and the action each class maps to",
        ["Class", "TraceBench family", "Meaning", "Action", "Min. tier"],
        [["PASS", "NM_CL, NM_DN", "No failure signature", "ALLOW_MERGE", "T2"],
         ["INFRA_FLAKE", "AN_Net_*, AN_Proc_*", "Transient network or process fault", "AUTO_RETRY_PIPELINE (budget 2)", "T2"],
         ["DATA_FAULT", "AN_Data_*", "Corruption or loss; deterministic", "ESCALATE_TO_INFRA", "T1"],
         ["ENV_FAULT", "AN_Sys_*", "Host-level failure", "QUARANTINE_NODE", "T3"],
         ["REAL_BUG", "COM_Sin_Bug", "Genuine defect signature", "BLOCK_MERGE_ALERT_DEV", "T1"]],
        [2.9, 2.9, 3.6, 5.2, 1.5]),
    H3("Minishop live traces"),
    P("To run the closed loop without borrowing a production system, the project includes a small "
      "application, named minishop, with four services: a frontend, an order service, an inventory service "
      "and a database. Faults can be injected into it: a slow database and a flaky network (INFRA_FLAKE), "
      "data corruption (DATA_FAULT), a host failure (ENV_FAULT), a code bug that aborts the request "
      "(REAL_BUG), a novel bug that adds an unseen operation, and a silent bug that leaves the trace "
      "identical to a healthy one. The injector knows whether the fault took effect on each request, so every "
      "trace carries an exact label. A dependency-free tracer writes Jaeger JSON and OTLP/HTTP JSON. "
      "Thirty-seven labelled experiments produced 1,850 traces: 866 PASS, 318 INFRA_FLAKE, 122 DATA_FAULT, "
      "300 ENV_FAULT and 244 REAL_BUG."),
    H3("Supporting records"),
    P("Five Jaeger exports serve the adapter and the drift guard: four real HDFS traces, one each of PASS, "
      "INFRA_FLAKE, DATA_FAULT and ENV_FAULT, and one trace from an unrelated routing application that the model must refuse "
      "to judge. The vulnerability report is the output of pip-audit on the project's own environment, "
      "27 findings."),

    H2("Explanation of System Components"),
    H3("Data foundation and clock alignment"),
    P("Each event in TraceBench carries a start time on the clock of its own host, so sorting events by raw "
      "start time interleaves hosts incorrectly and drops the true root operation. The edge table records, "
      "for every child, the parent's clock reading when it spawned the child. The offset of a child's host "
      "relative to its parent's is therefore the difference of the two readings, and offsets are propagated "
      "down the call tree from the root, as {F:align} illustrates. Ties are broken by depth so that a caller "
      "precedes its handler. All 81,777 traces align completely, and the aligned order recovers the true "
      "root operation. Each aligned event becomes a token of the form operation|status, with its duration, "
      "host role, agent type and depth kept as side features."),
    FIG("align", "Clock alignment of a parent and child operation recorded on different hosts", "fig_align.png", 13.0),
    H3("Triage model"),
    P("The classifier, shown in {F:model}, embeds each span token and concatenates small embeddings of the "
      "host role, the agent type and the depth, and a transformed duration. A one-layer bidirectional LSTM "
      "reads the sequence in both directions. Instead of keeping only the last hidden state, which is dominated "
      "by padding in a padded batch, the model pools all states with additive attention, so every span "
      "contributes in proportion to a learned weight. The attended vector, a max-pooled vector and nine "
      "trace-level scalar features feed a small dense head with five outputs. The attention weights are "
      "returned with every prediction and identify the spans that drove it."),
    FIG("model", "Architecture of the attention-pooled BiLSTM triage model", "fig_model.png", 15.0),
    H3("Trust engine"),
    P("The trust engine ({F:trust}) stores, for every agent, a tier and the history of its decisions. "
      "Promotion is decided by the Wilson lower bound of accuracy over the decisions whose true outcome "
      "is known, which prevents a short lucky streak from earning autonomy. Demotion has two triggers. "
      "The first is immediate: if the agent classified a real defect as something harmless and acted on it "
      "without a human, it drops one tier at once. The second is optional: if accuracy falls below a floor "
      "set ten points under the threshold that earned the current tier, the agent is demoted. Evidence is "
      "counted only from the last tier change onwards, so an agent cannot buy a tier straight back with "
      "the history that has just been proved insufficient."),
    FIG("trust", "Trust tiers, promotion thresholds and demotion rules", "fig_trust_fsm.png", 14.2),
    H3("Root-cause localisation"),
    P("For every trace classed as a fault, the localiser ranks spans by suspicion. A baseline of mean and "
      "standard deviation of duration per operation is built from healthy traces. A span scores its "
      "standardised slowness (never negative), plus a fixed bonus of 2.0 if its own status is an error, "
      "plus a small bonus for sitting near the root. The top spans and a ready-to-paste postmortem in "
      "Markdown are returned."),
    H3("Release gate and supporting modules"),
    P("The release gate ({F:gate}) combines four signals and chooses the most conservative. A security "
      "verdict of BLOCK wins over everything; a canary verdict of ROLLBACK comes next; a trace that fails "
      "the drift guard becomes a request for human review; otherwise the triage decision stands. The trust "
      "tier then decides whether the agent may execute the result or only recommend it. The drift guard "
      "checks two things: that the operation vocabulary of the trace is known to the model, and that the "
      "trace was produced with the same feature definitions the model was trained on. The canary module "
      "compares a new version's error rate and 95th-percentile latency with the baseline, using Wilson "
      "intervals so that few traces cannot cause a rollback. The security module decides which "
      "vulnerabilities the running service can reach. The chaos harness injects six kinds of fault into "
      "healthy traces and a control with none."),
    FIG("gate", "Precedence of the release gate", "fig_gate.png", 13.0),
    H3("Actuator and continuous-integration hook"),
    P("The actuator carries out a decision and contains no policy. By default it runs in dry-run mode and "
      "only records what it would do. Its real backend can set a commit status, post or update a single pull "
      "request comment marked so that it is not duplicated, rerun the failed jobs of a run, open an issue "
      "and run one configured command, such as cordoning a node or undoing a rollout. Commands are built as "
      "argument lists and never passed through a shell, and any node or deployment name that came from "
      "trace data is validated first, because traces are untrusted input. The workflow that calls the gate "
      "is triggered when the test workflow completes, so that it can rerun failed jobs, and it executes only "
      "code from the default branch; the artifacts of the pull request are treated as data and the pull "
      "request's code is never checked out in that workflow. Until the agent has earned a tier in "
      "deployment the check is advisory, since a T0 agent can only ask for a human."),
    H3("End-to-end workflow and integration"),
    P("{F:loop} shows the closed loop that the live system runs for every request. The request is served and "
      "traced, the adapter aligns the clocks and the drift guard checks the domain. The model, the policy "
      "and the gate produce a decision, which the actuator executes in dry-run mode if the tier allows. The "
      "true outcome is then reported back and the trust engine updates the tier, so the next request is "
      "governed by what the previous ones proved. Every component is called through one function, so the "
      "HTTP service, the command-line gate and the dashboard all use identical logic."),
    FIG("loop", "The closed decision loop of the live system", "fig_loop.png", 14.2),
]
