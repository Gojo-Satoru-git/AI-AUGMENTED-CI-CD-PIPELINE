from blocks import *

BLOCKS = [
    CH("INTRODUCTION"),
    H2("Continuous Delivery and the Triage Problem"),
    P("Modern software teams merge changes many times a day. Every change is built, tested and "
      "deployed by a continuous integration and delivery (CI/CD) pipeline {R:humble}. When a pipeline run "
      "fails, a person has to decide what the failure means, and the correct response depends on the answer. "
      "A transient network or process fault should simply be retried. A data fault such as corruption is "
      "deterministic, so retrying it only burns build minutes and fails again. A failing host should be taken "
      "out of rotation. A genuine defect must block the merge and alert the developer. On a dashboard these "
      "four situations look alike: a red run. Flaky failures are common in practice {R:luo}, which makes the "
      "distinction expensive to make by hand and costly to get wrong in either direction."),
    P("Distributed tracing offers the evidence needed to make the distinction. A trace records one request "
      "as a tree of timed operations called spans, each tagged with its host and status {R:dapper}, and open "
      "tools such as OpenTelemetry and Jaeger collect them {R:otel} {R:jaeger}. The difficulty is that "
      "reading a trace is a manual, expert task, and the clocks of the hosts that produced it do not agree."),
    P("A second question is rarely asked explicitly: how far may an automated agent be trusted to act on its "
      "own conclusion? Levels of automation have long been treated as a design variable rather than a switch "
      "{R:parasuraman}. An agent that has never been measured on a system should not merge code unattended, "
      "while one with a long, verified record may reasonably be allowed to retry a build without asking."),

    H2("About the Project"),
    P("This project builds an AI-augmented release gate for CI/CD pipelines. It classifies the trace of a "
      "failed run into one of five operational classes, **PASS**, **INFRA_FLAKE**, **DATA_FAULT**, "
      "**ENV_FAULT** and **REAL_BUG**, maps the class to an action (allow, retry, escalate, quarantine, block), "
      "and decides how much of that action the agent may carry out alone from its measured track record. "
      "The work is organised into five cooperating parts."),
    BUL("**Data foundation.** Real HDFS traces from the TraceBench data set are converted into clock-aligned, "
        "labelled operation sequences, and the split between training and test data is made by scenario so "
        "that no fault is seen on both sides."),
    BUL("**Triage model.** An attention-pooled bidirectional LSTM (BiLSTM) classifies each trace and exposes "
        "the spans it attended to as evidence."),
    BUL("**Trust engine.** Four autonomy tiers, T0 (observe) to T3 (act autonomously), are earned by a "
        "statistical test on the agent's own accuracy and lost immediately when it acts wrongly on a real "
        "defect."),
    BUL("**Root-cause localisation.** Spans are ranked against a per-operation latency baseline and a "
        "postmortem is generated automatically."),
    BUL("**Release gate.** A Jaeger adapter, a drift guard, a canary comparison, a vulnerability "
        "reachability check, a chaos harness and a dry-run actuator combine into one decision with an explicit "
        "precedence order."),
    P("The project also corrects an error in its own first phase. Phase 1 reported a validation accuracy of "
      "88.18% that stayed frozen for three epochs. Investigation showed that each row of its input file was an "
      "event-count vector rather than a sequence, and that a rule using one column alone scores about 88.2% "
      "on the same set. This thesis therefore reports per-class results against explicit baselines and states "
      "the limits of every number."),

    H2("Contributions of the Project"),
    P("The contributions of this work, each measured in Chapter 5, are the following."),
    BUL("A leakage-free sequence data pipeline for HDFS traces, in which clock alignment makes every one of "
        "the 81,777 traces orderable and the split is made by scenario."),
    BUL("An attention-pooled BiLSTM triage model with five operational classes mapped to pipeline actions, "
        "reported per class against five baselines."),
    BUL("A trust engine that grants autonomy from a statistical lower bound, withdraws it immediately on a "
        "missed defect, and counts only fresh evidence after a change."),
    BUL("A drift guard whose second condition, an attested feature schema, exists because stock trace "
        "exports were measured to be misjudged four times in five."),
    BUL("An integrated release gate, a closed-loop live system and an interactive dashboard that make the "
        "whole behaviour visible, including the case in which it fails."),

    H2("Problem Statement and Objectives"),
    P("The problem addressed is the automatic triage of failed CI/CD runs from trace evidence, together with "
      "a defensible answer to how much autonomy the triage agent should be given. The objectives are:"),
    BUL("to build a labelled, leakage-free sequence data set from real HDFS fault-injection traces;",),
    BUL("to train a classifier that beats order-free and count-based baselines on held-out scenarios;"),
    BUL("to gate autonomy on a statistical lower bound and to withdraw it when the agent is wrong;"),
    BUL("to localise suspect spans and to refuse to answer when a trace is outside the model's domain;"),
    BUL("to demonstrate the whole closed loop on a running system with injectable faults, and to report "
        "honestly where it fails."),

    H2("Organization of the Project"),
    P("This thesis is organized as follows:"),
    BUL("**Chapter 1: Introduction** presents the domain, the problem statement and the objectives."),
    BUL("**Chapter 2: Literature Survey** reviews published work on trace-based failure analysis, flaky "
        "failures, automation and trust, root-cause localisation, chaos engineering and dependency risk, and "
        "derives the research gap."),
    BUL("**Chapter 3: System Design** presents the architecture and the data sets and explains each component."),
    BUL("**Chapter 4: Detailed Design** gives, for each component, its inputs, processing and one structured "
        "algorithm, together with the parameters used."),
    BUL("**Chapter 5: Results and Discussion** lists the software and hardware requirements, reports the "
        "results of every component, and discusses limitations and threats to validity."),
    BUL("**Chapter 6: Conclusion and Future Enhancement** summarises what was achieved, separates it from "
        "what was not, and proposes enhancements."),

    CH("LITERATURE SURVEY"),
    P("This chapter reviews published research relevant to the five parts of the project. Section 2.1 covers "
      "trace-based and log-based failure analysis, Section 2.2 covers flaky failures and release engineering, "
      "Section 2.3 covers automation, trust and statistical gating, and Section 2.4 covers root-cause "
      "localisation, chaos engineering and dependency risk. Section 2.5 derives the research gap. Citation "
      "numbers follow the order in which the publications are first cited in the thesis."),

    H2("Trace-Based and Log-Based Failure Analysis"),
    P("Dapper, the tracing infrastructure of Google, showed that request-level traces can be collected "
      "continuously in production at low overhead and are useful for diagnosis {R:dapper}. X-Trace "
      "captured causal paths across layers of a networked system {R:xtrace}, and Pinpoint followed request "
      "paths through the components of an internet service and applied data mining to find the components "
      "associated with failures {R:pinpoint}. Together they established the causal call tree, rather than a "
      "flat log, as the unit of diagnosis, which is the representation used here."),
    P("Learning-based detection began mostly with logs. DeepLog models the sequence of log keys with an LSTM "
      "and flags a key that is not among the most likely successors {R:deeplog}. LogRobust combines semantic "
      "vectors of log events with an attention-based BiLSTM to tolerate unstable log data {R:logrobust}. "
      "An experience report compared supervised and unsupervised log-based detectors on public data sets and "
      "made the comparison reproducible {R:he}. For traces, Nedelkoski et al. use both the span sequence and "
      "its timing in a multimodal deep model {R:nedelkoski}, and Liu et al. detect anomalous microservice "
      "traces with service-level deep Bayesian networks {R:liu}. These works detect anomalies; they do not "
      "assign an operational class that maps to a pipeline action, and none ties the decision to a measured "
      "right to act."),
    P("The TraceBench data set {R:tracebench} fills part of this gap for data. It provides traces from the "
      "Hadoop distributed file system collected under injected network, process, data and system faults, "
      "and combinations of faults. Its scenario taxonomy is the source of the class labels used in this "
      "project. A study of microservice faults with a benchmark system and an empirical analysis of real "
      "faults {R:zhoux} motivates treating fault type, not only fault presence, as the quantity to predict."),
    P("Attention mechanisms {R:bahdanau} and bidirectional recurrent networks {R:schuster} on top of LSTM "
      "cells {R:lstm} are the building blocks of the classifier. The attention weights are tempting as "
      "explanations, but it has been argued that they are not reliable explanations {R:jain} and, in reply, "
      "that this depends on what is claimed {R:wiegreffe}. This project follows the cautious reading: the "
      "attended spans are reported as a hint, never as proof."),
    P("Finally, distributed systems cannot assume that hosts agree on the time. Lamport's work on ordering "
      "events {R:lamport} is the classical reason to derive order from causal relations rather than "
      "timestamps, and the clock alignment of Section 3.3.1 applies the same idea to the recorded call tree."),

    H2("Flaky Failures and Release Engineering"),
    P("An empirical analysis of flaky tests found that asynchronous waiting, concurrency and test-order "
      "dependence are among their most common causes {R:luo}. The cost of a flaky or misdiagnosed failure "
      "falls on the pipeline: retries consume time, and wrongly blocked or wrongly merged changes consume "
      "trust. Continuous delivery practice {R:humble} treats the pipeline as an automated, repeatable "
      "release path, and site reliability engineering adds canary releases, error budgets and blameless "
      "postmortems as operating practices {R:sre}. The project automates two of them: the canary comparison "
      "and the postmortem."),
    P("Machine learning inside such systems brings its own maintenance cost. Sculley et al. describe how a "
      "model's hidden dependencies and feedback loops create technical debt {R:sculley}. The closed loop of "
      "this project, in which the agent's own decisions change its permissions, is such a feedback loop, and "
      "the live experiments of Chapter 5 show one concrete failure of it."),

    H2("Automation, Trust and Statistical Gating"),
    P("Parasuraman, Sheridan and Wickens describe automation as a range of levels across four functions: "
      "acquiring information, analysing it, selecting a decision and implementing an action {R:parasuraman}. "
      "The four trust tiers of this project are a deliberately coarse version of that idea. T0 stops after "
      "analysis, T1 recommends a decision, T2 implements actions that can be reversed and T3 implements all "
      "of them."),
    P("Moving between levels needs a decision rule. The Wilson score interval {R:wilson} is a classical "
      "interval for a binomial proportion, and a comparison of interval methods recommends it over the "
      "simple normal approximation for small samples {R:brown}. Its lower bound is a conservative estimate "
      "of an accuracy measured on few trials: 17 correct out of 20 is 85%, but the lower bound is about "
      "0.64. The project uses this bound as the promotion test. A separate caution comes from calibration "
      "research: the confidence of modern neural networks is often poorly calibrated {R:guo}, so the "
      "confidence value of the classifier is used only as a coarse floor and never as a probability of "
      "being correct."),

    H2("Root-Cause Localisation, Chaos Engineering and Dependency Risk"),
    P("Pinpoint {R:pinpoint} and later trace-analysis work localise faults by comparing failing requests "
      "with normal ones. The localiser of this project follows the same principle in its simplest form: "
      "each span is scored against the latency distribution of the same operation in healthy traces, and "
      "errors add a fixed bonus. It is evaluated against a stated proxy because no data set marks the "
      "faulty span."),
    P("Chaos engineering injects faults into a system on purpose, to test a hypothesis about its steady "
      "state {R:basiri}. The chaos harness of this project applies the idea to the agent itself: faults "
      "are injected into healthy traces and the question is whether the trust engine reacts correctly."),
    P("Open-source dependencies add risk that is easy to over-count. Ponta et al. argue that a vulnerable "
      "dependency matters when the vulnerable code can actually be reached from the application "
      "{R:ponta}. The security gate of this project uses a static form of that reasoning to decide which "
      "reported vulnerabilities may block a release."),

    H2("Research Gap"),
    P("The literature supplies detectors for logs and traces, data sets of faults, interval statistics and "
      "operating practices, but they are seldom combined. The gaps addressed here are: (i) a class "
      "taxonomy that maps directly to pipeline actions; (ii) an explicit, statistical and reversible rule "
      "for how much autonomy the classifier receives; (iii) a guard that abstains when a trace is outside "
      "the training domain; and (iv) an evaluation that reports weak numbers as plainly as strong ones. "
      "{T:related} summarises the works reviewed and what each leaves open. Chapter 3 describes a system "
      "that integrates the four."),
    TAB("related", "Summary of the reviewed work and what each leaves open",
        ["Work", "Input", "Output", "Leaves open"],
        [["Dapper, X-Trace, Pinpoint", "Request traces and paths", "Collection; path-based diagnosis",
          "No operational class or action"],
         ["DeepLog, LogRobust", "Log-key sequences", "Normal or anomalous", "No fault type tied to an action"],
         ["Nedelkoski et al.; Liu et al.", "Traces with timing", "Anomaly decision", "No policy; no right to act"],
         ["Luo et al.; Zhou et al.", "Flaky tests; microservice faults", "Empirical causes of failure",
          "No automatic triage"],
         ["Parasuraman et al.", "Levels of automation", "Conceptual model", "No statistical rule for a level"],
         ["Ponta et al.", "Dependency code reachability", "Reachable vulnerabilities",
          "Not tied to a release decision"],
         ["This project", "Clock-aligned traces, canary, CVEs", "Five classes, actions, tiers",
          "Limits stated in Chapter 5"]],
        [3.7, 3.9, 4.0, 4.5]),
]
