from blocks import *

ALG1 = """Input : events of one trace (op, host, status, start, duration);
        edge records (father, child, father_start)
Output: ordered tokens, side features, trace-level scalars

 1. Build the call tree from the edge records. root <- event with no father.
 2. offset[host(root)] <- 0 ; queue <- [root]
 3. while queue is not empty:
      f <- pop(queue)
      for each child c of f:
         if host(c) has no offset yet:
            offset[host(c)] <- (father_start(c) + offset[host(f)])
                               - first_start(host(c))
            (father's clock reading, moved onto the common timeline, minus
             the first reading on the child's own host)
         push(queue, c)
 4. For every event e: t*(e) <- start(e) + offset[host(e)]
 5. Sort events by (t*(e), depth(e)); a caller precedes its handler on ties.
 6. token(e) <- normalise(op(e)) + "|" + status(e)
        normalise: spaces -> "_", "|" -> "/"
 7. side(e) <- (host role, agent type, depth, log(1 + duration))
 8. scalars <- log(1+n_events), log(1+n_thread_ids), log(1+n_hosts),
               log(1+n_edges), log(1+n_reports), log(1+latency),
               log(1+n_errors), n_errors / n_events, log(1+n_distinct_ops)
 9. Assign the class from the scenario-folder family (Table 3.2)."""

ALG2 = """Input : tokens s_1..s_L (L <= 256), side features, scalars x
Output: class, confidence, evidence spans, action

 1. e_t <- [Emb(s_t) ; Emb_host ; Emb_agent ; Emb_depth ; W_d * dur_t]
          (48 + 3 x 8 + 8 = 80 dimensions)
 2. h_1..h_L <- BiLSTM(e_1..e_L)                 (64 units per direction,
                                                  packed: no padding steps)
 3. a_t <- w' tanh(W h_t)  for real positions t
    alpha <- softmax(a)                           (padding masked out)
 4. c <- sum_t alpha_t h_t ;   m <- max_t h_t     (128 dimensions each)
 5. z <- [c ; m ; ReLU(W_x x)]                    (128 + 128 + 32)
 6. p <- softmax(Dense(ReLU(Dense(z))))           (5 classes)
    class <- argmax p ;  conf <- max p
 7. evidence <- the 5 spans with the largest alpha_t
 8. oov <- share of tokens absent from the training vocabulary
 9. drift <- in_distribution if oov <= 0.5 and the feature schema is attested
10. action <- policy(class, conf, tier):
        conf < 0.60                    -> REVIEW (human)
        tier < min_tier(action)        -> recommend only
        otherwise                      -> execute"""

ALG3 = """Input : decision d (class, action, autonomous, tier) and its true label y
Output: updated tier and the reason for any change

 1. Store y with d ; correct <- (class(d) = y)
 2. if autonomous(d) and y = REAL_BUG and class(d) != REAL_BUG:
        tier <- max(tier - 1, 0) ; floor <- id(d) ; return "demoted"
 3. E <- ground-truthed decisions with id > floor, most recent 200
    n <- |E| ; k <- number of correct decisions in E
 4. LB(k, n) <- (p + z^2/2n - z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)
        with p = k/n and z = 1.96
 5. if tier < 3 and n >= n_min[tier] and LB(k, n) >= thr[tier]:
        tier <- tier + 1 ; floor <- id(d) ; return "promoted"
 6. if retention is enabled and tier >= 1:
        W <- last 50 decisions with id > floor ; n <- |W|
        if n >= 30 and LB(k, n) < thr[tier-1] - 0.10:
            tier <- tier - 1 ; floor <- id(d) ; return "demoted"
 7. return "hold"
        n_min = (15, 30, 50) and thr = (0.70, 0.85, 0.93) for T0, T1, T2"""

ALG4 = """Input : trace (optional), security verdict, canary report (optional), tier
Output: recommended action, executed action, reasons

 1. if a trace is given:
        pred <- Algorithm 4.2 ; triage <- policy(pred.class, pred.conf, tier)
        drift <- vocabulary_ok(pred.oov) and schema_attested(trace)
 2. if security = BLOCK            : rec <- BLOCK_MERGE      ; rule <- security
 3. else if canary = ROLLBACK      : rec <- ROLLBACK_CANARY  ; rule <- canary
 4. else if drift = false          : rec <- REVIEW           ; rule <- ood-abstain
 5. else if triage exists          : rec <- triage.action
 6. else                           : rec <- ALLOW_MERGE
 7. if rec = ALLOW_MERGE and security = REVIEW : rec <- REVIEW
    if rec = ALLOW_MERGE and canary   = HOLD   : rec <- REVIEW
 8. autonomous <- (rec != REVIEW) and (tier >= min_tier(rec))
 9. executed <- rec if autonomous or rec = REVIEW else REVIEW
10. log the decision; the actuator executes `executed` (dry-run by default)"""

BLOCKS = [
    CH("DETAILED DESIGN"),
    H2("Data Foundation"),
    P("The data foundation turns the raw TraceBench tables into labelled sequences. {A:align} lists the "
      "steps for one trace. The offsets are recovered once per host from the first parent-child edge that "
      "reaches it and then reused down the tree. Operation names are normalised because several contain "
      "spaces, and the token stream is stored as a space-separated string; an unnormalised name would shift "
      "every later feature out of line, a defect that occurred once and is now covered by a test. The title "
      "of a task is deliberately not a model input. It records the client workload, and folder names encode "
      "the workload next to the fault, so feeding it would let the model recognise the scenario instead of "
      "the failure."),
    ALG("align", "Clock alignment and token construction for one trace", ALG1),
    P("The split is made by scenario, never by trace. Each scenario is assigned once, keyed on its dominant "
      "non-PASS class, so that a scenario that serves both faulty and healthy requests cannot move classes "
      "between splits. The code asserts that no trace and no scenario appears in two splits. This rule "
      "later exposed a real defect: an earlier per-class assignment silently emptied test classes in the "
      "small live corpus, which had produced an accuracy of 1.0 that meant nothing."),

    H2("Triage Model"),
    P("{A:triage} gives the inference procedure and {T:config} the configuration. The class weights in the "
      "loss are derived from class frequencies and softened by an exponent of 0.5. Full inverse-frequency "
      "weighting over-triggered ENV_FAULT, which has few examples, and merging ENV_FAULT into INFRA_FLAKE "
      "was rejected because the two map to different actions, quarantine and retry, and the distinction is "
      "the point of the project. Training uses the AdamW optimiser with a weight decay of 1e-4, a learning-rate "
      "scheduler that halves the rate when validation macro-F1 stops improving, label smoothing of 0.05, "
      "early stopping on validation macro-F1 and a fixed seed."),
    TAB("config", "Configuration of the triage model and its training",
        ["Item", "Value", "Item", "Value"],
        [["Token embedding", "48", "Dropout", "0.3"],
         ["Side embeddings (host, agent, depth, duration)", "8 each", "Batch size", "256 (length-bucketed)"],
         ["BiLSTM", "1 layer, 64 per direction", "Learning rate", "0.003"],
         ["Attention hidden size", "64", "Epochs (max) / patience", "20 / 4"],
         ["Max sequence length", "256 tokens", "Class-weight exponent", "0.5"],
         ["Vocabulary", "87 tokens", "Seed", "42"],
         ["Parameters", "125,430", "Training time (CPU)", "1,427 s"]],
        [5.6, 3.3, 3.8, 3.4]),
    ALG("triage", "Triage inference with attention pooling, drift check and policy", ALG2),
    P("Batches are bucketed by length. Random batching pads every sequence to the longest member of its "
      "batch and wastes most of the compute on a CPU, and the model packs each batch so that the recurrent "
      "layer never processes a padding step."),

    H2("Trust Engine"),
    P("The thresholds rise with the tier because a wrong autonomous action at T3 costs more than one at T1. "
      "{T:tiers} lists them with the actions each tier may execute, using the minimum tiers of {T:classes}. "
      "{A:trust} is the update executed each time the true outcome of a decision becomes known."),
    TAB("tiers", "Promotion thresholds, retention floors and permitted actions per tier",
        ["Tier", "Earned by (min. n, Wilson bound)", "Retention floor", "May execute"],
        [["T0 observe", "start", "n/a", "nothing; observation only"],
         ["T1 recommend", "15, 0.70", "0.60", "ESCALATE_TO_INFRA, BLOCK_MERGE_ALERT_DEV"],
         ["T2 act with rollback", "30, 0.85", "0.75", "adds ALLOW_MERGE, AUTO_RETRY_PIPELINE, ROLLBACK_CANARY"],
         ["T3 autonomous", "50, 0.93", "0.83", "adds QUARANTINE_NODE"]],
        [3.4, 3.8, 2.3, 6.6]),
    ALG("trust", "Trust-tier update on a ground-truth outcome", ALG3),
    P("Step 2 is deliberately asymmetric with step 5. One wrong autonomous action on a real defect is "
      "enough to lose a tier, while promotion needs sustained evidence. Step 3 implements the rule that only "
      "evidence earned since the last tier change counts."),

    H2("Root-Cause Localisation, Drift Guard and Release Gate"),
    P("The localiser, given a trace and its class, scores every span as z + 2.0 x error + 0.25 / (1 + depth), "
      "where z is the standardised excess of the span's duration over the baseline mean of its operation "
      "and is floored at zero. Operations that never occur in healthy traces fall back to a global "
      "baseline. The result is the top five spans and a postmortem containing the ranked table and a "
      "one-paragraph summary."),
    P("{A:gate} is the release gate. The drift guard treats a trace as in-distribution only if at most half "
      "of its operations are unknown to the model and the exporter attests that the trace's trace-level "
      "features follow the training definitions. The second condition exists because of a measurement "
      "reported in Section 5.6: vocabulary alone does not protect the model from stock exports."),
    ALG("gate", "Release gate with drift guard and tier-governed execution", ALG4),
    P("The canary component computes error rates and 95th-percentile latency for both sets and a health "
      "score of 0.6 x error component + 0.4 x latency component. It answers ROLLBACK when the Wilson "
      "intervals of the error rates do not overlap or when the latency ratio exceeds 2.0, HOLD when fewer "
      "than 30 canary traces exist, and PROMOTE when the score is at least 0.80. The security component "
      "builds the import graph of the project with the standard abstract syntax tree, follows installed "
      "dependency requirements, and marks a vulnerability reachable only if its package is imported "
      "directly or transitively from the service's entry points."),

    H2("Service Interface, Metrics and Actuator"),
    P("One function, used by the HTTP service, the command-line gate and the dashboard, performs the whole "
      "evaluation, so that all three behave identically. {T:api} lists the service interface. The metrics "
      "endpoint renders four series in the Prometheus text format: decisions by class and action, the "
      "number of autonomous decisions, the trust tier of each agent and its accuracy over scored "
      "decisions. The actuator receives the result of the gate, and for every autonomous decision it sets "
      "a status, posts the comment and, for ENV_FAULT or a rollback, runs the configured command with a "
      "validated target. A retry budget of two limits repeated INFRA_FLAKE retries before the decision "
      "escalates to a human."),
    TAB("api", "Service interface of the triage and release-gate service",
        ["Method and path", "Purpose"],
        [["POST /api/v1/triage", "Classify a trace given as tokens or spans; returns class, probabilities, action, evidence"],
         ["POST /api/v1/triage/jaeger", "Triage a Jaeger trace with the drift guard"],
         ["POST /api/v1/triage/ground-truth", "Report the true outcome; runs the trust engine and returns any tier change"],
         ["POST /api/v1/gate", "Full release gate: trace, canary sets, security verdict"],
         ["POST /api/v1/postmortem", "Ranked suspect spans and a Markdown postmortem"],
         ["GET /api/v1/agent/scorecard", "Measured accuracy and current tier of an agent"],
         ["GET /api/v1/security/verdict", "Stored reachability-aware vulnerability verdict"],
         ["GET /metrics", "Prometheus scrape endpoint"]],
        [5.8, 10.3]),

    H2("Live System and Dashboard"),
    P("The minishop services sleep for real and retry on failure, and each request returns a trace and "
      "an exact label. The same Jaeger adapter that serves exported traces converts them, and the model "
      "is trained on the adapter's own output, so training and serving use identical features. A browser "
      "dashboard served by the project's API framework runs the closed loop continuously. It shows the "
      "pipeline stages, the latest trace tree, the trust tier over time and the decision feed, lets an "
      "operator inject any fault or run the scripted schedule, and has further views for canary releases, "
      "root-cause analysis, the security gate, the real-data model, chaos results and the metrics endpoint."),
]
