from blocks import *

BLOCKS = [
    CH("CONCLUSION AND FUTURE ENHANCEMENT"),
    P("This thesis presented an AI-augmented release gate that classifies the trace of a failed CI/CD run, "
      "maps the class to a pipeline action and decides how much of that action the agent may carry out "
      "alone. A data foundation recovers the causal order of events on real HDFS traces by clock alignment "
      "and splits them by scenario. An attention-pooled BiLSTM reaches a macro-F1 of 0.7464 on 57 unseen "
      "scenarios, above a count-based representation of 0.5423 and above the strongest order-free baseline "
      "of 0.7037, with no real defect missed. A trust engine converts the agent's measured accuracy into "
      "four autonomy tiers by a Wilson lower bound and withdraws autonomy at once when a real defect is "
      "merged. Supporting modules localise suspect spans, abstain outside the training domain, compare "
      "canary releases, decide which vulnerabilities are reachable and inject faults to test the agent."),
    P("The honest reading of the results is narrower than the headline. The credible accuracy is the HDFS "
      "figure with REAL_BUG excluded, 0.6829, which ties an order-free baseline, so the recurrent "
      "sequence model does not improve classification over counts and scalars. REAL_BUG's perfect score "
      "reflects trace length, about 40% of healthy traces are flagged, ENV_FAULT is found in fewer than "
      "40% of cases, and compound faults are mostly missed. Stock exports of the model's own traces are "
      "misjudged four times in five unless the drift guard abstains. The live system demonstrates the loop "
      "rather than the accuracy: the agent earned T3, lost it when a defect with no trace signature was "
      "merged, and recovered only through new evidence."),
    P("The achieved outcomes must be distinguished from the future objectives. Achieved are a documented "
      "architecture, a leakage-free data pipeline, a trained and baselined model, a statistically gated "
      "trust engine whose design was corrected by a live experiment, a release gate with explicit "
      "precedence, a closed-loop live system with an interactive dashboard, a verified push and read-back "
      "through a real Jaeger server, and 183 passing automated tests. Not achieved are execution of the "
      "GitHub workflows, the Prometheus and Grafana stack, real chaos on infrastructure, and any "
      "evaluation on production traces."),

    H2("Future Enhancements"),
    P("The enhancements below are proposed. They are not existing capabilities, and they are ordered by the "
      "limitations identified above."),
    BUL("**Train and evaluate on a real system.** Collect labelled traces from an application such as a "
        "microservice benchmark under real fault injection, with several independent scenarios per class, "
        "and report the result with the same per-class and baseline discipline."),
    BUL("**Reduce false alarms.** Attack the 40% of healthy traces that are flagged, with a shape-matched "
        "control for REAL_BUG, a calibrated confidence and a cost-sensitive threshold."),
    BUL("**Handle compound faults.** Replace the five-way output with a multi-label output so that "
        "simultaneous faults are not classified as healthy."),
    BUL("**Make explanations trustworthy.** Try a much weaker attention penalty or multi-head pooling, and "
        "validate evidence spans against labelled root causes instead of a proxy."),
    BUL("**Catch defects with no trace signature.** Add signals outside the trace, such as test results, "
        "business metrics or canary error budgets, because no trace-only model can see a silent regression."),
    BUL("**Run the unexecuted infrastructure.** Execute the workflows on a repository, bring up the full "
        "container stack, and run chaos with real fault injectors against it; keep the AI gate "
        "advisory until the agent has earned T2 in deployment."),
    BUL("**Persist trust in deployment.** Store trust state outside the process, feed the true outcome of "
        "every run back automatically, and alert when the agent is demoted."),
]

REFS = {
    "humble": "J. Humble and D. Farley, *Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*. Boston, MA, USA: Addison-Wesley, 2010.",
    "luo": "Q. Luo, F. Hariri, L. Eloussi, and D. Marinov, “An empirical analysis of flaky tests,” in *Proc. 22nd ACM SIGSOFT Int. Symp. Found. Softw. Eng. (FSE)*, 2014, pp. 643–653.",
    "dapper": "B. H. Sigelman et al., “Dapper, a large-scale distributed systems tracing infrastructure,” Google Inc., Tech. Rep. dapper-2010-1, 2010.",
    "otel": "OpenTelemetry Authors, “OpenTelemetry documentation.” [Online]. Available: https://opentelemetry.io",
    "jaeger": "Jaeger Authors, “Jaeger: Open source, end-to-end distributed tracing.” [Online]. Available: https://www.jaegertracing.io",
    "parasuraman": "R. Parasuraman, T. B. Sheridan, and C. D. Wickens, “A model for types and levels of human interaction with automation,” *IEEE Trans. Syst., Man, Cybern. A, Syst. Humans*, vol. 30, no. 3, pp. 286–297, May 2000.",
    "xtrace": "R. Fonseca, G. Porter, R. H. Katz, S. Shenker, and I. Stoica, “X-Trace: A pervasive network tracing framework,” in *Proc. 4th USENIX Symp. Netw. Syst. Design Implement. (NSDI)*, 2007, pp. 271–284.",
    "pinpoint": "M. Y. Chen, E. Kiciman, E. Fratkin, A. Fox, and E. Brewer, “Pinpoint: Problem determination in large, dynamic Internet services,” in *Proc. Int. Conf. Dependable Syst. Netw. (DSN)*, 2002, pp. 595–604.",
    "deeplog": "M. Du, F. Li, G. Zheng, and V. Srikumar, “DeepLog: Anomaly detection and diagnosis from system logs through deep learning,” in *Proc. ACM SIGSAC Conf. Comput. Commun. Secur. (CCS)*, 2017, pp. 1285–1298.",
    "logrobust": "X. Zhang et al., “Robust log-based anomaly detection on unstable log data,” in *Proc. 27th ACM Joint Meeting Eur. Softw. Eng. Conf. Symp. Found. Softw. Eng. (ESEC/FSE)*, 2019, pp. 807–817.",
    "he": "S. He, J. Zhu, P. He, and M. R. Lyu, “Experience report: System log analysis for anomaly detection,” in *Proc. IEEE 27th Int. Symp. Softw. Rel. Eng. (ISSRE)*, 2016, pp. 207–218.",
    "nedelkoski": "S. Nedelkoski, J. Cardoso, and O. Kao, “Anomaly detection from system tracing data using multimodal deep learning,” in *Proc. IEEE 12th Int. Conf. Cloud Comput. (CLOUD)*, 2019, pp. 179–186.",
    "liu": "P. Liu et al., “Unsupervised detection of microservice trace anomalies through service-level deep Bayesian networks,” in *Proc. IEEE 31st Int. Symp. Softw. Rel. Eng. (ISSRE)*, 2020, pp. 48–58.",
    "tracebench": "J. Zhou, Z. Chen, J. Wang, Z. Zheng, and W. Dong, “TraceBench: An open data set for trace-oriented monitoring,” in *Proc. IEEE 6th Int. Conf. Cloud Comput. Technol. Sci. (CloudCom)*, 2014, pp. 519–526.",
    "zhoux": "X. Zhou et al., “Fault analysis and debugging of microservice systems: Industrial survey, benchmark system, and empirical study,” *IEEE Trans. Softw. Eng.*, vol. 47, no. 2, pp. 243–260, Feb. 2021.",
    "bahdanau": "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” in *Proc. 3rd Int. Conf. Learn. Represent. (ICLR)*, 2015.",
    "schuster": "M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673–2681, Nov. 1997.",
    "lstm": "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.",
    "jain": "S. Jain and B. C. Wallace, “Attention is not explanation,” in *Proc. NAACL-HLT*, 2019, pp. 3543–3556.",
    "wiegreffe": "S. Wiegreffe and Y. Pinter, “Attention is not not explanation,” in *Proc. EMNLP-IJCNLP*, 2019, pp. 11–20.",
    "lamport": "L. Lamport, “Time, clocks, and the ordering of events in a distributed system,” *Commun. ACM*, vol. 21, no. 7, pp. 558–565, Jul. 1978.",
    "sre": "B. Beyer, C. Jones, J. Petoff, and N. R. Murphy, Eds., *Site Reliability Engineering: How Google Runs Production Systems*. Sebastopol, CA, USA: O’Reilly Media, 2016.",
    "sculley": "D. Sculley et al., “Hidden technical debt in machine learning systems,” in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 28, 2015, pp. 2503–2511.",
    "wilson": "E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927.",
    "brown": "L. D. Brown, T. T. Cai, and A. DasGupta, “Interval estimation for a binomial proportion,” *Statist. Sci.*, vol. 16, no. 2, pp. 101–133, 2001.",
    "guo": "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.",
    "basiri": "A. Basiri et al., “Chaos engineering,” *IEEE Softw.*, vol. 33, no. 3, pp. 35–41, May/Jun. 2016.",
    "ponta": "S. E. Ponta, H. Plate, and A. Sabetta, “Detection, assessment and mitigation of vulnerabilities in open-source dependencies,” *Empir. Softw. Eng.*, vol. 25, no. 5, pp. 3175–3215, 2020.",
    "friedman": "J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.",
    "pytorch": "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.",
    "sklearn": "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.",
    "prom": "Prometheus Authors, “Prometheus: Monitoring system and time series database.” [Online]. Available: https://prometheus.io",
    "opa": "Open Policy Agent Authors, “Open Policy Agent documentation.” [Online]. Available: https://www.openpolicyagent.org",
}

ABBREVIATIONS = [
    ("AI", "Artificial Intelligence"), ("API", "Application Programming Interface"),
    ("AST", "Abstract Syntax Tree"), ("BiLSTM", "Bidirectional Long Short-Term Memory"),
    ("CI/CD", "Continuous Integration and Continuous Delivery"),
    ("CPU", "Central Processing Unit"), ("CVE", "Common Vulnerabilities and Exposures"),
    ("DAG", "Directed Acyclic Graph"), ("GBM", "Gradient Boosting Machine"),
    ("HDFS", "Hadoop Distributed File System"), ("HTTP", "Hypertext Transfer Protocol"),
    ("JSON", "JavaScript Object Notation"), ("LB", "Lower Bound (Wilson)"),
    ("LSTM", "Long Short-Term Memory"), ("OOV", "Out-of-Vocabulary"),
    ("OPA", "Open Policy Agent"), ("OTLP", "OpenTelemetry Protocol"),
    ("PR", "Pull Request"), ("RCA", "Root-Cause Analysis"),
    ("ROC", "Receiver Operating Characteristic"), ("SRE", "Site Reliability Engineering"),
    ("SVM", "Support Vector Machine"), ("TF-IDF", "Term Frequency - Inverse Document Frequency"),
    ("UI", "User Interface"),
]
