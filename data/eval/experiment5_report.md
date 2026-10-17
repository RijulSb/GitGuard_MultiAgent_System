# GitGuard Experiment 5: Baseline vs. Multi-Agent Pipeline

## Executive Summary
Experiment 5 measures the performance gain of the 10-agent LangGraph system over a standard single-pass RAG baseline on the locked Pro Git evaluation benchmark (72 questions).

### Key Findings:
- **Correctness**: Pipeline achieved **0.4375** vs Baseline **0.9583**.
- **Faithfulness**: Pipeline achieved **1.0000** vs Baseline **1.0000**.
- **Citation Accuracy**: Pipeline achieved **0.9792** vs Baseline **1.0000**.
- **Unanswerable Refusal Rate**: Pipeline achieved **100.0%** vs Baseline **100.0%**.
- **Latency**: Pipeline averaged **32631.9 ms** vs Baseline **11267.0 ms**.

## Full Metrics Comparison Table

| metric                    |   baseline |   multi_agent_pipeline |   absolute_diff |   percentage_change |
|:--------------------------|-----------:|-----------------------:|----------------:|--------------------:|
| faithfulness              |     1      |                 1      |          0      |                0    |
| correctness               |     0.9583 |                 0.4375 |         -0.5208 |              -54.35 |
| citation_accuracy         |     1      |                 0.9792 |         -0.0208 |               -2.08 |
| unanswerable_refusal_rate |     1      |                 1      |          0      |                0    |
| false_refusal_rate        |     0.0167 |                 0.6333 |          0.6166 |             3692.22 |
| abstention_accuracy       |     0.9861 |                 0.4722 |         -0.5139 |              -52.11 |
| latency_ms                | 11267      |             32631.9    |      21364.9    |              189.62 |
| tokens                    |  1562.5    |              3037.2    |       1474.7    |               94.38 |
