# 90-second judge demo
1. **Context (10s):** Ten hospitals collaboratively train a shared model; raw data remains partitioned in the local simulation.
2. **Baseline (15s):** Run `No attack + FedAvg`; note measured test metrics.
3. **Attack (20s):** Run `Sign flip + FedAvg`, targeting Hospital 07. Show update norms, cosine similarity and any detected flags.
4. **Defence (20s):** Run `Sign flip + coordinate median` or trimmed mean. Compare metrics. Explain that robust aggregation reduces some outlier influence but is not a universal guarantee.
5. **Trust/audit (15s):** Show quarantine status and persisted audit events.
6. **Close (10s):** This is a controlled research prototype; clinical deployment requires external validation, privacy/security review and clinical governance.

Tip: Repeat each scenario with multiple random seeds before presenting a performance claim. Never claim detection if the engine did not flag the injected client.
