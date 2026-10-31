# Feature scope

## Existing/core features (must-have)
- Federated hospital-client simulation (10 clients).
- Local training on separate data partitions.
- Model update gateway and payload validation.
- L2 norm and pairwise cosine inspection.
- Byzantine-robust aggregation: FedAvg baseline, coordinate median, trimmed mean.
- Controlled sign-flip and noise attacks.
- Quarantine state, trust score history and audit log.
- Held-out evaluation: accuracy, precision, recall, F1 and confusion matrix.
- Dashboard network topology and experiment controls.

## Differentiating features (add only if core is stable)
1. **Attack Replay Timeline:** replay each client update and show when the outlier emerged.
2. **Trust Decay + Recovery:** trust changes over repeated rounds; restoration requires clean rounds and operator approval.
3. **Non-IID Stress Test:** compare IID vs non-IID hospital partitions to show honest heterogeneity can look anomalous.
4. **Defence Benchmark:** run same seed across FedAvg, median and trimmed mean; show measured metric deltas and false-positive rate.
5. **Explainable Evidence Card:** show each node's L2 norm, cosine similarity, robust deviation and final decision.
6. **Round Integrity Snapshot:** hash model parameters and log the hash for tamper-evident experiment records (not a blockchain claim).
7. **Adaptive Threshold Calibration:** calibrate thresholds on clean validation rounds, freeze them before attack evaluation.

## Do not build in the 3-hour MVP
- Real hospital integrations or patient records.
- Clinical triage recommendations.
- Production identity/PKI or hospital-grade authentication.
- Blockchain.
- Complex multi-region Kubernetes.
- Autonomous permanent blacklisting.
- Claims of guaranteed Byzantine tolerance.
- LLM-based attack detection as the only detector.
- Large model architectures or massive datasets.
