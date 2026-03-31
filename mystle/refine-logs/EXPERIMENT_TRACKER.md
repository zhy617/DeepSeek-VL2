# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Bridge score computation | Bridge Score Module | 1k calibration | rank correlation, distribution | MUST | TODO | 3 calibration subsets |
| R002 | M1 | HC-SMoE baseline (25%) | HC-SMoE | InfoVQA/OCRBench/MMMU/MMBench | accuracy retention | MUST | TODO | |
| R003 | M1 | HC-SMoE baseline (50%) | HC-SMoE | same | accuracy retention | MUST | TODO | |
| R004 | M1 | MergeMoE baseline (25%) | MergeMoE | same | accuracy retention | MUST | TODO | |
| R005 | M1 | MergeMoE baseline (50%) | MergeMoE | same | accuracy retention | MUST | TODO | |
| R006 | M2 | Admissibility-gated merge (25%) | Ours | same | accuracy retention | MUST | TODO | 3 seeds |
| R007 | M2 | Admissibility-gated merge (50%) | Ours | same | accuracy retention | MUST | TODO | 3 seeds |
| R008 | M3 | Novelty: bridge-score gate | Bridge-score protection | InfoVQA/OCRBench | retention | MUST | TODO | fixed protection count |
| R009 | M3 | Novelty: routing-freq gate | Routing-frequency protection | same | retention | MUST | TODO | |
| R010 | M3 | Novelty: activation-sim gate | Activation-similarity protection | same | retention | MUST | TODO | |
| R011 | M3 | Novelty: random gate | Random protection | same | retention | MUST | TODO | 3 seeds |
| R012 | M3 | Novelty: no protection | All merge | same | retention | MUST | TODO | |
| R013 | M4 | Naive merge + Router KD | HC-SMoE + Router KD | same | retention | MUST | TODO | |
| R014 | M4 | Ours without Router KD | Ours (no KD) | same | retention | MUST | TODO | |
| R015 | M4 | Ours + Router KD | Ours + KD | same | retention | MUST | TODO | |
| R016 | M5 | Compression sweep 12.5% | Ours | bridge-sensitive | retention | NICE | TODO | |
| R017 | M5 | Compression sweep 37.5% | Ours | bridge-sensitive | retention | NICE | TODO | |
| R018 | M5 | Failure analysis | qualitative | 20-30 examples | categories | NICE | TODO | |
