# v004c Stage 2.3 Superseded

Status: SUPERSEDED

Historical tag:
`v004c-stage2-3-candidate-specs-0.1`

Historical commit:
`3eba1cbe06b3cc437541dd1451553d608e297cfc`

## Reason

The previous Stage 2.3 candidate freeze was retired because:

1. Candidate models were overly concentrated on D1 intraday structure.
2. MA5, MA10 and MA20 technical states were not adequately represented.
3. The design lacked a clear recent-seven-trading-day event path and D0-to-D1 transition structure.
4. M2 was primarily an expression-level sensitivity variant of M1.
5. The specification, audit and output architecture was disproportionate to the modelling task.
6. The final audit result was recorded but not uniformly enforced as a fail-closed release gate.
7. Finalization occurred after the temporary output directory had already been promoted.

## Still retained

- Stage 1 frozen dataset
- Stage 2.1 factor dictionary and leakage controls
- Stage 2.2 univariate research
- M0 intercept baseline concept
- Logistic regression with L2 regularization
- Train-fold-only preprocessing
- Exclusion of D2/D3, outcome fields and legacy model scores
- D1 structural factors as a future comparison baseline

This file does not define the replacement model specification.
