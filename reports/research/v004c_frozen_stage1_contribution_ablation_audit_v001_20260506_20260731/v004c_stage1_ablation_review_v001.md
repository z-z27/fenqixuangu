# v004c Frozen Stage1 Contribution Ablation Audit v001

## Q1. A0 是否精确复现 current frozen Stage1？

YES. A0 reconstructs the accepted frozen snapshot within the fixed numeric tolerance and reproduces every deterministic daily rank. Rows/dates **497/62**; score max abs error **4.989e-13**; rank mismatch **0**; July identity mismatch **0**.

## Q2. 删除 D0 timing family 是否有稳定价值？

NO STABLE VALUE. A1 changed no Top3 membership on any of the 60 mature dates and left Top3 economic/risk results exactly unchanged; its date-equal Target7 AUC was also effectively unchanged.

| Period | AUC delta | Top3 excess delta | Top3 LOSS-excess delta |
|---|---:|---:|---:|
| MAY | 0.0000% | 0.0000% | 0.0000% |
| JUNE | -0.0195% | 0.0000% | 0.0000% |
| JULY_MATURE_ONLY | 0.0000% | 0.0000% | 0.0000% |
| MAY_JUNE_JULY_MATURE_ONLY | -0.0076% | 0.0000% | 0.0000% |

- Mature Top3 membership: changed dates/slots **0/0**; promoted T7/PNT/LOSS **0/0/0**; demoted T7/PNT/LOSS **0/0/0**; net Target7 slots **0**; net LOSS slots removed **0**.

## Q3. 进一步删除 total_score family 是否有稳定价值？

NO STABLE VALUE. A2 repaired July ranking and Top3 loss contamination, but reduced May/June Top3 excess and worsened June loss contamination. The benefit is therefore July-concentrated rather than cross-month.

| Period | AUC delta | Top3 excess delta | Top3 LOSS-excess delta |
|---|---:|---:|---:|
| MAY | -0.1003% | -0.0544% | 0.0000% |
| JUNE | -1.1306% | -0.2113% | 1.5873% |
| JULY_MATURE_ONLY | 6.0546% | 0.3073% | -3.1746% |
| MAY_JUNE_JULY_MATURE_ONLY | 1.2587% | 0.0173% | -0.5556% |

- Mature Top3 membership: changed dates/slots **12/12**; promoted T7/PNT/LOSS **4/7/1**; demoted T7/PNT/LOSS **7/3/2**; net Target7 slots **-3**; net LOSS slots removed **1**.

## Q4. 进一步删除 legacy interactions 是否有稳定价值？

MIXED POSITIVE SIGNAL. A3 improved date-equal Target7 AUC and Top3 capped excess in May, June, and mature July, but it added one net LOSS slot overall and worsened June Top3 LOSS excess/negative-date rate.

| Period | AUC delta | Top3 excess delta | Top3 LOSS-excess delta |
|---|---:|---:|---:|
| MAY | 2.1437% | 0.0787% | 0.0000% |
| JUNE | 2.3630% | 0.1354% | 1.5873% |
| JULY_MATURE_ONLY | 5.7165% | 0.2183% | -0.0000% |
| MAY_JUNE_JULY_MATURE_ONLY | 3.2495% | 0.1474% | 0.5556% |

- Mature Top3 membership: changed dates/slots **17/20**; promoted T7/PNT/LOSS **7/7/6**; demoted T7/PNT/LOSS **5/10/5**; net Target7 slots **2**; net LOSS slots removed **-1**.

## Q5. 是否存在“越简单反而跨 May/June/July 越稳定”的证据？

PARTIAL. A3 is the only challenger with same-direction AUC and return improvement in all three months, but the simplification sequence is not monotone (A1 is inert and A2 harms May/June) and risk quality does not improve consistently.

## Q6. 现有证据是否足以授权下一步只训练一个 Reduced V4C Stage1 Logistic？

NO. The A3 return/ranking pattern is post-hoc on consumed May--July data, economically modest, and conflicts with LOSS contamination. This audit does not provide sufficient authorization to fit a reduced Stage1 Logistic.

Research conclusion: `MIXED`

NEXT_ACTION = INSUFFICIENT_EVIDENCE
