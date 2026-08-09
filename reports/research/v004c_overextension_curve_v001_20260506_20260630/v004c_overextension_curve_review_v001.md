# v004c Overextension-Aware v4a Curvature Benchmark v001

CAPABILITY_SCREEN_ONLY — ONE PREDECLARED CURVATURE MODEL — NOT A PRODUCTION MODEL

## 1. Did Limited Curvature Fix the v4a Transfer Failure?

| Model | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Cross-Threshold | LOW Top3 Excess |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CONTROL | 2.8393% | 3.2031% | -0.3025% | 3.0212% | 1.8539% | 0.4871 | -1.7133% |
| CURVE | 3.1929% | 3.5841% | -1.0739% | 3.3885% | 1.8539% | 0.4855 | -1.7133% |
| UNIVERSE | 2.7596% | 2.7596% | 2.6575% | 2.7596% | 2.6575% | NA | 0.0000% |

## 2. Did Rank3 Recover?

- Control Rank3: **-0.3025%**
- Curve Rank3: **-1.0739%**
- Delta: **-0.7714%**
- Universe Rank3: **2.6575%**

## 3. Did We Preserve Rank1 and Rank2?

- Rank1 delta: **0.3536%**; preserved: **YES**
- Rank2 delta: **0.3810%**; preserved: **YES**

## 4. Did Cross-Threshold Winner Separation Improve?

| Model | All | Cross Threshold | Within Non-Target | Within Target7 |
| --- | --- | --- | --- | --- |
| CONTROL | 0.5093 | 0.4871 | 0.5212 | 0.6275 |
| CURVE | 0.5223 | 0.4855 | 0.5292 | 0.6484 |

- Cross-threshold delta: **-0.0016**

## 5. Did Low-Opportunity Downside Improve?

| Model | Rank1 | Top2 | Top3 | Top3 Excess | Negative Rate | Median | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CONTROL | -0.3685% | 1.7422% | -0.9211% | -1.7133% | 71.4286% | -0.6462% | -6.7688% |
| CURVE | 0.6748% | 1.7422% | -0.9211% | -1.7133% | 71.4286% | -0.6462% | -6.7688% |

- LOW Top3 excess improvement: **0.0000%**

## 6. Did High-Opportunity Upside Survive?

| Model | Rank1 | Top2 | Top3 | Top1 Target7 | Top3 Precision |
| --- | --- | --- | --- | --- | --- |
| CONTROL | 5.6103% | 3.6574% | 4.6139% | 42.8571% | 44.4444% |
| CURVE | 5.6103% | 4.6571% | 4.6139% | 42.8571% | 44.4444% |

- Upside preserved under the predeclared 0.25pp tolerance: **YES**

## 7. What Curvature Did the Model Actually Learn?

| Term | Negative Fold % | Median Beta | Valid Turning % | Median Turning Point |
| --- | --- | --- | --- | --- |
| quad_close_ma10 | 0.0000% | 0.035313 | 0.0000% | NA |
| quad_low_ma10 | 0.0000% | 0.056810 | 0.0000% | NA |
| quad_trend_hold | 0.0000% | 0.025310 | 0.0000% | NA |
| quad_close_vwap | 0.0000% | 0.030301 | 0.0000% | NA |
| quad_log_base_price | 0.0000% | 0.055727 | 0.0000% | NA |

- CURVATURE_STRUCTURAL_SUPPORT: **NO**

## 8. Which Stocks Moved?

- CONTROL Top3 false positives: **44**
- False-positive demotion rate: **0.0000%**
- CONTROL Top3 true positives: **18**
- True-positive preservation rate: **94.4444%**
- Newly promoted winners: **1**
- Newly promoted false positives: **0**
- Extreme-strength rows: **11**
- Extreme-strength median rank change: **0.0000**

## 9. Formal Decision

- Frozen CONTROL parity: **PASS**
- Q1 preserve Rank1: **YES**
- Q2 preserve Rank2: **YES**
- Q3 Rank3 materially recovered: **NO**
- Q4 STRICT Top3 materially improved: **NO**
- Q5 cross-threshold >0.50 and improved: **NO**
- Q6 LOW-opportunity downside improved: **NO**
- Q7 HIGH-opportunity upside preserved: **YES**
- Q8 stable high-end negative curvature: **NO**
- Q9 non-monotonic hypothesis: **NOT_SUPPORTED**

OVEREXTENSION_CURVATURE_SIGNAL: **ABSENT**

READY_FOR_JULY_OOT: **NO**

- July actually run: **NO**
- New raw data: **NO**
- New external factors: **NO**
- Hyperparameter search: **NO**
- Feature selection: **NO**

Limited quadratic overextension modeling did not solve the v4a-transfer failure.
