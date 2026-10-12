# Router Score v3 SHADOW Specification

Status: `RESEARCH SHADOW`  
Policy version: `risk-policy-v8-nq-rule-stress-score-exposure-v3-shadow`  
Score source: `nq-rule-stress-score-v2-w756-m252`

This candidate changes only the long-exposure mapping and the historical cash-return assumption. It does not replace the public policy.

## 1. Risk score

Use Router Stress Score v2 without modification. The score remains causal, integer-valued from 0 to 100, and is computed only from information available on the decision date.

## 2. Base long exposure

Map the score to the following frozen anchors and linearly interpolate between adjacent anchors. Round half up to a whole percentage point.

| Risk score | Base long exposure |
|---:|---:|
| 0 | 100% |
| 20 | 100% |
| 40 | 90% |
| 60 | 80% |
| 80 | 40% |
| 100 | 25% |

These anchors are a single pre-registered candidate. Do not tune individual anchors after viewing its historical result.

## 3. Fast Overlay and recovery

Keep the v2 rules unchanged:

- Fast Overlay: when at least 3 of 8 channels trigger, subtract 20 percentage points once; floor at 25%.
- Risk reduction: apply immediately.
- Recovery confirmation: require three eligible decision days with a non-deteriorating raw target.
- Re-entry ladder: 25% -> 40% -> 60% -> 80% -> 100%, one rung per eligible day and never above the current raw target.
- Recovery episode completion: recommended exposure equals the raw target, the raw target is at least 90%, and Fast Overlay is clear.
- When no recovery episode is active, follow a benign raw-target increase immediately.

## 4. Historical residual-cash return

- Series: FRED/ALFRED `DGS3MO`, stored as `CASH_3M` in `macro_observations`.
- At each holding-period start, use only the latest observation vintage available by that date.
- Convert the annual percentage yield to a simple ACT/365 holding-period return.
- Portfolio gross return:

  `long exposure * NQ return + (1 - long exposure) * cash return`

Missing point-in-time cash coverage is a hard failure. No future vintage or zero-return fallback is allowed for v3.

## 5. Evaluation

- Benchmark proxy: `NQ=F`.
- Execution: signal at close t, exposure from the next available NQ return bar.
- Transaction-cost scenarios: gross and 10bp only.
- Role: advisory/research reference, not an executable trading strategy.
- Use a version-specific output directory, for example `outputs/policy_rule_score_v3`.

Historical results are descriptive development evidence. Promotion still requires a frozen forward shadow period and a separate approval decision.
