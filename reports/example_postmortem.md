# Postmortem: task `381A95EC78B4C4A0`

**Classification:** DATA_FAULT  **Action taken:** ESCALATE_TO_INFRA
**Scenario:** `AN_Data_corruptBlk_r_01FDN_30C_1to19B_0to120INT_15RT_5WT`  **Spans:** 145

## Top suspect spans

| rank | op | status | duration (us) | z-score | depth | score |
|---|---|---|---|---|---|---|
| 1 | `chooseDataNode` | OK | 1462997 | 5.051 | 2 | 5.134 |
| 2 | `checksumOk` | OK | 983487 | 1.57 | 2 | 1.653 |
| 3 | `chooseDataNode` | OK | 596662 | 1.349 | 2 | 1.432 |
| 4 | `RPC:getFileInfo` | OK | 8385577 | 0.647 | 2 | 0.73 |
| 5 | `RPC:getFileInfo` | OK | 7541524 | 0.521 | 2 | 0.604 |

## Summary
The highest-ranked suspect is `chooseDataNode` (z-score 5.051), 1462997us against the PASS-baseline for that operation. Investigate this span first.