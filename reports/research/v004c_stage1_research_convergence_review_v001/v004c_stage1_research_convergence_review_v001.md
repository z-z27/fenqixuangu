# v004c Stage1 Research Convergence Review v001

## 简单版结论

1. **现在卡在哪里？** 现有D1-known信息不能跨时间稳定地区分未来Target7与LOSS，Rank2/Rank3尤其容易把winner和LOSS混在一起。
2. **已经试过什么？** 18F/7F Logistic、训练规格、53F Ridge/GBDT、pairwise/Stage2、risk protector、Board3专模、D1 path、local regime、broad-market strength以及一次August regime-aware OOT都已正式覆盖。
3. **哪些路不用再试？** 继续调18F/7F参数、模型zoo、pairwise/PAIR_CAPPED7、risk protector、Board3-only/旧历史、recency weighting、close/VWAP×board4plus及broad-market ranking扩展均不再授权。
4. **还有没有真正没试过的？** 有且只有一个较窄候选：断板前2/3连板日自身的个股级封板/回封路径质量。它不是D1断板日path，也不是date-level板池生态；当前数据ready仅PARTIAL。
5. **下一步继续还是停？** 只授权一次blind information audit，先审coverage/lineage，再用固定描述检验；不授权任何模型。若该family也不成立，应停止当前Stage1 D1因子挖掘路线。

## 1. Frozen Goal and Pipeline

最终任务没有改变：D1收盘后，对v004c首次断板候选排序，用户关注Top3。Primary是 `Target7 = 1[D3 high / D2 open - 1 >= 7%]`；Secondary是在尽量不损失Target7的前提下控制LOSS与severe LOSS。

```text
V4C Board2/Board3 first-break candidate universe
    -> D1-known information
    -> Stage1 ranking
    -> Top3
    -> D2-open to D3-high outcome
```

当前主要问题位于 **D1-known information -> stable cross-sectional ranking**：候选池仍有oracle空间，但现有信息无法稳定把Target7推到Rank2/Rank3并把LOSS压下去。

## 2. Current Reference Baseline

S2仍然只是 `REFERENCE BASELINE`，不是successful model：

- 7F: rank_d1_close_ma10_pct, rank_d1_low_ma10_pct, rank_trend_hold_score, rank_theme_score, rank_log_candidate_base_price, rank_active_money_score, rank_d1_close_vwap_pct
- weighted L2 Logistic；L2=0.10；positive_weight=1.50；tail bonus=OFF。
- development: 2026-05-06..2026-07-29 mature as-of 2026-08-01，485 rows / 60 dates。
- L2、月份权重、tail、feature subset继续调整：全部 `NOT AUTHORIZED`。

## 3. Research Route Inventory

统一路线总表见 `v004c_research_route_inventory_v001.csv`，共 38 条。它区分了：正式negative、partial、data/provenance blocked、development-only positive和forward failure。关键证据链如下：

- 18F只获得PARTIAL，July current Stage1未击败universe。
- Reduced7F与S2修复了部分model-spec问题，但S2 T7-vs-LOSS concordance仅55.14%。
- 固定53F GBDT训练内AUC高而OOF Top3低于baseline，说明“更复杂模型”没有解决跨期泛化。
- Board3 overpromotion真实，但Board3 ablation、专模、risk tail和旧历史均未形成安全稳定方案。
- D1 path、repair-state、local regime在development有局部信息，但selection-aware或forward验证不成立。
- broad market strength只部分解释candidate environment，不解释S2 alpha与时间翻转。

## 4. Confirmed Findings

逐条裁决见 `v004c_confirmed_findings_v001.csv`。最重要的三点：

1. **Opportunity存在**：oracle gap明确；不是候选池完全没有winner。
2. **识别失败为主**：Top3 regret约73%-80%来自winner capture；S2有25/49可替换LOSS slots。
3. **信息稳定性不足**：7F、53F、local/broad context与非线性形式都没有建立稳定Rank2/Rank3 winner-loss separation。

其中“Rank1存在一定信息”和“broad market影响candidate environment”只能标 `PARTIALLY_CONFIRMED`，不能夸大成稳定alpha。

## 5. Failure Types

- `MODEL_SPEC_FAILURE`: 7F的L2=.30与legacy tail bonus；已由S2修复。
- `INFORMATION_FAILURE`: S2/53F不能稳定分开Target7与LOSS；当前主瓶颈。
- `TEMPORAL_INSTABILITY`: May/June/July的单因子与模型方向变化。
- `SELECTION_BIAS_RISK`: 88关系选最强后，close/VWAP×board4plus adjusted p=.3879。
- `FORWARD_FAILURE`: PAIR_CAPPED7、repair-state、current18F July与August challenger均未恢复稳定alpha。
- `DATA_LIMITATION`: Board3小样本；August complete mature仅8日。
- `LINEAGE_LIMITATION`: original-v4a strict provenance失败；broad-market历史universe形式不完美但M1-M3 materiality已证明robust。
- `HYPOTHESIS_NOT_SUPPORTED`: recency、overextension、veto/reranker等固定假设。

## 6. Do Not Repeat

`v004c_failed_routes_do_not_repeat_v001.csv`只收录有正式证据或明确停止合同的路线。特别是：

- 不把GBDT训练内拟合当作需要更多复杂模型的证据。
- 不把Board3局部问题当作全局解释。
- 不因local regime development关系而测试第二名关系。
- 不因broad market部分解释candidate environment而把它加进ranking。

## 7. Model vs Information

`MODEL_FORM_LIMITATION = NOT_SUPPORTED`。

已有信息若只是Logistic表达不够，固定GBDT、pairwise、repair-state interactions或Stage2至少应在OOF/forward中恢复一部分稳定Top3 alpha；事实没有发生。复杂模型有强train fit但弱OOF，说明不能把问题归为单纯model form。

`CURRENT_INFORMATION_LIMITATION = SUPPORTED`。

该判断由S2 pair concordance、raw-vs-rank审计、53F nonlinear OOF、跨月方向变化、local/broad regime结果和August方向性失败共同支持。

`BROADER_D1_INFORMATION_LIMITATION = [待核验]`，因为仍有一个机制上不同、尚未直接审计的pre-break board-day路径家族；在该家族完成盲审计前，不能宣称所有D1信息已穷尽。

## 8. Information Family Inventory

完整分类见 `v004c_information_family_inventory_v001.csv`。规则严格区分“字段存在”与“被直接验证”：

- 价格位置、趋势、资金/量能、D1 path、board history、local ecology、broad strength和target information均已有直接实验。
- 53F的存在本身不等于每个字段独立通过；这里只按正式family级模型/信息审计定性。
- point-in-time主题成员、auction/order-book、历史size/liquidity/index/limit-down属于 `DATA_NOT_AVAILABLE`，不进入候选清单。
- 唯一 `NOT_DIRECTLY_TESTED` 且满足低复杂度条件的是 `PRE_BREAK_BOARD_DAY_SEAL_PATH`。

## 9. The Only Authorized Gap

`HIGH_VALUE_UNTESTED_INFORMATION_FAMILY = PRE_BREAK_BOARD_DAY_SEAL_PATH`。

中文定义：**断板前2/3连板日的个股级封板/回封路径质量**。它与既有失败路线本质不同，因为既有D1 path描述的是“断板当天发生了什么”，local ecology描述的是“当天整个板池怎样”，而该family描述的是“这只股票在进入断板日以前，连续涨停形成过程的封板质量与潜在供给”。

仅授权：固定字段定义前的blind coverage/lineage audit，以及随后一次预注册information audit。禁止模型、阈值、窗口搜索和自动字段生成。

## 10. Conflicts and Evidence Evolution

- 仓库内未找到可读的 `FACTOR_ANALYSIS_4_STATE` 最新状态总结；已执行filename/content检索。故本报告完全按正式review裁决。
- local regime conditional audit在development给出EXPLAINS，但单关系selection correction只到SUPPORTED_BUT_SELECTION_RISK，August方向反转且challenger不利；最终不能写成已解决ranking。
- broad-market coverage最初为LINEAGE_RISK，后续sensitivity证明M1-M3 aggregate robust；这是风险被量化解决，不是报告冲突。M4仍不可用。
- Reduced7F temporal SUPPORTED与后续INFORMATION_LIMIT_DOMINANT不冲突：前者是相对18F改善，后者指出绝对winner-loss separation仍不足。
- August正式enum是SAMPLE_INSUFFICIENT，但方向性结果negative；两者必须同时保留。

## 11. Final Verdict

CURRENT_STAGE1_BOTTLENECK = 现有D1-known表征无法跨时间稳定地区分Target7与LOSS，尤其在Rank2/Rank3发生winner-loss混排。

MODEL_FORM_LIMITATION = NOT_SUPPORTED

CURRENT_INFORMATION_LIMITATION = SUPPORTED

BROADER_D1_INFORMATION_LIMITATION = [待核验]

HIGH_VALUE_UNTESTED_INFORMATION_FAMILY = PRE_BREAK_BOARD_DAY_SEAL_PATH

STAGE1_RESEARCH_STATE = ONE_NARROW_INFORMATION_AUDIT_JUSTIFIED

NEXT_ACTION = BLIND_PRE_BREAK_BOARD_DAY_SEAL_PATH_INFORMATION_AUDIT

AUGUST_HOLDOUT_STATUS = CONSUMED

后续任何August使用只能标 `DEVELOPMENT / AUXILIARY / HISTORICAL`，不能再称fresh/untouched OOT。
