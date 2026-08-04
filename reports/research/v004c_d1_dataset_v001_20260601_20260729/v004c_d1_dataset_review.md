# v004c 阶段1 — D1 首次断板训练数据冻结与审计报告 (自动生成)

- 数据集版本: v004c-d1-dataset-0.1
- 来源数据集: v004c-dataset-0.2.2 (已验证源 manifest)
- 候选定义: two_or_three_board_first_break_d1
- 标签定义: daily_d2open_to_d3high_target7
- adjustment: none
- 生成时间: 2026-08-05T01:47:01+0800 (Asia/Shanghai)
- git root: F:\fenqixuangu
- git HEAD: 7f4db91e06513a08de382831d401c5cc4d5ea3c4

## 统一交易时序

D0 = 最后一个成功连板日; D1 = 首次断板日 (旧数据 stage_group=='d0' / days_since_break==0);
D2 = D1 后下一有效交易日 (早盘买入); D3 = D2 后下一有效交易日 (盘中检验 +7%)。

## git_status_after 语义

git_status_after = 本次构建所有非 manifest 输出文件写完后的 Git 状态
(git_dirty_before=True, git_dirty_after=True)。

## 冻结统计

| 指标 | 值 |
|---|---|
| 输入行数 | 879 |
| D1 输出行数 | 333 |
| 唯一 event_id | 333 |
| 唯一信号日 | 42 |
| Target7 正样本 | 100 (0.300300) |
| 六月 / 七月 | 173/160 行, 正样本 59/41 |
| 二板 / 三板 | 284/49 行, 正样本 81/19 |
| 重复 (event/code-break/code-signal) | 0/0/0 |
| 时间关系失败行 | 0 |
| Target7 重算不一致 | 0 (最大收益差异 0.0) |
| 交易日邻接 verified/mismatch/not_verified | 333/0/0 |
| 日线缓存股票数 / 文件数 | 296/296 |
| 泄漏违规列 | 0 |
| forbidden_unknown 列 | 0 |
| 冻结状态 | FROZEN |

## 声明

本阶段只冻结 D1 训练数据; 未筛选因子、未训练模型、未运行 walk-forward、
未修改 v004a/v005.1 冻结逻辑; 未自动 commit 或 push。
