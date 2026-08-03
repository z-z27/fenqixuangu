# -*- coding: utf-8 -*-
"""v004c fold 内动态训练权重工具 (只读工具; 不训练任何模型)

compute_fold_weights(train_df):
  输入必须只是当前训练 fold 的数据 (不得包含验证集/全数据集信息)。

  算法:
  1. 在输入 fold 内计算每个 event_id 的观察次数 event_count_i;
  2. 基础权重 base_weight_i = 1 / event_count_i;
  3. 对每个 signal_date 归一化: weight_i = base_weight_i / sum(base_weight_j, j 同 signal_date);
  4. 将 fold 内权重整体归一化, 使均值为 1: weight_i = weight_i / mean(weight)。

  约定:
  - 只使用输入 DataFrame 的 event_id 与 signal_date 两列;
  - 不读取验证集、全数据集事件次数、未来日期候选数量、CSV 中的 proposed_training_weight;
  - 函数不修改输入 DataFrame (内部 copy)。
"""
from __future__ import annotations

import pandas as pd

EVENT_COL = "event_id"
DATE_COL = "signal_date"


def compute_fold_weights(train_df: pd.DataFrame) -> pd.Series:
    """输入: 训练 fold DataFrame (含 event_id 与 signal_date 列)。
    输出: 与 train_df 行序一致的权重 Series (均值为 1)。"""
    if EVENT_COL not in train_df.columns or DATE_COL not in train_df.columns:
        raise ValueError(f"train_df 必须包含 {EVENT_COL} 与 {DATE_COL} 列")
    frame = train_df[[EVENT_COL, DATE_COL]].copy()
    if frame.empty:
        return pd.Series(dtype=float)

    # 1. fold 内每个事件的观察次数
    event_count = frame.groupby(EVENT_COL)[EVENT_COL].transform("count")
    # 2. 基础权重
    base_weight = 1.0 / event_count.astype(float)
    # 3. 按 signal_date 归一化 (每个日期内权重和 = 1)
    date_sum = base_weight.groupby(frame[DATE_COL]).transform("sum")
    weight = base_weight / date_sum
    # 4. fold 内均值归一化到 1
    mean_weight = float(weight.mean())
    weight = weight / mean_weight
    return weight.astype(float)
