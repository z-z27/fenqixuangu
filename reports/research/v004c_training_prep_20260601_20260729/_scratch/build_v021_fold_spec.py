# -*- coding: utf-8 -*-
"""v004c walk-forward fold 规格 (只读, 不运行模型)

- 基于 v0.2.1 TRAIN_PRIMARY 的唯一 signal_date, 按时间顺序扩展式 fold
- 初始训练 >= 14 个信号日; 验证窗 6-8 日(最后一个窗口因样本末尾压缩, 已记录);
- 训练/验证边界 purge >= 2 个交易日; 同一 event_id 不得跨训练/验证
- 每个 fold 列出四个模型头的训练行数/事件数/正样本数
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
OUT = ROOT / "reports" / "research" / "v004c_training_prep_20260601_20260729"

train = pd.read_csv(OUT / "v004c_training_primary_v021.csv", dtype={"code": str})
train["code"] = train["code"].astype(str).str.zfill(6)
T7 = "target7_daily_d2open_d3high"
TL = "tail_loss_daily_5pct"

dates = sorted(train["signal_date"].unique())
print("signal dates:", len(dates), dates[0], "->", dates[-1])

# 日期索引
dpos = {d: i for i, d in enumerate(dates)}
train = train.assign(_dpos=train["signal_date"].map(dpos))

INITIAL_TRAIN = 14
PURGE = 2
VALID_WINDOW = 6

# 折叠布局 (索引): train_until, valid_start, valid_end
layout = []
cursor = INITIAL_TRAIN  # train 覆盖 [0, cursor)
while cursor < len(dates):
    valid_start = cursor + PURGE
    valid_end = min(valid_start + VALID_WINDOW, len(dates))
    if valid_end <= valid_start:
        break
    layout.append((cursor, valid_start, valid_end))
    cursor = valid_end
print("fold layout (train_until, valid_start, valid_end):", layout)

HEADS = [("d0_target", "d0", T7), ("d0_tail", "d0", TL),
         ("post_target", "post", T7), ("post_tail", "post", TL)]

rows = []
for fold_idx, (train_until, vs, ve) in enumerate(layout, start=1):
    tr = train[train["_dpos"] < train_until]
    va = train[(train["_dpos"] >= vs) & (train["_dpos"] < ve)]
    # 事件隔离检查
    tr_events = set(tr["event_id"])
    va_events = set(va["event_id"])
    overlap = tr_events & va_events
    purged_dates = dates[train_until:vs]
    for head_name, stage, target in HEADS:
        tr_h = tr[tr["stage_group"] == stage]
        va_h = va[va["stage_group"] == stage]
        if stage == "post":
            tr_h = tr_h[tr_h["post_day"].notna()]
            va_h = va_h[va_h["post_day"].notna()]
        train_pos = int(tr_h[target].fillna(False).sum())
        note = "standard" if (ve - vs) >= 5 else "final_window_compressed_by_sample_end"
        if train_pos < 40:
            note += f";positive_sufficiency:train_{head_name}_positives={train_pos}<40(建议下限), 正式训练时需调整边界或降权该fold评估"
        rows.append({
            "fold": fold_idx,
            "model_head": head_name,
            "train_signal_date_count": int(tr["signal_date"].nunique()),
            "train_row_count": int(len(tr_h)),
            "train_event_count": int(tr_h["event_id"].nunique()),
            "train_positive_count": train_pos,
            "train_negative_count": int(len(tr_h) - train_pos),
            "valid_signal_date_count": int(va["signal_date"].nunique()),
            "valid_row_count": int(len(va_h)),
            "valid_event_count": int(va_h["event_id"].nunique()),
            "valid_positive_count": int(va_h[target].fillna(False).sum()),
            "purge_date_count": len(purged_dates),
            "purge_dates": ",".join(purged_dates),
            "event_overlap_train_valid": len(overlap),
            "boundary_note": note,
        })

fold_df = pd.DataFrame(rows)
fold_df.to_csv(OUT / "v004c_walk_forward_fold_spec.csv", index=False, encoding="utf-8-sig")
print("fold spec rows:", len(fold_df))
print("\n== per fold summary (d0_target) ==")
print(fold_df[fold_df["model_head"] == "d0_target"][
    ["fold", "train_row_count", "train_event_count", "train_positive_count",
     "valid_row_count", "valid_positive_count", "purge_date_count",
     "event_overlap_train_valid", "boundary_note"]].to_string(index=False))

# 正样本充足性检查: 训练正样本不足时记录 (>= 40 为建议下限)
print("\n== positive sufficiency check (train_positive_count) ==")
for h in HEADS:
    hh = fold_df[fold_df["model_head"] == h[0]]
    print(f"{h[0]}: train positives per fold = {hh['train_positive_count'].tolist()}")
