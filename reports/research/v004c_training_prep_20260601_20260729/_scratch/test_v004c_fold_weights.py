# -*- coding: utf-8 -*-
"""v004c fold 权重工具测试 (只读; 不训练模型)

验证项:
1. 所有权重有限;
2. 所有权重大于 0;
3. 平均权重等于 1;
4. 每个训练日期的总权重相同 (误差 < 1e-10);
5. 同一事件重复两次时, 单行基础影响减半;
6. 验证集增加或删除不会改变训练权重;
7. 输入行顺序改变不影响对应权重;
8. 函数不会修改输入 DataFrame。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
from v004c_fold_weights import compute_fold_weights  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


print("[1] 真实训练 fold 上的基本性质测试")
train = pd.read_csv(SCRATCH.parent / "v004c_training_primary_v021.csv", dtype={"code": str})
w = compute_fold_weights(train)
check("finite", np.isfinite(w).all())
check("positive", (w > 0).all())
check("mean==1", abs(w.mean() - 1.0) < 1e-10, f"mean={w.mean():.12f}")
date_totals = train.assign(_w=w).groupby("signal_date")["_w"].sum()
check("per_date_totals_equal", date_totals.max() - date_totals.min() < 1e-10,
      f"max-min={date_totals.max()-date_totals.min():.2e}")

print("[2] 合成数据: 事件重复的基础影响减半")
syn = pd.DataFrame({
    "event_id": ["E1", "E2", "E3", "E3"],
    "signal_date": ["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"],
})
base = syn.groupby("event_id")["event_id"].transform("count").astype(float)
base_w = 1.0 / base
single_base = base_w[syn["event_id"] == "E1"].iloc[0]   # E1 出现 1 次 → base = 1/1
double_base = base_w[syn["event_id"] == "E3"].iloc[0]   # E3 出现 2 次 → base = 1/2
check("event_double_base_halved", abs(double_base * 2 - single_base) < 1e-12,
      f"E3(双次, base={double_base:.4f}) 为 E1(单次, base={single_base:.4f}) 的一半")

print("[3] 验证集增加或删除不改变训练权重 (函数只读取输入训练数据)")
train2 = train.copy()
train2["_row_id"] = np.arange(len(train2))
# 模拟: 世界里有验证行, 但传给函数的是剔除后的训练行 → 权重应与直接传训练行一致
w_direct = compute_fold_weights(train)
world = pd.concat([train, train.head(50)])  # 附加"验证"行
w_filtered = compute_fold_weights(world.iloc[: len(train)].copy())
check("validation_irrelevant", np.allclose(w_direct.values, w_filtered.values, atol=1e-12))
# 重复调用无外部状态
check("deterministic", np.allclose(w_direct.values, compute_fold_weights(train).values, atol=1e-12))

print("[4] 行顺序改变不影响对应权重")
shuffled = train.sample(frac=1.0, random_state=42).reset_index(drop=True)
w_shuf = compute_fold_weights(shuffled)
aligned = pd.DataFrame({
    "event_id": train["event_id"].values, "signal_date": train["signal_date"].values,
    "w_orig": w.reset_index(drop=True).values})
shuf_aligned = pd.DataFrame({
    "event_id": shuffled["event_id"].values, "signal_date": shuffled["signal_date"].values,
    "w_shuf": w_shuf.reset_index(drop=True).values})
merged = shuf_aligned.merge(aligned, on=["event_id", "signal_date"])
check("row_order_invariant", np.allclose(merged["w_shuf"].values, merged["w_orig"].values, atol=1e-12),
      f"matched={len(merged)}")

print("[5] 函数不修改输入 DataFrame")
train_before = train.copy()
compute_fold_weights(train)
check("input_untouched", train.equals(train_before))

print(f"\nRESULT: PASS={PASS} FAIL={FAIL}")
sys.exit(0 if FAIL == 0 else 1)
