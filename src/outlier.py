#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BRFSS 糖尿病健康指标（combined.csv）异常值检测脚本（仅 IQR 法）

功能概述：
- 针对连续变量（BMI, MentHlth, PhysHlth）使用 IQR 法检测异常值（默认 k=1.5）
- 生成：
    - outliers_flagged.csv：带标记列的 DataFrame
    - outliers_summary.csv：每列异常数量汇总
    - outliers_indices.csv：异常行索引长表
    - outliers_meta.json：检测参数与阈值
- 终端输出：每个变量的异常值数量简要表

输出目录：项目文件夹内的 output/ （自动创建）

使用方式：
    python src/outlier.py --input dataset_891/combined.csv --outdir output
可选参数：
    --iqr_k 1.5
"""
import argparse
import os
import sys
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONTINUOUS = ["BMI", "MentHlth", "PhysHlth"]


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def detect_outliers_iqr(df: pd.DataFrame, cols: List[str], k: float = 1.5) -> Tuple[Dict[str, pd.Series], Dict[str, Tuple[float, float]]]:
    """
    IQR 法检测异常值。返回：
    - 每列的布尔掩码（True=异常）
    - 每列的上下界阈值 (lower, upper)
    """
    mask_map = {}
    bounds = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            mask_map[c] = pd.Series(False, index=df.index)
            bounds[c] = (np.nan, np.nan)
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask = ((s < lower) | (s > upper)) & ~s.isna()
        mask_map[c] = mask
        bounds[c] = (float(lower), float(upper))
    return mask_map, bounds


def summarize_masks(mask_groups: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
    """
    汇总掩码为表格，统计每列的异常数量。
    """
    rows = []
    for method, masks in mask_groups.items():
        for c, m in masks.items():
            rows.append({
                "method": method,
                "column": c,
                "n_outliers": int(m.sum()),
                "pct_outliers": float(m.mean() * 100.0)
            })
    summary = pd.DataFrame(rows).sort_values(["column", "method"]).reset_index(drop=True)
    return summary


def masks_to_long_indices(mask_groups: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
    """
    将掩码转成长表：method, column, index（异常行索引）
    """
    records = []
    for method, masks in mask_groups.items():
        for c, m in masks.items():
            idxs = m[m].index.tolist()
            for idx in idxs:
                records.append({"method": method, "column": c, "row_index": idx})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="combined.csv 异常值检测（仅 IQR 法）")
    parser.add_argument("--input", type=str, default="dataset_891/combined.csv", help="输入 CSV 文件（默认 dataset_891/combined.csv）")
    parser.add_argument("--outdir", type=str, default="output", help="输出目录（默认 output）")
    parser.add_argument("--iqr_k", type=float, default=1.5, help="IQR 方法的 k（默认 1.5）")
    args = parser.parse_args()

    in_path = args.input
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(in_path):
        print(f"[ERROR] 输入文件不存在：{in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"[INFO] 载入数据：{in_path}, 形状={df.shape}")

    # 连续变量列表（默认 BMI, MentHlth, PhysHlth，若不存在则跳过）
    continuous_cols = [c for c in DEFAULT_CONTINUOUS if c in df.columns]
    print("[INFO] 检测连续变量：", continuous_cols)

    if not continuous_cols:
        print("[WARNING] 无连续变量可检测，退出。")
        return

    # IQR 检测
    iqr_masks, iqr_bounds = detect_outliers_iqr(df, continuous_cols, k=args.iqr_k)

    # 汇总
    mask_groups = {
        "iqr_outlier": iqr_masks
    }
    summary = summarize_masks(mask_groups)
    indices_long = masks_to_long_indices(mask_groups)

    # 终端简要输出
    print("\n[OUTLIER SUMMARY]")
    print(summary.to_string(index=False))

    # 生成带标记的 DataFrame
    flagged_df = df.copy()
    for method, masks in mask_groups.items():
        for c, m in masks.items():
            col_flag = f"{c}__iqr_outlier"
            flagged_df[col_flag] = m.astype(int)

    # 汇总异常标识
    flag_cols = [c for c in flagged_df.columns if c.endswith("__iqr_outlier")]
    flagged_df["__any_outlier_flag"] = (flagged_df[flag_cols].sum(axis=1) > 0).astype(int) if flag_cols else 0

    # 保存输出
    flagged_path = os.path.join(outdir, "outliers_flagged.csv")
    summary_path = os.path.join(outdir, "outliers_summary.csv")
    indices_path = os.path.join(outdir, "outliers_indices.csv")
    meta_path = os.path.join(outdir, "outliers_meta.json")

    flagged_df.to_csv(flagged_path, index=False)
    summary.to_csv(summary_path, index=False)
    indices_long.to_csv(indices_path, index=False)

    meta = {
        "input": in_path,
        "shape": list(df.shape),
        "continuous_cols": continuous_cols,
        "params": {"iqr_k": args.iqr_k},
        "iqr_bounds": {k: {"low": v[0], "high": v[1]} for k, v in iqr_bounds.items()},
        "flag_columns_suffix_meaning": {
            "__iqr_outlier": "IQR方法判定为异常",
            "__any_outlier_flag": "行级别存在异常"
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 已保存到 {outdir}/:")
    print(f"  - outliers_flagged.csv")
    print(f"  - outliers_summary.csv")
    print(f"  - outliers_indices.csv")
    print(f"  - outliers_meta.json")


if __name__ == "__main__":
    main()