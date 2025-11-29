#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
导出 ucimlrepo 数据集的 features / targets / combined 到本地目录。

依赖：
    pip install ucimlrepo pandas

功能说明：
1. fetch_ucirepo(id=891) 获取数据集对象（可用 --id 修改）
2. 保存：
   - metadata.json
   - features.csv（如存在）
   - targets.csv（如存在）
   - combined.csv（features + targets 合并，若二者都存在）
3. 若缺少其一，输出提示，不做复杂解析。

示例：
    python export_data.py
    python export_data.py --id 891 --outdir dataset_891 --overwrite

"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from ucimlrepo import fetch_ucirepo


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def ensure_outdir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            log(f"ERROR: 目录已存在：{path}，使用 --overwrite 允许覆盖。")
            sys.exit(1)
        else:
            log(f"目录已存在（允许覆盖）：{path}")
    else:
        os.makedirs(path, exist_ok=True)
        log(f"创建输出目录：{path}")


def save_metadata(ds, outdir: str) -> None:
    meta = getattr(ds, "metadata", {})
    path = os.path.join(outdir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"保存 metadata.json (键数={len(meta)})")


def export_features_targets(ds, outdir: str):
    data_obj = getattr(ds, "data", None)
    if data_obj is None:
        log("没有 ds.data 对象，无法导出。")
        return

    features = getattr(data_obj, "features", None)
    targets = getattr(data_obj, "targets", None)

    if features is None and targets is None:
        log("既没有 features 也没有 targets。可能该数据集需手动解析 original_files。")
        return

    if features is not None:
        f_path = os.path.join(outdir, "features.csv")
        try:
            features.to_csv(f_path, index=False)
            log(f"保存 features.csv shape={features.shape}")
        except Exception as e:
            log(f"保存 features.csv 失败：{e}")

    target_df = None
    if targets is not None:
        try:
            # 可能是 Series
            if isinstance(targets, pd.Series):
                target_df = targets.to_frame(name=targets.name or "target")
            else:
                target_df = targets
            t_path = os.path.join(outdir, "targets.csv")
            target_df.to_csv(t_path, index=False)
            log(f"保存 targets.csv shape={target_df.shape}")
        except Exception as e:
            log(f"保存 targets.csv 失败：{e}")

    # 合并
    if features is not None and target_df is not None:
        try:
            combined = features.copy()
            # 避免同名冲突
            rename_map = {}
            for c in target_df.columns:
                if c in combined.columns:
                    new_c = f"{c}_target"
                    rename_map[c] = new_c
            if rename_map:
                log(f"检测到同名列，重命名 targets 列：{rename_map}")
                target_df = target_df.rename(columns=rename_map)
            for c in target_df.columns:
                combined[c] = target_df[c]
            c_path = os.path.join(outdir, "combined.csv")
            combined.to_csv(c_path, index=False)
            log(f"保存 combined.csv shape={combined.shape}")
        except Exception as e:
            log(f"生成 combined.csv 失败：{e}")
    else:
        log("无法生成 combined.csv：features 或 targets 缺失。")


def main():
    parser = argparse.ArgumentParser(description="仅导出 ucimlrepo 数据集的 features/targets/combined。")
    parser.add_argument("--id", type=int, default=891, help="数据集 id (默认 891)")
    parser.add_argument("--outdir", type=str, default=None, help="输出目录 (默认 dataset_<id>)")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有目录")
    args = parser.parse_args()

    dataset_id = args.id
    outdir = args.outdir or f"dataset_{dataset_id}"

    log(f"开始获取数据集 id={dataset_id}")
    ensure_outdir(outdir, args.overwrite)

    try:
        ds = fetch_ucirepo(id=dataset_id)
    except Exception as e:
        log(f"获取数据集失败：{e}")
        sys.exit(1)

    # 基本信息确认
    meta = getattr(ds, "metadata", {})
    name = meta.get("name", "unknown")
    log(f"数据集名称：{name}")

    # 保存 metadata
    save_metadata(ds, outdir)

    # 导出 features / targets / combined
    export_features_targets(ds, outdir)

    log("完成。请查看输出目录。")


if __name__ == "__main__":
    main()