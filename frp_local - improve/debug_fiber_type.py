#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速调试纤维类型数据
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 加载数据
file_path = "E:/大学/intern/2025-summer-concret/database 4.xlsx"
df = pd.read_excel(file_path, header=None)
df_filtered = df[df.iloc[:, 0] != 0].copy()

print(f"数据形状: {df_filtered.shape}")

# 检查位置8的数据（Glass_or_Basalt）
fiber_col = df_filtered.iloc[:, 8]
print(f"位置8的数据类型: {fiber_col.dtype}")
print(f"位置8的前20个值:")
print(fiber_col.head(20).tolist())

print(f"\n位置8的唯一值:")
unique_values = fiber_col.unique()
print(unique_values[:20])  # 显示前20个唯一值

print(f"\n位置8的值计数:")
value_counts = fiber_col.value_counts()
print(value_counts.head(10))

print(f"\n检查是否包含Glass字符串:")
fiber_series = fiber_col.astype(str).str.lower()
glass_mask = fiber_series.str.contains('glass', na=False)
basalt_mask = fiber_series.str.contains('basalt', na=False)

print(f"包含'glass'的行数: {glass_mask.sum()}")
print(f"包含'basalt'的行数: {basalt_mask.sum()}")

if glass_mask.sum() > 0:
    print(f"包含'glass'的前10个值:")
    glass_values = fiber_series[glass_mask].head(10)
    print(glass_values.tolist())

if basalt_mask.sum() > 0:
    print(f"包含'basalt'的前10個值:")
    basalt_values = fiber_series[basalt_mask].head(10)
    print(basalt_values.tolist())