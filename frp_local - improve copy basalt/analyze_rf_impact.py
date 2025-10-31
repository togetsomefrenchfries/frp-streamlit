#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析配置79、80、81的参数变化和对R²的影响
"""

print("RandomForest配置79、80、81的关键差异分析:")
print("=" * 60)

configs = [
    {
        'id': 79,
        'n_estimators': 80,
        'max_depth': 12,
        'min_samples_split': 5,
        'bootstrap': True,
        'min_samples_leaf': None  # 默认值1
    },
    {
        'id': 80, 
        'n_estimators': 80,
        'max_depth': 12,
        'min_samples_split': 6,
        'bootstrap': True,
        'min_samples_leaf': None  # 默认值1
    },
    {
        'id': 81,
        'n_estimators': 100,
        'max_depth': 3,  # 从12急剧降到3！
        'min_samples_split': 2,
        'bootstrap': False,  # 从True变为False！
        'min_samples_leaf': 1  # 显式设置
    }
]

print("参数对比:")
for config in configs:
    print(f"\n配置 {config['id']}:")
    for key, value in config.items():
        if key != 'id':
            print(f"  {key}: {value}")

print("\n关键变化分析:")
print("79 → 80:")
print("  - min_samples_split: 5 → 6 (轻微增加，对过拟合控制更严格)")
print("  - 其他参数相同")
print("  - 影响：可能略微降低R²，但变化不大")

print("\n80 → 81 (剧烈变化):")
print("  - n_estimators: 80 → 100 (+20树)")
print("  - max_depth: 12 → 3 (深度急剧减少！)")
print("  - min_samples_split: 6 → 2 (分裂要求放松)")
print("  - bootstrap: True → False (改变采样方式)")
print("  - 影响：这是一个完全不同的模型复杂度！")

print("\n模型复杂度分析:")
print("配置79-80: 高复杂度模型")
print("  - max_depth=12: 允许很深的树，可能过拟合")
print("  - bootstrap=True: 使用袋装法，增加随机性")

print("\n配置81: 低复杂度模型") 
print("  - max_depth=3: 非常浅的树，可能欠拟合")
print("  - bootstrap=False: 使用全部数据，减少随机性")
print("  - 这解释了R²突然下降的原因！")

print("\n建议修改:")
print("1. 避免参数的剧烈跳跃")
print("2. 在max_depth之间增加渐进过渡")
print("3. 将bootstrap变化与其他参数解耦")
print("4. 考虑重新排列参数组合顺序")