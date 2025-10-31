#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析RandomForest配置79、80、81的具体参数差异
"""

from itertools import product

# 复制配置生成逻辑
n_estimators_list = [50, 80, 100, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 800]
max_depth_list = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, None]
min_samples_split_list = [2, 3, 4, 5, 6, 8, 10, 12]
min_samples_leaf_list = [1, 2, 3, 4, 5, 6, 8, 10]
max_features_list = ['sqrt', 'log2', None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# 创建参数组合
base_combinations = list(product(
    n_estimators_list[:10],  # 取前10个n_estimators  
    max_depth_list[:8],      # 取前8个max_depth
    min_samples_split_list[:5],  # 取前5个min_samples_split
))

# 从400个组合中选择150个
selected_combinations = base_combinations[:150]

print("RandomForest配置79、80、81的参数分析:")
print("=" * 60)

# 分析配置79、80、81 (索引78、79、80)
for config_idx in [78, 79, 80]:
    if config_idx < len(selected_combinations):
        n_est, max_d, min_split = selected_combinations[config_idx]
        
        config = {
            'n_estimators': n_est,
            'max_depth': max_d,
            'min_samples_split': min_split,
            'random_state': 42
        }
        
        # 添加额外参数
        i = config_idx
        if i % 8 == 0:
            config['min_samples_leaf'] = min_samples_leaf_list[i % len(min_samples_leaf_list)]
        if i % 12 == 0:
            config['max_features'] = max_features_list[i % len(max_features_list)]
        if i % 20 == 0:
            config['bootstrap'] = False
        else:
            config['bootstrap'] = True
            
        print(f"\n配置 {config_idx + 1} (索引{config_idx}):")
        for key, value in config.items():
            print(f"  {key}: {value}")
            
        # 分析关键变化
        print(f"  特殊参数:")
        print(f"    i % 8 = {i % 8} ({'添加min_samples_leaf' if i % 8 == 0 else '无额外leaf参数'})")
        print(f"    i % 12 = {i % 12} ({'添加max_features' if i % 12 == 0 else '无额外features参数'})")
        print(f"    i % 20 = {i % 20} ({'bootstrap=False' if i % 20 == 0 else 'bootstrap=True'})")

# 分析参数组合的规律
print(f"\n参数组合规律分析:")
print(f"n_estimators: {n_estimators_list[:10]}")
print(f"max_depth: {max_depth_list[:8]}")
print(f"min_samples_split: {min_samples_split_list[:5]}")

# 计算配置79的位置
config_79_idx = 78
total_combinations = 10 * 8 * 5  # 400个组合
n_est_cycle = 8 * 5  # 40个配置一个n_estimators循环
depth_cycle = 5      # 5个配置一个max_depth循环

n_est_pos = config_79_idx // n_est_cycle
depth_pos = (config_79_idx % n_est_cycle) // depth_cycle
split_pos = config_79_idx % depth_cycle

print(f"\n配置79 (索引78) 在组合中的位置:")
print(f"  n_estimators位置: {n_est_pos} -> {n_estimators_list[n_est_pos]}")
print(f"  max_depth位置: {depth_pos} -> {max_depth_list[depth_pos]}")
print(f"  min_samples_split位置: {split_pos} -> {min_samples_split_list[split_pos]}")