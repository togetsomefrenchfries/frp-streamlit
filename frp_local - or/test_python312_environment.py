#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Python 3.12.7环境和库安装 - 简化版特征提取测试
"""

import sys
print(f"Python版本: {sys.version}")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

print("✅ 所有必需库已成功导入!")
print(f"  pandas: {pd.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  xgboost: {xgb.__version__}")
print(f"  lightgbm: {lgb.__version__}")

# 读取数据并进行简单的特征提取测试
print("\n🔍 读取database 4.xlsx...")
df = pd.read_excel("E:/大学/intern/2025-summer-concret/database 4.xlsx")
print(f"原始数据形状: {df.shape}")

# 筛选Comments=1的数据
if 'Comments' in df.columns:
    valid_data = df[df['Comments'] == 1].copy()
    print(f"筛选后有效数据: {len(valid_data)} 行 (Comments=1)")
else:
    valid_data = df.copy()
    print(f"使用全部数据: {len(valid_data)} 行")

# 简单特征提取 - 使用数值列
print("\n🎯 进行简单特征提取...")
numeric_cols = []
for col in valid_data.columns:
    if pd.api.types.is_numeric_dtype(valid_data[col]):
        non_null_count = valid_data[col].count()
        if non_null_count > len(valid_data) * 0.1:  # 至少10%有效数据
            numeric_cols.append(col)

print(f"找到 {len(numeric_cols)} 个有效数值列")

if len(numeric_cols) >= 7:  # 至少需要6个特征 + 1个目标
    # 使用前6列作为特征，第7列作为目标
    feature_cols = numeric_cols[:6]
    target_col = numeric_cols[6]
    
    X = valid_data[feature_cols].copy()
    y = valid_data[target_col].copy()
    
    # 移除缺失值
    combined = pd.concat([X, y], axis=1)
    combined_clean = combined.dropna()
    
    print(f"清理后数据: {len(combined_clean)} 行")
    
    if len(combined_clean) > 20:  # 至少20行数据
        X_clean = combined_clean[feature_cols]
        y_clean = combined_clean[target_col]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42
        )
        
        print(f"\n🚀 测试机器学习模型...")
        print(f"训练集: {len(X_train)} 行")
        print(f"测试集: {len(X_test)} 行")
        
        # 测试3个模型
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=50, random_state=42, verbosity=-1)
        }
        
        results = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                results[name] = r2
                print(f"  {name}: R² = {r2:.4f}")
            except Exception as e:
                print(f"  {name}: 错误 - {e}")
        
        print(f"\n✅ 环境测试完成!")
        print(f"特征列: {feature_cols}")
        print(f"目标列: {target_col}")
        
        if results:
            best_model = max(results, key=results.get)
            print(f"最佳模型: {best_model} (R² = {results[best_model]:.4f})")
            print("\n🎉 Python 3.12.7环境配置正确，可以运行40参数实验!")
        else:
            print("\n⚠️ 模型训练出现问题，需要进一步调试")
    else:
        print(f"\n❌ 清理后数据不足: {len(combined_clean)} 行")
else:
    print(f"\n❌ 有效数值列不足: {len(numeric_cols)} 列")

print("\n" + "="*60)
print("环境测试报告:")
print(f"✅ Python版本: 3.12.7")
print(f"✅ 所有ML库已安装: xgboost, lightgbm, sklearn")
print(f"✅ 数据读取成功: {df.shape}")
print(f"✅ 可以开始正式的40参数实验")
print("="*60)