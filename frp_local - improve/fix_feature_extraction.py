#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复特征提取 - 适配database 4.xlsx的实际结构
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def analyze_database_structure():
    """分析database 4.xlsx的实际结构"""
    print("分析database 4.xlsx的实际结构...")
    
    df = pd.read_excel("E:/大学/intern/2025-summer-concret/database 4.xlsx")
    
    print(f"数据形状: {df.shape}")
    print(f"列总数: {len(df.columns)}")
    
    # 显示所有列名
    print("\n所有列名:")
    for i, col in enumerate(df.columns):
        print(f"  {i:3d}: {col}")
    
    # 寻找可能的目标变量和特征
    print("\n寻找可能的目标变量:")
    potential_targets = []
    for i, col in enumerate(df.columns):
        col_str = str(col).lower()
        if any(term in col_str for term in ['strength', 'modulus', 'retention', 'tg', 'value']):
            potential_targets.append((i, col))
            print(f"  候选目标 {i:3d}: {col}")
    
    # 检查数据内容
    print(f"\nComments列分布:")
    if 'Comments' in df.columns:
        print(df['Comments'].value_counts())
    
    return df

def create_6_feature_dataset(df):
    """使用6个基本列索引创建特征数据集"""
    print("创建6特征数据集（兼容模式）...")
    
    # 筛选Comments=1的数据
    if 'Comments' in df.columns:
        valid_data = df[df['Comments'] == 1].copy()
        print(f"筛选后有效数据: {len(valid_data)} 行")
    else:
        valid_data = df.copy()
        print(f"使用全部数据: {len(valid_data)} 行")
    
    if len(valid_data) == 0:
        print("错误：没有有效数据")
        return None, None
    
    # 使用前6列作为特征（跳过第0列Comments）
    feature_indices = [1, 2, 5, 6, 7, 12]  # 根据实际数据选择有意义的列
    
    # 检查这些列的数据质量
    print("\n检查特征列数据质量:")
    features = []
    feature_names = []
    
    for idx in feature_indices:
        if idx < len(df.columns):
            col_name = df.columns[idx]
            feature_names.append(f"feature_{idx}_{col_name[:20]}")  # 截断长列名
            
            # 尝试转换为数值
            feature_data = pd.to_numeric(valid_data.iloc[:, idx], errors='coerce')
            valid_count = feature_data.count()
            
            print(f"  列 {idx:2d} ({col_name[:30]}): 有效数值 {valid_count}/{len(valid_data)}")
            features.append(feature_data)
    
    # 组合特征
    X = pd.DataFrame(features).T
    X.columns = feature_names
    
    # 寻找合适的目标变量
    target_candidates = [12, 13, 15, 16, 17]  # Tg1, Tg2等可能的目标
    
    y = None
    target_name = None
    
    for target_idx in target_candidates:
        if target_idx < len(df.columns):
            col_name = df.columns[target_idx]
            target_data = pd.to_numeric(valid_data.iloc[:, target_idx], errors='coerce')
            valid_count = target_data.count()
            
            print(f"  目标候选 {target_idx:2d} ({col_name[:30]}): 有效数值 {valid_count}/{len(valid_data)}")
            
            if valid_count > len(valid_data) * 0.3:  # 至少30%有效数据
                y = target_data
                target_name = f"target_{target_idx}_{col_name[:20]}"
                print(f"  --> 选择为目标变量")
                break
    
    if y is None:
        print("警告：未找到合适的目标变量，使用列12作为默认目标")
        y = pd.to_numeric(valid_data.iloc[:, 12], errors='coerce')
        target_name = "target_default"
    
    # 移除缺失值
    combined = pd.concat([X, y], axis=1)
    combined.columns = list(X.columns) + [target_name]
    combined_clean = combined.dropna()
    
    print(f"\n最终数据集:")
    print(f"  原始数据: {len(valid_data)} 行")
    print(f"  清理后: {len(combined_clean)} 行")
    print(f"  特征数: {len(feature_names)}")
    print(f"  目标变量: {target_name}")
    
    if len(combined_clean) == 0:
        print("错误：清理后无有效数据")
        return None, None
    
    X_final = combined_clean[feature_names]
    y_final = combined_clean[target_name]
    
    return X_final, y_final

def run_simple_experiment():
    """运行简单的实验验证"""
    print("="*60)
    print("开始修复的特征提取实验")
    print("="*60)
    
    # 分析数据结构
    df = analyze_database_structure()
    
    # 创建特征数据集
    X, y = create_6_feature_dataset(df)
    
    if X is None or y is None:
        print("实验失败：无法创建有效的特征数据集")
        return
    
    # 运行简单的机器学习测试
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    print("\n运行机器学习测试...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n模型性能:")
    print(f"  训练集大小: {len(X_train)}")
    print(f"  测试集大小: {len(X_test)}")
    print(f"  R² 分数: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # 特征重要性
    print(f"\n特征重要性:")
    for i, (name, importance) in enumerate(zip(X.columns, model.feature_importances_)):
        print(f"  {name}: {importance:.4f}")
    
    print("\n✅ 修复的特征提取实验完成！")
    print("现在可以基于这个结构创建40参数实验")

if __name__ == "__main__":
    run_simple_experiment()