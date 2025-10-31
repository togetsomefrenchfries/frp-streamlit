#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP超参数优化快速演示
Quick Hyperparameter Optimization Demo for FRP Models

这是一个简化的超参数优化演示脚本，用于快速测试和验证超参数优化功能。
点击运行按钮(▶️)即可执行！
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import time

def create_demo_frp_data(n_samples=500):
    """创建FRP演示数据"""
    print("🎯 创建FRP演示数据...")
    
    np.random.seed(42)
    
    # 创建特征数据
    data = {
        'pH': np.random.uniform(5, 12, n_samples),
        'exposure_time': np.random.uniform(100, 2000, n_samples),
        'temperature': np.random.uniform(20, 80, n_samples),
        'fiber_content': np.random.uniform(0.1, 5.0, n_samples),
        'diameter': np.random.uniform(6, 16, n_samples),
        'load': np.random.uniform(0, 100, n_samples),
        'concrete': np.random.choice([0, 1], n_samples),
        'chloride': np.random.choice([0, 1], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 模拟真实的强度保留率关系
    ph_effect = (df['pH'] - 7) ** 2 * -0.02
    temp_effect = df['temperature'] * -0.003
    time_effect = np.log(df['exposure_time']) * -0.08
    fiber_effect = df['fiber_content'] * 0.05
    
    retention = (0.85 + ph_effect + temp_effect + time_effect + fiber_effect + 
                np.random.normal(0, 0.08, n_samples))
    
    # 限制在合理范围
    df['tensile_retention'] = np.clip(retention, 0.3, 1.0)
    
    print(f"✅ 数据创建完成: {df.shape}")
    print(f"   目标变量范围: {df['tensile_retention'].min():.3f} - {df['tensile_retention'].max():.3f}")
    
    return df

def run_hyperparameter_optimization():
    """运行超参数优化实验"""
    
    print("=" * 60)
    print("🚀 FRP钢筋耐久性预测 - 超参数优化演示")
    print("=" * 60)
    
    # 1. 准备数据
    df = create_demo_frp_data(500)
    
    # 2. 分离特征和目标
    X = df.drop('tensile_retention', axis=1)
    y = df['tensile_retention']
    
    # 3. 数据分割 (7:2:1)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.22, random_state=42)  # 0.22 * 0.9 ≈ 0.2
    
    print(f"\n📊 数据分割:")
    print(f"   训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"   测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # 4. 定义超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    param_random = {
        'n_estimators': [10, 50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }
    
    results = []
    
    # 5. 网格搜索
    print(f"\n🔍 开始网格搜索超参数优化...")
    start_time = time.time()
    
    rf_grid = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf_grid, 
        param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 在验证集上评估
    grid_pred = grid_search.predict(X_val)
    grid_r2 = r2_score(y_val, grid_pred)
    grid_rmse = np.sqrt(mean_squared_error(y_val, grid_pred))
    grid_time = time.time() - start_time
    
    results.append({
        'method': '网格搜索',
        'best_params': grid_search.best_params_,
        'val_r2': grid_r2,
        'val_rmse': grid_rmse,
        'time': grid_time
    })
    
    print(f"✅ 网格搜索完成!")
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   验证集R²: {grid_r2:.4f}")
    print(f"   验证集RMSE: {grid_rmse:.4f}")
    print(f"   耗时: {grid_time:.1f}秒")
    
    # 6. 随机搜索
    print(f"\n🎲 开始随机搜索超参数优化...")
    start_time = time.time()
    
    rf_random = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        rf_random, 
        param_random, 
        n_iter=30,
        cv=3, 
        scoring='r2', 
        n_jobs=-1, 
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    # 在验证集上评估
    random_pred = random_search.predict(X_val)
    random_r2 = r2_score(y_val, random_pred)
    random_rmse = np.sqrt(mean_squared_error(y_val, random_pred))
    random_time = time.time() - start_time
    
    results.append({
        'method': '随机搜索',
        'best_params': random_search.best_params_,
        'val_r2': random_r2,
        'val_rmse': random_rmse,
        'time': random_time
    })
    
    print(f"✅ 随机搜索完成!")
    print(f"   最佳参数: {random_search.best_params_}")
    print(f"   验证集R²: {random_r2:.4f}")
    print(f"   验证集RMSE: {random_rmse:.4f}")
    print(f"   耗时: {random_time:.1f}秒")
    
    # 7. 选择最佳模型并在测试集上评估
    if grid_r2 > random_r2:
        best_model = grid_search.best_estimator_
        best_method = "网格搜索"
        best_params = grid_search.best_params_
    else:
        best_model = random_search.best_estimator_
        best_method = "随机搜索"
        best_params = random_search.best_params_
    
    # 测试集评估
    test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # 8. 打印最终结果
    print(f"\n" + "=" * 60)
    print(f"🏆 超参数优化结果汇总")
    print(f"=" * 60)
    
    for result in results:
        print(f"\n📋 {result['method']}:")
        print(f"   验证集R²: {result['val_r2']:.4f}")
        print(f"   验证集RMSE: {result['val_rmse']:.4f}")
        print(f"   耗时: {result['time']:.1f}秒")
        print(f"   最佳参数: {result['best_params']}")
    
    print(f"\n🎯 最终测试结果 (使用{best_method}的最佳模型):")
    print(f"   测试集R²: {test_r2:.4f}")
    print(f"   测试集RMSE: {test_rmse:.4f}")
    print(f"   最佳参数: {best_params}")
    
    # 9. 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 特征重要性排序:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n✅ 超参数优化演示完成!")
    print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, best_model

def main():
    """主函数 - 点击运行按钮执行此函数"""
    try:
        print("🚀 启动FRP超参数优化演示...")
        results, best_model = run_hyperparameter_optimization()
        print(f"\n🎉 演示成功完成!")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()