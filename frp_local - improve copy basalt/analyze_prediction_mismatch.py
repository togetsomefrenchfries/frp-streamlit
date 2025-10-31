#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def analyze_prediction_mismatch():
    """分析预测值和实际值的不匹配问题"""
    try:
        # 读取预测结果
        df = pd.read_csv('random_prediction_results_20250924_035143.csv')
        
        print("🔍 分析预测值与实际值的不匹配问题")
        print("=" * 60)
        
        # 基本统计
        actual = df['actual_retention']
        pred_rf = df['prediction_RandomForest']
        pred_xgb = df['prediction_XGBoost']
        pred_lgb = df['prediction_LightGBM']
        pred_mean = df['prediction_mean']
        
        print("📊 数据统计:")
        print(f"  样本数量: {len(df)}")
        print(f"\n实际retention值:")
        print(f"  范围: {actual.min():.6f} - {actual.max():.6f}")
        print(f"  均值: {actual.mean():.6f}")
        print(f"  中位数: {actual.median():.6f}")
        print(f"  标准差: {actual.std():.6f}")
        
        print(f"\n预测retention值 (RandomForest):")
        print(f"  范围: {pred_rf.min():.6f} - {pred_rf.max():.6f}")
        print(f"  均值: {pred_rf.mean():.6f}")
        print(f"  中位数: {pred_rf.median():.6f}")
        print(f"  标准差: {pred_rf.std():.6f}")
        
        # 计算R²
        r2_rf = r2_score(actual, pred_rf)
        r2_xgb = r2_score(actual, pred_xgb)
        r2_lgb = r2_score(actual, pred_lgb)
        r2_mean = r2_score(actual, pred_mean)
        
        print(f"\n📈 R²得分:")
        print(f"  RandomForest: {r2_rf:.6f}")
        print(f"  XGBoost: {r2_xgb:.6f}")
        print(f"  LightGBM: {r2_lgb:.6f}")
        print(f"  预测均值: {r2_mean:.6f}")
        
        # 分析相关性
        corr_rf = np.corrcoef(actual, pred_rf)[0,1]
        corr_xgb = np.corrcoef(actual, pred_xgb)[0,1]
        corr_lgb = np.corrcoef(actual, pred_lgb)[0,1]
        
        print(f"\n🔗 相关系数:")
        print(f"  RandomForest: {corr_rf:.6f}")
        print(f"  XGBoost: {corr_xgb:.6f}")
        print(f"  LightGBM: {corr_lgb:.6f}")
        
        # 检查数据分布
        print(f"\n📊 数据分布分析:")
        
        # 实际值分布
        actual_below_1 = (actual < 1).sum()
        actual_above_1 = (actual >= 1).sum()
        print(f"  实际值 < 1: {actual_below_1} ({actual_below_1/len(actual)*100:.1f}%)")
        print(f"  实际值 >= 1: {actual_above_1} ({actual_above_1/len(actual)*100:.1f}%)")
        
        # 预测值分布
        pred_below_1 = (pred_rf < 1).sum()
        pred_above_1 = (pred_rf >= 1).sum()
        print(f"  预测值 < 1: {pred_below_1} ({pred_below_1/len(pred_rf)*100:.1f}%)")
        print(f"  预测值 >= 1: {pred_above_1} ({pred_above_1/len(pred_rf)*100:.1f}%)")
        
        # 分析预测偏差
        bias = pred_rf - actual
        print(f"\n📉 预测偏差分析:")
        print(f"  平均偏差: {bias.mean():.6f}")
        print(f"  偏差标准差: {bias.std():.6f}")
        print(f"  偏差范围: {bias.min():.6f} - {bias.max():.6f}")
        
        # 检查是否存在系统性偏差
        print(f"\n🎯 系统性偏差检查:")
        positive_bias = (bias > 0).sum()
        negative_bias = (bias < 0).sum()
        print(f"  正偏差 (预测>实际): {positive_bias} ({positive_bias/len(bias)*100:.1f}%)")
        print(f"  负偏差 (预测<实际): {negative_bias} ({negative_bias/len(bias)*100:.1f}%)")
        
        # 显示一些具体例子
        print(f"\n📋 具体样本分析 (前10个):")
        for i in range(min(10, len(df))):
            print(f"  样本{i+1}: 实际={actual.iloc[i]:.3f}, 预测={pred_rf.iloc[i]:.3f}, 偏差={bias.iloc[i]:.3f}")
            
        # 检查训练数据的范围问题
        print(f"\n🤔 可能的问题分析:")
        print(f"  1. 数据尺度问题：实际值在0-1范围，预测值在1+范围")
        print(f"  2. 模型可能学习了错误的目标变量")
        print(f"  3. 特征工程或数据预处理存在问题")
        print(f"  4. 训练时的target变量定义可能有误")
        
        # 计算如果去掉系统偏差后的R²
        adjusted_pred = pred_rf - bias.mean()  # 去掉系统偏差
        r2_adjusted = r2_score(actual, adjusted_pred)
        print(f"  5. 去掉系统偏差后的R²: {r2_adjusted:.6f}")
        
        # 检查是否存在线性关系但有偏移
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(actual.values.reshape(-1, 1), pred_rf.values)
        linear_pred = lr.predict(actual.values.reshape(-1, 1))
        r2_linear = r2_score(pred_rf, linear_pred)
        
        print(f"  6. 线性拟合系数: slope={lr.coef_[0]:.6f}, intercept={lr.intercept_:.6f}")
        print(f"  7. 如果存在线性关系的R²: {r2_linear:.6f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_prediction_mismatch()