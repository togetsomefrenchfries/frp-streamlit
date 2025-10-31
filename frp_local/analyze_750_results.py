#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
750参数实验结果分析脚本
"""
import pandas as pd
import os

def analyze_results(results_dir):
    """分析实验结果"""
    csv_path = os.path.join(results_dir, 'complete_results.csv')
    
    if not os.path.exists(csv_path):
        print(f"结果文件不存在: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("🎯 750参数超参数优化实验结果分析")
    print("=" * 50)
    print(f"总配置数: {len(df)}")
    print(f"数据筛选条件: Comments=1 且 BU列=SMD")
    
    # 各模型结果统计
    print("\n📊 各模型结果统计:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  配置数量: {len(model_df)}")
        print(f"  最佳R²: {model_df['test_r2'].max():.6f}")
        print(f"  平均R²: {model_df['test_r2'].mean():.6f}")
        print(f"  R²标准差: {model_df['test_r2'].std():.6f}")
        print(f"  R²>0.5的配置: {len(model_df[model_df['test_r2'] > 0.5])}")
    
    # 全局最佳结果
    print("\n🏆 全局最佳配置TOP10:")
    top10 = df.nlargest(10, 'test_r2')
    for i, row in top10.iterrows():
        print(f"{row.name+1:2d}. {row['model']} #{row['config_id']:3d}: R²={row['test_r2']:.6f}, CV={row['cv_mean']:.6f}")
    
    # 各模型最佳配置
    print("\n🎖️ 各模型最佳配置:")
    for model in df['model'].unique():
        model_best = df[df['model'] == model].nlargest(1, 'test_r2').iloc[0]
        print(f"\n{model}:")
        print(f"  配置ID: {model_best['config_id']}")
        print(f"  测试R²: {model_best['test_r2']:.6f}")
        print(f"  CV均值: {model_best['cv_mean']:.6f}")
        print(f"  CV标准差: {model_best['cv_std']:.6f}")
        print(f"  参数配置: {model_best['config']}")
    
    # 性能分布
    print("\n📈 性能分布:")
    print(f"R² > 0.6: {len(df[df['test_r2'] > 0.6])} 个配置")
    print(f"R² > 0.5: {len(df[df['test_r2'] > 0.5])} 个配置")
    print(f"R² > 0.3: {len(df[df['test_r2'] > 0.3])} 个配置")
    print(f"R² > 0.1: {len(df[df['test_r2'] > 0.1])} 个配置")
    print(f"R² < 0: {len(df[df['test_r2'] < 0])} 个配置")
    
    return df

if __name__ == "__main__":
    results_dir = "experiments/valid_750param_exp_20250921_044234"
    df = analyze_results(results_dir)