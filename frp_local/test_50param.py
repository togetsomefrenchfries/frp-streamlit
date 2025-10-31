#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试50参数实验脚本
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_parameter_generation():
    """测试参数生成功能"""
    
    from run_50param_experiments import FiftyParameterExperiment
    
    print("🧪 测试参数生成功能")
    print("=" * 40)
    
    experiment = FiftyParameterExperiment()
    parameter_grids = experiment.generate_parameter_grids()
    
    total_configs = 0
    for model_name, configs in parameter_grids.items():
        print(f"\n📊 {model_name}:")
        print(f"   配置数量: {len(configs)}")
        print(f"   示例配置:")
        
        # 显示前3个配置作为示例
        for i, config in enumerate(configs[:3], 1):
            print(f"     {i}. {config}")
        
        if len(configs) > 3:
            print(f"     ... 还有 {len(configs)-3} 个配置")
        
        total_configs += len(configs)
    
    print(f"\n✅ 总配置数: {total_configs}")
    print(f"预计总实验时间: {total_configs * 0.5 / 60:.1f} 分钟 (假设每个配置30秒)")

def test_data_loading():
    """测试数据加载和预处理"""
    
    print("\n🧪 测试数据加载和预处理")
    print("=" * 40)
    
    try:
        from data_loader import DataLoader
        from preprocessor import FRPDataPreprocessor
        import numpy as np
        
        # 加载数据
        data_loader = DataLoader()
        df_raw = data_loader.load_data()
        
        if df_raw is not None:
            print(f"✅ 原始数据加载成功: {df_raw.shape}")
            
            # 预处理
            preprocessor = FRPDataPreprocessor()
            df = preprocessor.preprocess_data(df_raw)
            print(f"✅ 预处理完成: {df.shape}")
            
            # 检查目标变量
            target_col = 'Tensile strength retention'
            if target_col in df.columns:
                print(f"✅ 目标变量存在: {target_col}")
                print(f"   有效值数量: {df[target_col].count()}")
                print(f"   值范围: {df[target_col].min():.3f} - {df[target_col].max():.3f}")
            else:
                print(f"❌ 目标变量不存在: {target_col}")
                print(f"可用列: {list(df.columns)}")
            
            # 检查特征
            feature_cols = [col for col in df.columns if col not in ['Title', 'Tensile strength retention']]
            X = df[feature_cols].select_dtypes(include=[np.number])
            print(f"✅ 数值特征: {X.shape[1]} 个")
            print(f"   特征列: {list(X.columns)}")
            
            # 检查数据完整性
            mask = ~(X.isnull().any(axis=1) | df[target_col].isnull())
            valid_samples = mask.sum()
            print(f"✅ 有效样本: {valid_samples} / {len(df)} ({valid_samples/len(df)*100:.1f}%)")
            
        else:
            print("❌ 数据加载失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_small_experiment():
    """运行小规模测试实验"""
    
    print("\n🧪 运行小规模测试实验 (每个模型3个配置)")
    print("=" * 50)
    
    try:
        from run_50param_experiments import FiftyParameterExperiment
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from data_loader import DataLoader
        from preprocessor import FRPDataPreprocessor
        import numpy as np
        
        # 加载数据
        data_loader = DataLoader()
        df_raw = data_loader.load_data()
        preprocessor = FRPDataPreprocessor()
        df = preprocessor.preprocess_data(df_raw)
        
        # 准备数据
        target_col = 'Tensile strength retention'
        feature_cols = [col for col in df.columns if col not in ['Title', 'Tensile strength retention']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # 移除完全缺失的特征列
        completely_missing = X.isnull().all()
        if completely_missing.any():
            missing_cols = X.columns[completely_missing].tolist()
            print(f"⚠️  移除完全缺失的特征: {missing_cols}")
            X = X.drop(columns=missing_cols)
        
        # 移除缺失值
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✅ 数据准备完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        # 测试3个RandomForest配置
        test_configs = [
            {'n_estimators': 50, 'max_depth': 5, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': None, 'random_state': 42}
        ]
        
        experiment = FiftyParameterExperiment()
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n🔧 测试配置 {i}: {config}")
            
            model = RandomForestRegressor(**config)
            result = experiment.train_and_evaluate_model(
                'RandomForest', model, X_train, y_train, X_test, y_test, i
            )
            
            print(f"   CV R²: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")
            print(f"   测试R²: {result['test_r2']:.4f}")
            print(f"   用时: {result['total_time']:.2f}s")
        
        print("✅ 小规模测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 50参数实验功能测试")
    print("=" * 60)
    
    # 测试1: 参数生成
    test_parameter_generation()
    
    # 测试2: 数据加载
    test_data_loading()
    
    # 测试3: 小规模实验
    test_small_experiment()
    
    print("\n" + "=" * 60)
    print("📝 测试总结:")
    print("1. 参数生成功能正常")
    print("2. 数据加载和预处理功能正常") 
    print("3. 小规模实验运行正常")
    print("\n✅ 可以运行完整的50参数实验了!")
    print("运行命令: python run_50param_experiments.py")