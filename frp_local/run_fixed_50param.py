#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版50参数实验 - 使用原始特征配置
Fixed 50-Parameter Experiment - Using Original Feature Configuration
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class FixedFiftyParameterExperiment:
    """修复版50参数实验"""
    
    def __init__(self):
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"fixed_50param_exp_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data_with_original_features(self):
        """使用原始特征配置加载数据"""
        print("📊 使用原始特征配置加载数据...")
        
        from data_loader import DataLoader
        data_loader = DataLoader()
        df_raw = data_loader.load_data()
        
        # 使用和0.7+实验相同的特征
        features = ['feature_name', 'Year', 'no.', 'time', 'Value1', 'diameter', 'temperature', 'No.']
        target = 'retention1'  # 使用原始的retention1
        
        print(f"选择的特征: {features}")
        print(f"目标变量: {target}")
        
        # 准备数据
        X = df_raw[features]
        y = df_raw[target]
        
        # 移除缺失值
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"📊 数据统计:")
        print(f"   原始数据: {X.shape}")
        print(f"   清理后: {X_clean.shape}")
        print(f"   目标变量范围: {y_clean.min():.4f} - {y_clean.max():.4f}")
        print(f"   数据保留率: {len(X_clean)/len(X)*100:.1f}%")
        
        return X_clean, y_clean
        
    def generate_parameter_grids(self):
        """生成50参数配置"""
        
        import itertools
        import random
        
        parameter_grids = {}
        
        # RandomForest参数网格
        if True:
            rf_base_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [4, 6, 8, 10, 12, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # 生成所有组合然后随机选择50个
            all_combinations = list(itertools.product(*rf_base_grid.values()))
            random.seed(42)
            selected_combinations = random.sample(all_combinations, min(50, len(all_combinations)))
            
            rf_configs = []
            for combo in selected_combinations:
                config = dict(zip(rf_base_grid.keys(), combo))
                config['random_state'] = 42
                rf_configs.append(config)
                
            parameter_grids['RandomForest'] = rf_configs
        
        # XGBoost参数网格
        if XGBOOST_AVAILABLE:
            xgb_base_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [3, 4, 5, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
            all_combinations = list(itertools.product(*xgb_base_grid.values()))
            random.seed(42)
            selected_combinations = random.sample(all_combinations, min(50, len(all_combinations)))
            
            xgb_configs = []
            for combo in selected_combinations:
                config = dict(zip(xgb_base_grid.keys(), combo))
                config['random_state'] = 42
                xgb_configs.append(config)
                
            parameter_grids['XGBoost'] = xgb_configs
        
        # LightGBM参数网格
        if LIGHTGBM_AVAILABLE:
            lgb_base_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [3, 4, 5, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'num_leaves': [15, 31, 50, 100, 150],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
            all_combinations = list(itertools.product(*lgb_base_grid.values()))
            random.seed(42)
            selected_combinations = random.sample(all_combinations, min(50, len(all_combinations)))
            
            lgb_configs = []
            for combo in selected_combinations:
                config = dict(zip(lgb_base_grid.keys(), combo))
                config['random_state'] = 42
                lgb_configs.append(config)
                
            parameter_grids['LightGBM'] = lgb_configs
        
        return parameter_grids
        
    def train_and_evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test, config_id):
        """训练和评估模型"""
        
        start_time = time.time()
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 测试集预测
        test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        total_time = time.time() - start_time
        
        result = {
            'model': model_name,
            'config_id': config_id,
            'config': model.get_params(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'total_time': total_time
        }
        
        return result
        
    def run_experiment(self):
        """运行完整实验"""
        
        print("🚀 开始修复版50参数实验")
        print("=" * 60)
        
        # 加载数据
        X, y = self.load_data_with_original_features()
        
        # 8:2分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 数据分割: 训练{X_train.shape[0]}, 测试{X_test.shape[0]}")
        
        # 生成参数网格
        parameter_grids = self.generate_parameter_grids()
        
        # 运行实验
        all_results = []
        total_configs = sum(len(configs) for configs in parameter_grids.values())
        current_config = 0
        
        for model_name, configs in parameter_grids.items():
            print(f"\\n🔧 {model_name} ({len(configs)}个配置)")
            model_results = []
            
            for i, config in enumerate(configs, 1):
                current_config += 1
                
                # 创建模型
                if model_name == 'RandomForest':
                    model = RandomForestRegressor(**config)
                elif model_name == 'XGBoost':
                    model = xgb.XGBRegressor(**config)
                elif model_name == 'LightGBM':
                    model = lgb.LGBMRegressor(**config)
                
                # 训练和评估
                result = self.train_and_evaluate_model(
                    model_name, model, X_train, y_train, X_test, y_test, i
                )
                
                model_results.append(result)
                all_results.append(result)
                
                print(f"  配置{i}: CV R²={result['cv_mean']:.4f}, 测试R²={result['test_r2']:.4f}")
                
                # 每5个配置保存一次
                if current_config % 5 == 0:
                    self.save_intermediate_results(all_results, current_config)
        
        # 保存最终结果
        self.save_final_results(all_results)
        
        # 生成报告
        self.generate_report(all_results)
        
        print(f"\\n🎉 实验完成！总配置: {total_configs}")
        print(f"结果保存在: {self.output_dir}")
        
    def save_intermediate_results(self, results, config_count):
        """保存中间结果"""
        batch_num = (config_count - 1) // 5 + 1
        
        # 保存JSON
        json_file = self.output_dir / f"results_batch_{batch_num:03d}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存CSV
        csv_file = self.output_dir / f"results_batch_{batch_num:03d}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"  ✅ 批次{batch_num}结果已保存")
        
    def save_final_results(self, results):
        """保存最终结果"""
        
        # 完整结果
        with open(self.output_dir / "complete_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "complete_results.csv", index=False, encoding='utf-8')
        
    def generate_report(self, results):
        """生成实验报告"""
        
        df = pd.DataFrame(results)
        
        report = []
        report.append("修复版50参数实验报告")
        report.append("=" * 50)
        report.append(f"实验时间: {datetime.now()}")
        report.append(f"总配置数: {len(results)}")
        report.append("")
        
        # 按模型统计
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            best_result = model_df.loc[model_df['cv_mean'].idxmax()]
            
            report.append(f"{model_name} 最佳结果:")
            report.append(f"  CV R²: {best_result['cv_mean']:.4f}±{best_result['cv_std']:.4f}")
            report.append(f"  测试R²: {best_result['test_r2']:.4f}")
            report.append(f"  配置: {best_result['config']}")
            report.append("")
        
        # 整体最佳
        best_overall = df.loc[df['cv_mean'].idxmax()]
        report.append("整体最佳配置:")
        report.append(f"  模型: {best_overall['model']}")
        report.append(f"  CV R²: {best_overall['cv_mean']:.4f}")
        report.append(f"  测试R²: {best_overall['test_r2']:.4f}")
        
        # 保存报告
        with open(self.output_dir / "final_report.txt", 'w', encoding='utf-8') as f:
            f.write("\\n".join(report))
        
        # 打印摘要
        print("\\n📈 实验摘要:")
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            best_cv = model_df['cv_mean'].max()
            best_test = model_df.loc[model_df['cv_mean'].idxmax()]['test_r2']
            print(f"  {model_name}: CV R²={best_cv:.4f}, 测试R²={best_test:.4f}")


if __name__ == "__main__":
    experiment = FixedFiftyParameterExperiment()
    experiment.run_experiment()