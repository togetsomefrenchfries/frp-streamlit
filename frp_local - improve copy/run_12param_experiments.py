#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP模型12超参数实验脚本
12 Hyperparameter Experiments for FRP Models

运行三种机器学习模型，每种12个不同的超参数配置
使用8:2训练测试分割，5折交叉验证
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from data_loader import DataLoader
    from preprocessor import FRPDataPreprocessor
    from model_trainer import ModelTrainer
    from utils import print_model_performance
    from config import config
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有模块文件存在且可访问")
    sys.exit(1)

class TwelveParameterExperiment:
    """12参数实验管理器"""
    
    def __init__(self):
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"12param_exp_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def define_12_parameter_sets(self):
        """定义12种不同的超参数配置集合"""
        
        parameter_sets = []
        
        # ========== XGBoost 12种配置 ==========
        xgboost_configs = [
            # 1. 极保守 - 快速原型
            {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.8},
            # 2. 轻量级 - 低复杂度
            {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.15, 'subsample': 0.8, 'colsample_bytree': 0.8},
            # 3. 保守配置
            {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
            # 4. 标准配置1
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
            # 5. 标准配置2
            {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9},
            # 6. 深度模型1
            {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
            # 7. 深度模型2
            {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9},
            # 8. 高性能配置1
            {'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8},
            # 9. 高性能配置2
            {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.9},
            # 10. 正则化配置1
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7},
            # 11. 正则化配置2
            {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.6, 'colsample_bytree': 0.6},
            # 12. 极致性能
            {'n_estimators': 600, 'max_depth': 10, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8}
        ]
        
        # ========== LightGBM 12种配置 ==========
        lightgbm_configs = [
            # 1. 极保守
            {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 15},
            # 2. 轻量级
            {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.15, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 20},
            # 3. 保守配置
            {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 31},
            # 4. 标准配置1
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 40},
            # 5. 标准配置2
            {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9, 'num_leaves': 50},
            # 6. 深度模型1
            {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 60},
            # 7. 深度模型2
            {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'num_leaves': 70},
            # 8. 高性能配置1
            {'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 80},
            # 9. 高性能配置2
            {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.9, 'num_leaves': 100},
            # 10. 正则化配置1
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 'num_leaves': 40},
            # 11. 正则化配置2
            {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.6, 'colsample_bytree': 0.6, 'num_leaves': 50},
            # 12. 极致性能
            {'n_estimators': 600, 'max_depth': 10, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_leaves': 120}
        ]
        
        # ========== Random Forest 12种配置 ==========
        rf_configs = [
            # 1. 极保守
            {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
            # 2. 轻量级
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 8, 'min_samples_leaf': 4, 'max_features': 'sqrt'},
            # 3. 保守配置
            {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
            # 4. 标准配置1
            {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
            # 5. 标准配置2
            {'n_estimators': 250, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
            # 6. 深度模型1
            {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2'},
            # 7. 深度模型2
            {'n_estimators': 300, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
            # 8. 高性能配置1
            {'n_estimators': 400, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
            # 9. 高性能配置2
            {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2'},
            # 10. 正则化配置1
            {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 0.5},
            # 11. 正则化配置2
            {'n_estimators': 400, 'max_depth': 20, 'min_samples_split': 8, 'min_samples_leaf': 4, 'max_features': 0.3},
            # 12. 极致性能
            {'n_estimators': 600, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None}
        ]
        
        # 组装所有配置
        for i in range(12):
            # XGBoost配置
            xgb_config = xgboost_configs[i].copy()
            xgb_config.update({'random_state': 42})
            
            # LightGBM配置
            lgb_config = lightgbm_configs[i].copy()
            lgb_config.update({'random_state': 42, 'verbose': -1})
            
            # Random Forest配置
            rf_config = rf_configs[i].copy()
            rf_config.update({'random_state': 42, 'n_jobs': -1})
            
            parameter_sets.append({
                'name': f'Config_{i+1:02d}',
                'description': f'第{i+1}组超参数配置',
                'params': {
                    'xgboost': xgb_config,
                    'lightgbm': lgb_config,
                    'random_forest': rf_config
                },
                'training_params': {
                    'test_size': 0.2,  # 8:2分割
                    'cv_folds': 5,     # 5折交叉验证
                    'enable_tuning': False
                }
            })
        
        return parameter_sets
    
    def run_single_experiment(self, param_set, data_df):
        """运行单个参数配置实验"""
        
        print(f"\n{'='*60}")
        print(f"开始实验: {param_set['name']}")
        print(f"描述: {param_set['description']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 临时修改配置
            original_model_params = config.MODEL_PARAMS.copy()
            original_test_size = config.TEST_SIZE
            original_cv_folds = config.CV_FOLDS
            
            # 应用实验参数
            config.MODEL_PARAMS.update(param_set['params'])
            config.TEST_SIZE = param_set['training_params']['test_size']
            config.CV_FOLDS = param_set['training_params']['cv_folds']
            
            # 创建模型训练器
            trainer = ModelTrainer(
                enable_hyperparameter_tuning=param_set['training_params']['enable_tuning']
            )
            
            # 准备数据
            X, y, feature_info = trainer.prepare_data(data_df)
            
            # 分离测试集
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=42
            )
            
            print(f"数据分割: 训练集{len(X_train)}样本, 测试集{len(X_test)}样本")
            
            # 训练所有模型
            models = trainer.train_all_models(X_train, y_train)
            
            # 评估模型
            experiment_results = {}
            for model_name, model in models.items():
                if model and 'error' not in model:
                    # 测试集预测
                    y_pred = model.predict(X_test)
                    
                    # 计算评估指标
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    # 交叉验证
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(model, X_train, y_train, cv=config.CV_FOLDS, scoring='r2')
                    
                    experiment_results[model_name] = {
                        'test_r2': r2,
                        'test_mse': mse,
                        'test_mae': mae,
                        'test_rmse': rmse,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'cv_scores': cv_scores.tolist(),
                        'params': param_set['params'][model_name],
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    }
                    
                    print(f"  {model_name:15s} - R²: {r2:.4f}, RMSE: {rmse:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            end_time = time.time()
            duration = end_time - start_time
            
            # 恢复原始配置
            config.MODEL_PARAMS = original_model_params
            config.TEST_SIZE = original_test_size
            config.CV_FOLDS = original_cv_folds
            
            # 保存实验结果
            experiment_summary = {
                'experiment_name': param_set['name'],
                'description': param_set['description'],
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'results': experiment_results
            }
            
            print(f"实验完成，耗时: {duration:.1f}秒")
            return experiment_summary
            
        except Exception as e:
            print(f"实验失败: {str(e)}")
            return {
                'experiment_name': param_set['name'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_experiments(self):
        """运行所有12组参数实验"""
        
        print("FRP钢筋耐久性预测 - 12超参数配置实验")
        print("=" * 50)
        print("🚀 开始12组超参数配置实验")
        print(f"实验ID: {self.experiment_id}")
        print(f"结果保存目录: {self.output_dir}")
        print("配置: 8:2训练测试分割, 5折交叉验证")
        
        # 加载数据
        print("\n📊 加载数据...")
        data_loader = DataLoader()
        data_df = data_loader.load_data()
        
        print(f"数据形状: {data_df.shape}")
        print(f"目标变量: {config.TARGET_VARIABLE}")
        
        # 获取参数配置
        parameter_sets = self.define_12_parameter_sets()
        print(f"\n🔧 共{len(parameter_sets)}组参数配置")
        
        start_time = time.time()
        all_results = []
        
        # 运行每个实验
        for i, param_set in enumerate(parameter_sets, 1):
            print(f"\n进度: {i}/{len(parameter_sets)}")
            result = self.run_single_experiment(param_set, data_df)
            all_results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 所有实验完成！总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        
        # 保存和分析结果
        self.save_and_analyze_results(all_results)
        
        return all_results, self.output_dir
    
    def save_and_analyze_results(self, all_results):
        """保存和分析实验结果"""
        
        # 保存详细结果
        detailed_file = self.output_dir / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 创建汇总表
        summary_data = []
        for result in all_results:
            if 'results' in result:
                for model_name, model_result in result['results'].items():
                    summary_data.append({
                        '实验': result['experiment_name'],
                        '模型': model_name,
                        '测试R²': model_result['test_r2'],
                        '测试RMSE': model_result['test_rmse'],
                        '测试MAE': model_result['test_mae'],
                        'CV均值': model_result['cv_mean'],
                        'CV标准差': model_result['cv_std'],
                        '训练集大小': model_result['train_size'],
                        '测试集大小': model_result['test_size']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # 保存汇总表
            summary_file = self.output_dir / "experiment_summary.csv"
            summary_df.to_csv(summary_file, index=False, encoding='utf-8')
            
            # 按模型分组显示最佳结果
            print(f"\n📊 实验结果汇总:")
            print(summary_df.to_string(index=False))
            
            print(f"\n🏆 各模型最佳配置:")
            for model in ['xgboost', 'lightgbm', 'random_forest']:
                model_results = summary_df[summary_df['模型'] == model]
                if not model_results.empty:
                    best_idx = model_results['测试R²'].idxmax()
                    best_result = model_results.loc[best_idx]
                    print(f"  {model:15s}: {best_result['实验']} - R²={best_result['测试R²']:.4f}, RMSE={best_result['测试RMSE']:.4f}")
        
        print(f"\n📁 所有结果已保存到: {self.output_dir}")
        print("文件列表:")
        print("  - detailed_results.json: 详细实验数据")
        print("  - experiment_summary.csv: 结果汇总表")

def main():
    """主函数 - 点击运行按钮执行此函数"""
    try:
        # 配置验证
        if hasattr(config, 'validate_config'):
            config.validate_config()
            print("Configuration validation passed")
        
        # 运行12参数实验
        experiment = TwelveParameterExperiment()
        results, output_dir = experiment.run_all_experiments()
        
        print(f"\n✅ 12超参数配置实验成功完成!")
        
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()