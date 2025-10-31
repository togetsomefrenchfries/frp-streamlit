#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP超参数优化实验脚本
Hyperparameter Optimization Experiment Runner for FRP Models

使用7:2:1数据分割策略进行超参数优化
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

class HyperparameterOptimizationExperiment:
    """参数实验管理器"""
    
    def __init__(self):
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"hyperopt_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def define_parameter_sets(self):
        """定义不同的参数配置集合"""
        
        parameter_sets = [
            # 1. 保守配置 - 小模型快速训练
            {
                'name': 'Conservative',
                'description': '保守参数配置，适合快速验证',
                'params': {
                    'xgboost': {
                        'n_estimators': 100,
                        'max_depth': 4,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42
                    },
                    'lightgbm': {
                        'n_estimators': 100,
                        'max_depth': 4,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'verbose': -1
                    },
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 4,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'training_params': {
                    'test_size': 0.2,
                    'cv_folds': 3,
                    'enable_tuning': False
                }
            },
            
            # 2. 标准配置 - 平衡性能和速度
            {
                'name': 'Standard',
                'description': '标准参数配置，平衡性能和训练时间',
                'params': {
                    'xgboost': {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42
                    },
                    'lightgbm': {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'verbose': -1
                    },
                    'random_forest': {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'training_params': {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'enable_tuning': False
                }
            },
            
            # 3. 高性能配置 - 追求最佳效果
            {
                'name': 'High_Performance',
                'description': '高性能参数配置，追求最佳模型效果',
                'params': {
                    'xgboost': {
                        'n_estimators': 500,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'subsample': 0.9,
                        'colsample_bytree': 0.9,
                        'random_state': 42
                    },
                    'lightgbm': {
                        'n_estimators': 500,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'subsample': 0.9,
                        'colsample_bytree': 0.9,
                        'random_state': 42,
                        'verbose': -1
                    },
                    'random_forest': {
                        'n_estimators': 500,
                        'max_depth': 8,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'training_params': {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'enable_tuning': False
                }
            },
            
            # 4. 深度学习风格配置 - 复杂模型
            {
                'name': 'Deep_Learning_Style',
                'description': '深度学习风格配置，复杂模型结构',
                'params': {
                    'xgboost': {
                        'n_estimators': 1000,
                        'max_depth': 10,
                        'learning_rate': 0.01,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1,
                        'random_state': 42
                    },
                    'lightgbm': {
                        'n_estimators': 1000,
                        'max_depth': 10,
                        'learning_rate': 0.01,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1,
                        'random_state': 42,
                        'verbose': -1
                    },
                    'random_forest': {
                        'n_estimators': 1000,
                        'max_depth': 10,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'training_params': {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'enable_tuning': False
                }
            },
            
            # 5. 正则化配置 - 防止过拟合
            {
                'name': 'Regularized',
                'description': '正则化配置，防止过拟合',
                'params': {
                    'xgboost': {
                        'n_estimators': 300,
                        'max_depth': 5,
                        'learning_rate': 0.08,
                        'subsample': 0.7,
                        'colsample_bytree': 0.7,
                        'reg_alpha': 0.5,
                        'reg_lambda': 0.5,
                        'random_state': 42
                    },
                    'lightgbm': {
                        'n_estimators': 300,
                        'max_depth': 5,
                        'learning_rate': 0.08,
                        'subsample': 0.7,
                        'colsample_bytree': 0.7,
                        'reg_alpha': 0.5,
                        'reg_lambda': 0.5,
                        'random_state': 42,
                        'verbose': -1
                    },
                    'random_forest': {
                        'n_estimators': 300,
                        'max_depth': 5,
                        'min_samples_split': 10,
                        'min_samples_leaf': 5,
                        'max_features': 'sqrt',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'training_params': {
                    'test_size': 0.25,
                    'cv_folds': 5,
                    'enable_tuning': False
                }
            },
            
            # 6. 超参数优化配置 - 自动调参
            {
                'name': 'Auto_Tuning',
                'description': '启用超参数自动优化',
                'params': {
                    'xgboost': {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': 42
                    },
                    'lightgbm': {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': 42,
                        'verbose': -1
                    },
                    'random_forest': {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'training_params': {
                    'test_size': 0.2,
                    'cv_folds': 3,  # 减少CV折数以加快超参数优化
                    'enable_tuning': True
                }
            }
        ]
        
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
            
            # 训练所有模型
            training_results = trainer.train_all_models(
                data_df, 
                target_column=config.TARGET_VARIABLE
            )
            
            # 获取性能对比
            comparison_results = trainer.compare_models()
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 保存实验结果
            experiment_result = {
                'experiment_name': param_set['name'],
                'description': param_set['description'],
                'parameters': param_set['params'],
                'training_params': param_set['training_params'],
                'training_time_seconds': training_time,
                'model_results': training_results,
                'comparison_results': comparison_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # 恢复原始配置
            config.MODEL_PARAMS = original_model_params
            config.TEST_SIZE = original_test_size
            config.CV_FOLDS = original_cv_folds
            
            return experiment_result
            
        except Exception as e:
            print(f"实验 {param_set['name']} 失败: {e}")
            return {
                'experiment_name': param_set['name'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_experiment_results(self, all_results):
        """保存实验结果"""
        
        # 保存完整结果到JSON
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 创建结果汇总表
        summary_data = []
        
        for result in all_results:
            if 'error' not in result:
                exp_name = result['experiment_name']
                training_time = result['training_time_seconds']
                
                # 获取每个模型的最佳性能
                for model_name, model_result in result['model_results'].items():
                    if 'test_metrics' in model_result:
                        metrics = model_result['test_metrics']
                        summary_data.append({
                            'Experiment': exp_name,
                            'Model': model_name,
                            'R2_Score': metrics.get('r2', 0),
                            'RMSE': metrics.get('rmse', float('inf')),
                            'MAE': metrics.get('mae', float('inf')),
                            'Training_Time_Sec': training_time,
                            'Hyperparameter_Tuning': result['training_params']['enable_tuning']
                        })
        
        # 保存汇总表
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('R2_Score', ascending=False)
            
            summary_file = self.output_dir / "experiment_summary.csv"
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            
            print(f"\n📊 实验结果汇总:")
            print(summary_df.to_string(index=False))
            
            # 找出最佳配置
            best_result = summary_df.iloc[0]
            print(f"\n🏆 最佳配置:")
            print(f"   实验名称: {best_result['Experiment']}")
            print(f"   模型类型: {best_result['Model']}")
            print(f"   R2得分: {best_result['R2_Score']:.4f}")
            print(f"   RMSE: {best_result['RMSE']:.4f}")
            print(f"   训练时间: {best_result['Training_Time_Sec']:.1f}秒")
            
        return self.output_dir
    
    def run_all_experiments(self):
        """运行所有参数实验"""
        
        print("🚀 开始FRP模型参数实验")
        print(f"实验ID: {self.experiment_id}")
        print(f"结果保存目录: {self.output_dir}")
        
        # 加载数据
        print("\n📊 加载数据...")
        loader = DataLoader("csv")
        df = loader.load_data()
        
        if df is None:
            print("数据加载失败！")
            return
        
        # 数据预处理
        print("🔧 数据预处理...")
        preprocessor = FRPDataPreprocessor()
        processed_df = preprocessor.create_model_dataset(df)
        
        if processed_df is None:
            print("数据预处理失败！")
            return
        
        print(f"预处理后数据形状: {processed_df.shape}")
        
        # 获取参数配置集合
        parameter_sets = self.define_parameter_sets()
        print(f"总共 {len(parameter_sets)} 个实验配置")
        
        # 运行所有实验
        all_results = []
        total_start_time = time.time()
        
        for i, param_set in enumerate(parameter_sets, 1):
            print(f"\n进度: {i}/{len(parameter_sets)}")
            result = self.run_single_experiment(param_set, processed_df)
            all_results.append(result)
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 所有实验完成！总用时: {total_time:.1f}秒")
        
        # 保存结果
        output_dir = self.save_experiment_results(all_results)
        print(f"\n📁 结果已保存到: {output_dir}")
        
        return all_results, output_dir

def main():
    """主函数"""
    
    print("FRP钢筋耐久性预测 - 超参数优化实验")
    print("="*50)
    
    # 创建实验管理器
    experiment = HyperparameterOptimizationExperiment()
    
    # 运行所有实验
    results, output_dir = experiment.run_all_experiments()
    
    print(f"\n✅ 超参数优化实验完成！")
    print(f"详细结果请查看: {output_dir}")
    print(f"- hyperopt_results.json: 完整实验数据")
    print(f"- hyperopt_summary.csv: 结果汇总表")

if __name__ == "__main__":
    main()