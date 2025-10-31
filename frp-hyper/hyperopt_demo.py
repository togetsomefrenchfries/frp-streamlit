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
    """超参数优化实验管理器"""
    
    def __init__(self):
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"hyperopt_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_demo_data(self, n_samples=1000):
        """创建演示数据用于超参数优化测试"""
        print("🎯 创建演示数据进行超参数优化测试...")
        
        np.random.seed(42)
        
        # 创建特征数据
        data = {
            'Title': range(n_samples),
            'pH of condition environment': np.random.uniform(5, 12, n_samples),
            'Exposure time': np.random.uniform(100, 2000, n_samples),
            'Fibre content': np.random.uniform(0.1, 5.0, n_samples),
            'Exposure temperature': np.random.uniform(20, 80, n_samples),
            'Diameter': np.random.uniform(6, 16, n_samples),
            'Presence of concrete': np.random.choice([0, 1], n_samples),
            'Load': np.random.uniform(0, 100, n_samples),
            'Presence of chloride ion': np.random.choice([0, 1], n_samples),
            'Fibre type': np.random.choice([0, 1], n_samples),
            'Matrix type': np.random.choice([0, 1], n_samples),
            'Surface treatment': np.random.choice([0, 1], n_samples),
            'Strength of unconditioned rebar': np.random.uniform(500, 1500, n_samples)
        }
        
        # 创建目标变量 - 模拟真实关系
        df = pd.DataFrame(data)
        
        # 模拟复杂的非线性关系
        ph_effect = (df['pH of condition environment'] - 7) ** 2 * -0.01
        temp_effect = df['Exposure temperature'] * -0.005
        time_effect = np.log(df['Exposure time']) * -0.05
        fiber_effect = df['Fibre content'] * 0.02
        
        base_retention = 0.8
        retention = (base_retention + ph_effect + temp_effect + time_effect + fiber_effect + 
                    np.random.normal(0, 0.1, n_samples))
        
        # 确保在合理范围内
        retention = np.clip(retention, 0.2, 1.0)
        
        df['Tensile strength retention'] = retention
        
        print(f"✅ 演示数据创建完成: {df.shape}")
        print(f"   目标变量范围: {retention.min():.3f} - {retention.max():.3f}")
        print(f"   目标变量均值: {retention.mean():.3f}")
        
        return df
        
    def define_hyperopt_strategies(self):
        """定义超参数优化策略"""
        
        strategies = [
            {
                'name': 'RandomForest_GridSearch',
                'description': '随机森林网格搜索',
                'search_method': 'grid',
                'model_focus': 'random_forest'
            },
            {
                'name': 'RandomForest_RandomSearch', 
                'description': '随机森林随机搜索',
                'search_method': 'random',
                'model_focus': 'random_forest'
            }
        ]
        
        # 如果有XGBoost，添加XGBoost策略
        try:
            import xgboost
            strategies.extend([
                {
                    'name': 'XGBoost_GridSearch',
                    'description': 'XGBoost网格搜索',
                    'search_method': 'grid',
                    'model_focus': 'xgboost'
                },
                {
                    'name': 'XGBoost_RandomSearch',
                    'description': 'XGBoost随机搜索',
                    'search_method': 'random',
                    'model_focus': 'xgboost'
                }
            ])
        except ImportError:
            print("⚠️ XGBoost不可用，跳过XGBoost超参数优化")
            
        # 如果有LightGBM，添加LightGBM策略
        try:
            import lightgbm
            strategies.extend([
                {
                    'name': 'LightGBM_GridSearch',
                    'description': 'LightGBM网格搜索',
                    'search_method': 'grid',
                    'model_focus': 'lightgbm'
                },
                {
                    'name': 'LightGBM_RandomSearch',
                    'description': 'LightGBM随机搜索',
                    'search_method': 'random',
                    'model_focus': 'lightgbm'
                }
            ])
        except ImportError:
            print("⚠️ LightGBM不可用，跳过LightGBM超参数优化")
        
        return strategies
        
    def run_hyperparameter_optimization(self, df):
        """运行超参数优化实验"""
        
        strategies = self.define_hyperopt_strategies()
        all_results = []
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n进度: {i}/{len(strategies)}")
            print("=" * 60)
            print(f"开始实验: {strategy['name']}")
            print(f"描述: {strategy['description']}")
            print("=" * 60)
            
            start_time = time.time()
            
            try:
                # 临时修改配置
                original_method = config.HYPERPARAMETER_SEARCH_METHOD
                original_iter = config.TUNING_N_ITER
                
                config.HYPERPARAMETER_SEARCH_METHOD = strategy['search_method']
                if strategy['search_method'] == 'grid':
                    config.TUNING_N_ITER = 20  # 网格搜索用较少迭代
                else:
                    config.TUNING_N_ITER = 50  # 随机搜索用更多迭代
                
                # 初始化训练器（启用超参数优化）
                trainer = ModelTrainer(enable_hyperparameter_tuning=True)
                
                # 只训练特定模型
                target_model = strategy.get('model_focus', 'random_forest')
                if target_model in trainer.models:
                    # 训练单个模型
                    X, y, feature_info = trainer.prepare_data(df)
                    result = trainer.train_model_with_hyperopt(target_model, X, y)
                    results = {target_model: result}
                else:
                    print(f"⚠️ 模型 {target_model} 不可用")
                    continue
                
                # 恢复原始配置
                config.HYPERPARAMETER_SEARCH_METHOD = original_method
                config.TUNING_N_ITER = original_iter
                
                end_time = time.time()
                duration = end_time - start_time
                
                if results and target_model in results and 'error' not in results[target_model]:
                    # 添加实验信息
                    experiment_result = {
                        'strategy_name': strategy['name'],
                        'strategy_description': strategy['description'],
                        'search_method': strategy['search_method'],
                        'model_focus': target_model,
                        'duration_seconds': duration,
                        'model_result': results[target_model],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    all_results.append(experiment_result)
                    
                    # 打印结果
                    result = results[target_model]
                    if 'r2_score' in result:
                        print(f"✅ {target_model} 超参数优化完成!")
                        print(f"   R² 分数: {result['r2_score']:.4f}")
                        print(f"   RMSE: {result.get('rmse', 'N/A'):.4f}")
                        print(f"   训练集: {result.get('train_size', 'N/A')} 样本")
                        print(f"   验证集: {result.get('val_size', 'N/A')} 样本")
                        print(f"   测试集: {result.get('test_size', 'N/A')} 样本")
                        print(f"⏱️ 耗时: {duration:.1f}秒")
                else:
                    print(f"❌ 实验 {strategy['name']} 失败")
                
            except Exception as e:
                print(f"实验 {strategy['name']} 失败: {str(e)}")
                continue
        
        return all_results
    
    def save_experiment_results(self, all_results):
        """保存实验结果"""
        
        # 保存完整结果（转换为可序列化格式）
        serializable_results = []
        for result in all_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    serializable_result[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = value.item()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        results_file = self.output_dir / "hyperopt_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 创建汇总表
        summary_data = []
        for result in all_results:
            if 'model_result' in result:
                model_result = result['model_result']
                if isinstance(model_result, dict) and 'r2_score' in model_result:
                    summary_data.append({
                        '策略': result['strategy_name'],
                        '搜索方法': result['search_method'],
                        '模型': result.get('model_focus', 'unknown'),
                        'R²': model_result['r2_score'],
                        'RMSE': model_result.get('rmse', 'N/A'),
                        '训练集大小': model_result.get('train_size', 'N/A'),
                        '验证集大小': model_result.get('val_size', 'N/A'),
                        '测试集大小': model_result.get('test_size', 'N/A'),
                        '超参数优化': model_result.get('hyperparameter_optimized', False),
                        '耗时(秒)': round(result['duration_seconds'], 2)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.output_dir / "hyperopt_summary.csv"
            summary_df.to_csv(summary_file, index=False, encoding='utf-8')
            
            print(f"\n📊 超参数优化结果汇总:")
            print(summary_df.to_string(index=False))
            
            # 找到最佳结果
            best_idx = summary_df['R²'].idxmax()
            best_result = summary_df.loc[best_idx]
            print(f"\n🏆 最佳结果:")
            print(f"   策略: {best_result['策略']}")
            print(f"   模型: {best_result['模型']}")
            print(f"   R²: {best_result['R²']:.4f}")
            print(f"   RMSE: {best_result['RMSE']:.4f}")
        
        return self.output_dir
    
    def run_all_experiments(self):
        """运行所有超参数优化实验"""
        
        print("FRP钢筋耐久性预测 - 超参数优化实验")
        print("=" * 50)
        print("🚀 开始FRP超参数优化实验")
        print(f"实验ID: {self.experiment_id}")
        print(f"结果保存目录: {self.output_dir}")
        
        # 使用演示数据（因为原始数据目标变量为空）
        print("\n📊 准备数据...")
        df = self.create_demo_data(n_samples=1000)
        
        print(f"数据形状: {df.shape}")
        print(f"目标变量: {df['Tensile strength retention'].describe()}")
        
        start_time = time.time()
        
        # 运行超参数优化实验
        all_results = self.run_hyperparameter_optimization(df)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 所有实验完成！总用时: {total_time:.1f}秒")
        
        # 保存结果
        output_dir = self.save_experiment_results(all_results)
        
        print(f"\n📁 结果已保存到: {output_dir}")
        print(f"\n✅ 超参数优化实验完成！")
        print(f"详细结果请查看: {output_dir}")
        print("- hyperopt_results.json: 完整实验数据")
        print("- hyperopt_summary.csv: 结果汇总表")
        
        return all_results, output_dir

def main():
    """主函数"""
    try:
        # 配置验证
        if hasattr(config, 'validate_config'):
            config.validate_config()
            print("Configuration validation passed")
        
        # 运行超参数优化实验
        experiment = HyperparameterOptimizationExperiment()
        results, output_dir = experiment.run_all_experiments()
        
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()