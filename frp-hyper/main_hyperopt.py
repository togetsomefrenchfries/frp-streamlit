#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP超参数优化主程序
Main Hyperparameter Optimization Script for FRP Models

这是FRP钢筋耐久性预测的超参数优化主程序。
点击运行按钮(▶️)即可执行完整的超参数优化流程！

功能特点:
- 7:2:1数据分割 (训练:验证:测试)
- 网格搜索和随机搜索超参数优化
- 自动模型选择和评估
- 详细的结果报告和保存
- 支持多种机器学习模型
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

# 导入本地模块
try:
    from data_loader import DataLoader
    from preprocessor import FRPDataPreprocessor
    from model_trainer import ModelTrainer
    from utils import print_model_performance
    from config import config
    print("✅ 成功导入所有本地模块")
except ImportError as e:
    print(f"⚠️ 导入本地模块失败: {e}")
    print("使用内置的超参数优化功能")

# 机器学习库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class FRPHyperparameterOptimizer:
    """FRP超参数优化器"""
    
    def __init__(self):
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"frp_hyperopt_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def create_comprehensive_demo_data(self, n_samples=1000):
        """创建全面的FRP演示数据"""
        print("🎯 创建综合FRP数据集...")
        
        np.random.seed(42)
        
        # 创建更全面的特征数据
        data = {
            # 环境条件
            'pH_environment': np.random.uniform(4, 14, n_samples),
            'exposure_temperature': np.random.uniform(10, 90, n_samples),
            'exposure_time_hours': np.random.uniform(100, 8760, n_samples),  # 最多1年
            'humidity': np.random.uniform(30, 100, n_samples),
            'chloride_concentration': np.random.uniform(0, 5, n_samples),
            
            # 材料属性
            'fiber_content_percent': np.random.uniform(0.5, 8.0, n_samples),
            'fiber_diameter_mm': np.random.uniform(6, 20, n_samples),
            'matrix_strength_mpa': np.random.uniform(400, 1800, n_samples),
            'concrete_strength_mpa': np.random.uniform(20, 80, n_samples),
            
            # 几何和荷载
            'rebar_diameter_mm': np.random.uniform(8, 32, n_samples),
            'applied_load_percent': np.random.uniform(0, 80, n_samples),
            'cover_thickness_mm': np.random.uniform(20, 80, n_samples),
            
            # 分类特征
            'fiber_type': np.random.choice([0, 1, 2], n_samples),  # 0:玻璃纤维, 1:碳纤维, 2:芳纶纤维
            'matrix_type': np.random.choice([0, 1], n_samples),    # 0:环氧树脂, 1:乙烯基酯
            'surface_treatment': np.random.choice([0, 1], n_samples),  # 0:无, 1:有
            'loading_type': np.random.choice([0, 1, 2], n_samples),  # 0:静载, 1:疲劳, 2:冲击
        }
        
        df = pd.DataFrame(data)
        
        # 创建复杂的多因素目标变量模型
        # 基础强度保留率
        base_retention = 0.9
        
        # pH效应 (非线性，中性最好)
        ph_effect = -0.02 * (df['pH_environment'] - 8.5) ** 2
        
        # 温度效应 (高温有害)
        temp_effect = -0.004 * (df['exposure_temperature'] - 20)
        
        # 时间效应 (对数衰减)
        time_effect = -0.15 * np.log(df['exposure_time_hours'] / 100)
        
        # 氯离子效应 (线性有害)
        chloride_effect = -0.08 * df['chloride_concentration']
        
        # 纤维含量效应 (适量最佳)
        fiber_effect = 0.03 * df['fiber_content_percent'] - 0.002 * df['fiber_content_percent'] ** 2
        
        # 荷载效应 (高荷载有害)
        load_effect = -0.002 * df['applied_load_percent']
        
        # 纤维类型效应 (碳纤维>芳纶>玻璃)
        fiber_type_effect = np.where(df['fiber_type'] == 1, 0.05,  # 碳纤维
                                   np.where(df['fiber_type'] == 2, 0.02, 0))  # 芳纶纤维
        
        # 表面处理效应
        surface_effect = np.where(df['surface_treatment'] == 1, 0.03, 0)
        
        # 综合效应
        retention = (base_retention + ph_effect + temp_effect + time_effect + 
                    chloride_effect + fiber_effect + load_effect + 
                    fiber_type_effect + surface_effect + 
                    np.random.normal(0, 0.05, n_samples))
        
        # 确保在合理范围内
        df['tensile_strength_retention'] = np.clip(retention, 0.1, 1.0)
        
        print(f"✅ 综合数据集创建完成: {df.shape}")
        print(f"   目标变量范围: {df['tensile_strength_retention'].min():.3f} - {df['tensile_strength_retention'].max():.3f}")
        print(f"   目标变量均值: {df['tensile_strength_retention'].mean():.3f}")
        print(f"   目标变量标准差: {df['tensile_strength_retention'].std():.3f}")
        
        return df
    
    def split_data_7_2_1(self, X, y):
        """7:2:1数据分割"""
        # 首先分出10%作为测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=None
        )
        
        # 从剩余90%中分出训练集(77.8%)和验证集(22.2%)，使得最终比例为7:2:1
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.222, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def define_hyperparameter_grids(self):
        """定义超参数搜索空间"""
        
        # 随机森林参数空间
        rf_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf_random = {
            'n_estimators': [10, 50, 100, 200, 300, 500],
            'max_depth': [3, 5, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],
            'bootstrap': [True, False]
        }
        
        return {
            'random_forest': {
                'grid': rf_grid,
                'random': rf_random,
                'model': RandomForestRegressor(random_state=42)
            }
        }
    
    def evaluate_model(self, model, X_test, y_test):
        """模型评估"""
        y_pred = model.predict(X_test)
        
        return {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'predictions': y_pred.tolist(),
            'actuals': y_test.tolist()
        }
    
    def run_hyperparameter_search(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """运行超参数搜索"""
        
        results = {}
        
        # 网格搜索
        print(f"\n🔍 {model_name} - 网格搜索...")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['grid'],
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        grid_eval = self.evaluate_model(grid_search, X_val, y_val)
        grid_time = time.time() - start_time
        
        results['grid_search'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'evaluation': grid_eval,
            'time_seconds': grid_time,
            'model': grid_search.best_estimator_
        }
        
        print(f"   ✅ 网格搜索完成 - R²: {grid_eval['r2_score']:.4f}, 耗时: {grid_time:.1f}秒")
        
        # 随机搜索
        print(f"\n🎲 {model_name} - 随机搜索...")
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            model_config['model'],
            model_config['random'],
            n_iter=50,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        random_eval = self.evaluate_model(random_search, X_val, y_val)
        random_time = time.time() - start_time
        
        results['random_search'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'evaluation': random_eval,
            'time_seconds': random_time,
            'model': random_search.best_estimator_
        }
        
        print(f"   ✅ 随机搜索完成 - R²: {random_eval['r2_score']:.4f}, 耗时: {random_time:.1f}秒")
        
        return results
    
    def run_full_optimization(self):
        """运行完整的超参数优化流程"""
        
        print("=" * 70)
        print("🚀 FRP钢筋耐久性预测 - 完整超参数优化流程")
        print("=" * 70)
        print(f"实验ID: {self.experiment_id}")
        print(f"结果保存目录: {self.output_dir}")
        
        # 1. 数据准备
        print(f"\n📊 第1步: 数据准备")
        df = self.create_comprehensive_demo_data(n_samples=1500)
        
        # 分离特征和目标
        feature_cols = [col for col in df.columns if col != 'tensile_strength_retention']
        X = df[feature_cols]
        y = df['tensile_strength_retention']
        
        print(f"   特征数量: {X.shape[1]}")
        print(f"   样本数量: {X.shape[0]}")
        print(f"   特征列表: {list(X.columns)}")
        
        # 2. 数据分割
        print(f"\n🔄 第2步: 7:2:1数据分割")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_7_2_1(X, y)
        
        print(f"   训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"   测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # 3. 超参数搜索
        print(f"\n🔧 第3步: 超参数优化")
        hyperparameter_grids = self.define_hyperparameter_grids()
        
        all_results = {}
        
        for model_name, model_config in hyperparameter_grids.items():
            print(f"\n📈 优化模型: {model_name}")
            model_results = self.run_hyperparameter_search(
                model_name, model_config, X_train, y_train, X_val, y_val
            )
            all_results[model_name] = model_results
        
        # 4. 最佳模型选择和测试集评估
        print(f"\n🏆 第4步: 最佳模型选择和最终评估")
        
        best_overall_r2 = -np.inf
        best_model_info = None
        
        for model_name, model_results in all_results.items():
            for search_type, search_results in model_results.items():
                val_r2 = search_results['evaluation']['r2_score']
                if val_r2 > best_overall_r2:
                    best_overall_r2 = val_r2
                    best_model_info = {
                        'model_name': model_name,
                        'search_type': search_type,
                        'model': search_results['model'],
                        'params': search_results['best_params']
                    }
        
        # 在测试集上评估最佳模型
        if best_model_info:
            final_evaluation = self.evaluate_model(
                best_model_info['model'], X_test, y_test
            )
            
            print(f"\n🎯 最佳模型: {best_model_info['model_name']} ({best_model_info['search_type']})")
            print(f"   验证集R²: {best_overall_r2:.4f}")
            print(f"   测试集R²: {final_evaluation['r2_score']:.4f}")
            print(f"   测试集RMSE: {final_evaluation['rmse']:.4f}")
            print(f"   测试集MAE: {final_evaluation['mae']:.4f}")
            print(f"   最佳参数: {best_model_info['params']}")
        
        # 5. 结果汇总和保存
        print(f"\n📋 第5步: 结果汇总")
        
        summary_data = []
        for model_name, model_results in all_results.items():
            for search_type, search_results in model_results.items():
                eval_results = search_results['evaluation']
                summary_data.append({
                    '模型': model_name,
                    '搜索方法': search_type,
                    '验证集R²': eval_results['r2_score'],
                    '验证集RMSE': eval_results['rmse'],
                    '验证集MAE': eval_results['mae'],
                    '耗时(秒)': search_results['time_seconds'],
                    '最佳参数': str(search_results['best_params'])
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('验证集R²', ascending=False)
        
        print("\n📊 所有实验结果汇总:")
        print(summary_df.to_string(index=False))
        
        # 保存结果
        summary_file = self.output_dir / "optimization_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        # 保存详细结果
        detailed_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(X),
                'features': list(X.columns),
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            },
            'best_model': {
                'name': best_model_info['model_name'] if best_model_info else None,
                'search_type': best_model_info['search_type'] if best_model_info else None,
                'params': best_model_info['params'] if best_model_info else None,
                'validation_r2': best_overall_r2,
                'test_evaluation': final_evaluation if best_model_info else None
            },
            'all_results': summary_data
        }
        
        results_file = self.output_dir / "detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 特征重要性分析
        if best_model_info and hasattr(best_model_info['model'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model_info['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 特征重要性分析 (前10名):")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
            
            importance_file = self.output_dir / "feature_importance.csv"
            feature_importance.to_csv(importance_file, index=False, encoding='utf-8')
        
        print(f"\n✅ 超参数优化完成!")
        print(f"📁 所有结果已保存到: {self.output_dir}")
        print(f"📄 主要文件:")
        print(f"   - optimization_summary.csv: 结果汇总表")
        print(f"   - detailed_results.json: 详细实验数据")
        print(f"   - feature_importance.csv: 特征重要性")
        print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return detailed_results, self.output_dir

def main():
    """主函数 - 点击运行按钮(▶️)执行此函数"""
    try:
        print("🚀 启动FRP超参数优化主程序...")
        
        optimizer = FRPHyperparameterOptimizer()
        results, output_dir = optimizer.run_full_optimization()
        
        print(f"\n🎉 FRP超参数优化主程序执行成功!")
        print(f"🏆 获得了经过充分优化的FRP耐久性预测模型")
        
    except Exception as e:
        print(f"❌ 主程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()