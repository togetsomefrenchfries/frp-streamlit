#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP模型12超参数实验脚本（简化版）
12 Hyperparameter Experiments for FRP Models (Simplified)

使用模拟数据进行三种机器学习模型的12超参数配置实验
8:2训练测试分割，5折交叉验证
"""

import time
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 尝试导入XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost不可用")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM不可用")

class SimplifiedTwelveParameterExperiment:
    """简化的12参数实验管理器"""
    
    def __init__(self):
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"simple_12param_exp_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_frp_demo_data(self, n_samples=2000):
        """创建FRP演示数据"""
        print("🎯 创建FRP演示数据...")
        
        np.random.seed(42)
        
        # 创建特征数据
        data = {
            'pH_environment': np.random.uniform(4, 14, n_samples),
            'exposure_time': np.random.uniform(100, 8760, n_samples),
            'temperature': np.random.uniform(10, 90, n_samples),
            'fiber_content': np.random.uniform(0.5, 8.0, n_samples),
            'diameter': np.random.uniform(6, 20, n_samples),
            'load_value': np.random.uniform(0, 80, n_samples),
            'concrete': np.random.choice([0, 1], n_samples),
            'chloride_ion': np.random.choice([0, 1], n_samples),
            'fiber_type': np.random.choice([0, 1, 2], n_samples),
            'matrix_type': np.random.choice([0, 1], n_samples),
            'surface_treatment': np.random.choice([0, 1], n_samples),
            'max_strength': np.random.uniform(400, 1800, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # 创建目标变量 - 模拟复杂关系
        ph_effect = -0.02 * (df['pH_environment'] - 8.5) ** 2
        temp_effect = -0.004 * (df['temperature'] - 20)
        time_effect = -0.12 * np.log(df['exposure_time'] / 100)
        fiber_effect = 0.03 * df['fiber_content'] - 0.002 * df['fiber_content'] ** 2
        load_effect = -0.002 * df['load_value']
        
        base_retention = 0.85
        retention = (base_retention + ph_effect + temp_effect + time_effect + 
                    fiber_effect + load_effect + np.random.normal(0, 0.05, n_samples))
        
        df['tensile_strength_retention'] = np.clip(retention, 0.1, 1.0)
        
        print(f"✅ 数据创建完成: {df.shape}")
        print(f"   目标变量范围: {df['tensile_strength_retention'].min():.3f} - {df['tensile_strength_retention'].max():.3f}")
        
        return df
        
    def define_12_parameter_sets(self):
        """定义12种不同的超参数配置"""
        
        parameter_sets = []
        
        # ========== XGBoost 12种配置 ==========
        if XGBOOST_AVAILABLE:
            xgboost_configs = [
                {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2},
                {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.15},
                {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1},
                {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
                {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.08},
                {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
                {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05},
                {'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.03},
                {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.02},
                {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05},
                {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.03},
                {'n_estimators': 600, 'max_depth': 10, 'learning_rate': 0.01}
            ]
        else:
            xgboost_configs = [None] * 12
        
        # ========== LightGBM 12种配置 ==========
        if LIGHTGBM_AVAILABLE:
            lightgbm_configs = [
                {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'num_leaves': 15},
                {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.15, 'num_leaves': 20},
                {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 31},
                {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 40},
                {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.08, 'num_leaves': 50},
                {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'num_leaves': 60},
                {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05, 'num_leaves': 70},
                {'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.03, 'num_leaves': 80},
                {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.02, 'num_leaves': 100},
                {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 40},
                {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.03, 'num_leaves': 50},
                {'n_estimators': 600, 'max_depth': 10, 'learning_rate': 0.01, 'num_leaves': 120}
            ]
        else:
            lightgbm_configs = [None] * 12
        
        # ========== Random Forest 12种配置 ==========
        rf_configs = [
            {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 8, 'min_samples_leaf': 4},
            {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
            {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 2},
            {'n_estimators': 250, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1},
            {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 300, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 400, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5},
            {'n_estimators': 400, 'max_depth': 20, 'min_samples_split': 8, 'min_samples_leaf': 4},
            {'n_estimators': 600, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
        ]
        
        # 组装所有配置
        for i in range(12):
            config_dict = {
                'name': f'Config_{i+1:02d}',
                'description': f'第{i+1}组超参数配置',
                'params': {}
            }
            
            # Random Forest (始终可用)
            rf_config = rf_configs[i].copy()
            rf_config.update({'random_state': 42, 'n_jobs': -1})
            config_dict['params']['random_forest'] = rf_config
            
            # XGBoost (如果可用)
            if XGBOOST_AVAILABLE and xgboost_configs[i]:
                xgb_config = xgboost_configs[i].copy()
                xgb_config.update({'random_state': 42})
                config_dict['params']['xgboost'] = xgb_config
            
            # LightGBM (如果可用)
            if LIGHTGBM_AVAILABLE and lightgbm_configs[i]:
                lgb_config = lightgbm_configs[i].copy()
                lgb_config.update({'random_state': 42, 'verbose': -1})
                config_dict['params']['lightgbm'] = lgb_config
            
            parameter_sets.append(config_dict)
        
        return parameter_sets
    
    def train_and_evaluate_model(self, model_name, model_params, X_train, X_test, y_train, y_test):
        """训练和评估单个模型"""
        
        try:
            # 创建模型
            if model_name == 'random_forest':
                model = RandomForestRegressor(**model_params)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(**model_params)
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(**model_params)
            else:
                return None
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 测试集预测
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # 5折交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            return {
                'test_r2': r2,
                'test_mse': mse,
                'test_mae': mae,
                'test_rmse': rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'params': model_params,
                'model': model
            }
            
        except Exception as e:
            print(f"    ❌ {model_name} 训练失败: {str(e)}")
            return None
    
    def run_all_experiments(self):
        """运行所有12组参数实验"""
        
        print("FRP钢筋耐久性预测 - 12超参数配置实验（简化版）")
        print("=" * 60)
        print("🚀 开始12组超参数配置实验")
        print(f"实验ID: {self.experiment_id}")
        print(f"结果保存目录: {self.output_dir}")
        print("配置: 8:2训练测试分割, 5折交叉验证")
        
        # 创建演示数据
        print("\n📊 准备数据...")
        
        # 尝试加载真实数据
        try:
            # 首先尝试使用data_loader加载真实数据
            sys.path.insert(0, str(Path(__file__).parent))
            from data_loader import DataLoader
            from preprocessor import FRPDataPreprocessor
            from config import config
            
            data_loader = DataLoader()
            df_raw = data_loader.load_data()
            
            if df_raw is not None and len(df_raw) > 0:
                print(f"✅ 成功加载真实数据: {df_raw.shape}")
                print(f"   数据来源: {config.DEFAULT_DATA_FILE}")
                
                # 应用预处理
                print("\n🔧 开始数据预处理...")
                preprocessor = FRPDataPreprocessor()
                df = preprocessor.preprocess_data(df_raw)
                print(f"✅ 预处理完成: {df.shape}")
                
            else:
                print("⚠️ 真实数据加载失败，使用模拟数据")
                df = self.create_frp_demo_data(n_samples=2000)
                
        except Exception as e:
            print(f"⚠️ 加载真实数据时出错: {e}")
            print("使用模拟数据进行实验")
            df = self.create_frp_demo_data(n_samples=2000)
        
        # 分离特征和目标
        # 根据数据来源确定目标变量名称
        target_candidates = [
            'tensile_strength_retention',  # 模拟数据
            'Tensile strength retention',   # 配置文件
            'retention1',                   # 真实数据选项1
            'retention2',                   # 真实数据选项2  
            'retention3',                   # 真实数据选项3
            'ultimate_tensile_strength',    # 真实数据选项4
            'tensile_modulus'              # 真实数据选项5
        ]
        
        # 查找可用的目标变量
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                # 检查该列是否有有效数据
                test_col = df[candidate]
                if not test_col.isna().all() and test_col.sum() != 0 and test_col.std() > 0:
                    target_col = candidate
                    break
        
        if target_col is None:
            # 如果找不到有效的目标变量，使用模拟数据
            print("⚠️ 找不到有效的目标变量，使用模拟数据")
            df = self.create_frp_demo_data(n_samples=2000)
            target_col = 'tensile_strength_retention'
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"目标变量: {target_col}")
        print(f"目标变量统计: 均值={y.mean():.4f}, 标准差={y.std():.4f}, 范围=[{y.min():.4f}, {y.max():.4f}]")
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        
        # 再次检查目标变量是否有效  
        if y.isna().sum() > 0.5 * len(y):  # 如果超过50%是缺失值
            print("⚠️ 目标变量缺失值过多，使用模拟数据")
            df = self.create_frp_demo_data(n_samples=2000)
            feature_cols = [col for col in df.columns if col != 'tensile_strength_retention']
            X = df[feature_cols]
            y = df['tensile_strength_retention']
            print(f"已切换到模拟数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        # 8:2分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        
        # 获取参数配置
        parameter_sets = self.define_12_parameter_sets()
        print(f"\n🔧 共{len(parameter_sets)}组参数配置")
        
        # 显示可用模型
        available_models = ['random_forest']
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available_models.append('lightgbm')
        print(f"可用模型: {', '.join(available_models)}")
        
        start_time = time.time()
        all_results = []
        
        # 运行每个实验
        for i, param_set in enumerate(parameter_sets, 1):
            print(f"\n{'='*60}")
            print(f"进度: {i}/{len(parameter_sets)} - {param_set['name']}")
            print(f"描述: {param_set['description']}")
            print(f"{'='*60}")
            
            experiment_start = time.time()
            experiment_results = {}
            
            # 训练每个可用模型
            for model_name, model_params in param_set['params'].items():
                print(f"  🔧 训练 {model_name}...")
                result = self.train_and_evaluate_model(
                    model_name, model_params, X_train, X_test, y_train, y_test
                )
                
                if result:
                    experiment_results[model_name] = result
                    print(f"    ✅ R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.4f}, CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")
            
            experiment_duration = time.time() - experiment_start
            
            # 保存实验结果
            experiment_summary = {
                'experiment_name': param_set['name'],
                'description': param_set['description'],
                'duration_seconds': experiment_duration,
                'timestamp': datetime.now().isoformat(),
                'results': experiment_results
            }
            
            all_results.append(experiment_summary)
            print(f"  ⏱️ 本实验耗时: {experiment_duration:.1f}秒")
        
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
            # 移除不可序列化的模型对象
            serializable_results = []
            for result in all_results:
                clean_result = result.copy()
                if 'results' in clean_result:
                    clean_result['results'] = {}
                    for model_name, model_result in result['results'].items():
                        clean_model_result = model_result.copy()
                        if 'model' in clean_model_result:
                            del clean_model_result['model']  # 移除模型对象
                        clean_result['results'][model_name] = clean_model_result
                serializable_results.append(clean_result)
            
            json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 创建汇总表
        summary_data = []
        for result in all_results:
            if 'results' in result:
                for model_name, model_result in result['results'].items():
                    summary_data.append({
                        '实验': result['experiment_name'],
                        '模型': model_name,
                        '测试R²': round(model_result['test_r2'], 4),
                        '测试RMSE': round(model_result['test_rmse'], 4),
                        '测试MAE': round(model_result['test_mae'], 4),
                        'CV均值': round(model_result['cv_mean'], 4),
                        'CV标准差': round(model_result['cv_std'], 4),
                        '耗时(秒)': round(result['duration_seconds'], 1)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # 保存汇总表
            summary_file = self.output_dir / "experiment_summary.csv"
            summary_df.to_csv(summary_file, index=False, encoding='utf-8')
            
            # 显示结果
            print(f"\n📊 实验结果汇总:")
            print(summary_df.to_string(index=False))
            
            # 各模型最佳配置
            print(f"\n🏆 各模型最佳配置:")
            for model in summary_df['模型'].unique():
                model_results = summary_df[summary_df['模型'] == model]
                if not model_results.empty:
                    best_idx = model_results['测试R²'].idxmax()
                    best_result = model_results.loc[best_idx]
                    print(f"  {model:15s}: {best_result['实验']} - R²={best_result['测试R²']:.4f}, RMSE={best_result['测试RMSE']:.4f}")
        
        print(f"\n📁 所有结果已保存到: {self.output_dir}")
        print("📄 文件列表:")
        print("  - detailed_results.json: 详细实验数据")
        print("  - experiment_summary.csv: 结果汇总表")

def main():
    """主函数 - 点击运行按钮(▶️)执行此函数"""
    try:
        print("🚀 启动FRP 12超参数配置实验（简化版）...")
        
        # 运行实验
        experiment = SimplifiedTwelveParameterExperiment()
        results, output_dir = experiment.run_all_experiments()
        
        print(f"\n✅ 12超参数配置实验成功完成!")
        print(f"🎯 总共完成了 {len(results)} 组实验配置")
        
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()