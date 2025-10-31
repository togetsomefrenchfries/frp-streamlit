#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP钢筋耐久性预测 - 最佳模型评估脚本

功能：
1. 从140参数实验结果中提取每个模型的最佳超参数
2. 随机选取500条数据进行预测评估
3. 使用85%/15%划分进行模型重新训练和测试
4. 输出R²和RMSE指标
5. 包含5折交叉验证流程
"""

import pandas as pd
import numpy as np
import json
import warnings
import time
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 尝试导入可选的模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

warnings.filterwarnings('ignore')

class BestModelEvaluator:
    """最佳模型评估器"""
    
    def __init__(self):
        self.best_params = {}
        self.data_loader = None
        
    def load_best_parameters(self, experiment_file=None):
        """从实验结果中加载最佳参数"""
        print("🔍 查找最新的实验结果文件...")
        
        if experiment_file is None:
            # 自动找到最新的实验文件
            exp_dir = Path("experiments")
            if not exp_dir.exists():
                raise FileNotFoundError("experiments目录不存在！请先运行140参数实验。")
            
            # 查找最新的详细结果文件
            json_files = list(exp_dir.glob("*_detailed.json"))
            if not json_files:
                raise FileNotFoundError("未找到实验结果文件！请先运行140参数实验。")
            
            # 按修改时间排序，选择最新的
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            experiment_file = latest_file
        
        print(f"📂 加载实验结果: {experiment_file}")
        
        # 读取实验结果
        with open(experiment_file, 'r', encoding='utf-8') as f:
            exp_data = json.load(f)
        
        results = exp_data['results']
        
        # 按模型分组并找到最佳参数
        model_results = {}
        for result in results:
            # 兼容新旧格式的结果数据
            model_name = result.get('model', 'Unknown')
            
            # 检查结果是否有效（有R²分数）
            test_r2 = result.get('test_r2', result.get('test_score', -np.inf))
            if test_r2 != -np.inf and not np.isnan(test_r2):
                if model_name not in model_results:
                    model_results[model_name] = []
                model_results[model_name].append(result)
        
        # 为每个模型找到最佳参数（按test_r2排序）
        for model_name, model_list in model_results.items():
            if model_list:
                # 按test_r2降序排序
                best_result = max(model_list, key=lambda x: x.get('test_r2', x.get('test_score', -np.inf)))
                
                # 提取参数 - 兼容不同格式
                params = best_result.get('params', {})
                if isinstance(params, str):
                    # 如果params是字符串，尝试解析
                    try:
                        import ast
                        params = ast.literal_eval(params)
                    except:
                        # 如果解析失败，从config字段获取
                        config_str = best_result.get('config', '{}')
                        try:
                            params = ast.literal_eval(config_str)
                        except:
                            params = {}
                
                self.best_params[model_name] = {
                    'params': params,
                    'cv_r2_mean': best_result.get('cv_mean', best_result.get('cv_r2_mean', 0)),
                    'test_r2': best_result.get('test_r2', best_result.get('test_score', 0)),
                    'test_rmse': np.sqrt(best_result.get('test_mse', best_result.get('test_rmse', np.inf)**2))
                }
        
        print("✅ 最佳参数加载完成:")
        for model_name, info in self.best_params.items():
            print(f"   {model_name}: R²={info['test_r2']:.4f}, RMSE={info['test_rmse']:.4f}")
        
        return self.best_params
    
    def load_data_from_experiment(self):
        """从run_40param_experiment.py加载数据处理逻辑"""
        print("📂 使用实验脚本的数据加载逻辑...")
        
        # 导入数据加载器
        import sys
        sys.path.append('.')
        
        try:
            from run_40param_experiment import ValidDataLoader
        except ImportError:
            raise ImportError("无法导入ValidDataLoader！请确保run_40param_experiment.py存在。")
        
        # 使用相同的数据加载和处理流程
        loader = ValidDataLoader()
        data = loader.load_valid_data()
        
        if data is None or len(data) == 0:
            raise ValueError("无法加载有效数据！")
        
        # 使用相同的特征提取流程
        result = loader.prepare_features_target(data, generate_plots=False)
        
        if result is None or len(result) != 4:
            raise ValueError("特征提取失败！")
        
        X, y, feature_names, sample_weights = result
        
        print(f"✅ 数据加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
        print(f"   特征: {feature_names}")
        print(f"   目标范围: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y, feature_names, sample_weights
    
    def evaluate_with_random_500(self, X, y, feature_names):
        """任务1：随机选取500条数据进行预测评估"""
        print("\n" + "="*80)
        print("📊 任务1: 使用最佳参数在随机500条数据上进行预测评估")
        print("="*80)
        
        # 检查数据量
        if len(X) < 20:
            print(f"⚠️ 数据量太少 ({len(X)} 样本)，建议至少20个样本才能进行有效评估")
            return {}
        
        # 随机选取数据（如果数据不足500条，则使用全部数据）
        n_samples = min(500, len(X))
        np.random.seed(42)  # 确保可重现性
        random_indices = np.random.choice(len(X), size=n_samples, replace=False)
        
        X_sample = X.iloc[random_indices]
        y_sample = y.iloc[random_indices]
        
        print(f"🎲 随机选取了 {n_samples} 条数据进行评估")
        
        # 检查模型可用性
        if not self.best_params:
            print("❌ 没有找到可用的最佳参数!")
            return {}
        
        results = {}
        
        for model_name, model_info in self.best_params.items():
            print(f"\n🔬 评估模型: {model_name}")
            print(f"   最佳参数: {model_info['params']}")
            
            try:
                # 创建模型
                model = self._create_model(model_name, model_info['params'])
                if model is None:
                    continue
                
                # 数据预处理
                X_processed = self._preprocess_data(X_sample, model_name)
                
                # 训练测试分割 (80%/20%)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_sample, test_size=0.2, random_state=42
                )
                
                # 训练模型
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 计算指标
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                results[model_name] = {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'training_time': training_time,
                    'test_samples': len(y_test)
                }
                
                print(f"   ✅ R²: {r2:.4f}")
                print(f"   ✅ RMSE: {rmse:.4f}")
                print(f"   ✅ MAE: {mae:.4f}")
                print(f"   ⏱️ 训练时间: {training_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ 模型 {model_name} 评估失败: {str(e)}")
                results[model_name] = None
        
        return results
    
    def evaluate_with_85_15_split(self, X, y, feature_names):
        """任务2：使用85%/15%划分重新训练和评估，包含5折交叉验证"""
        print("\n" + "="*80)
        print("📊 任务2: 85%/15%划分 + 5折交叉验证重新训练评估")
        print("="*80)
        
        # 检查数据量
        if len(X) < 20:
            print(f"⚠️ 数据量太少 ({len(X)} 样本)，无法进行85/15划分和5折交叉验证")
            return {}
        
        # 检查模型可用性
        if not self.best_params:
            print("❌ 没有找到可用的最佳参数!")
            return {}
        
        results = {}
        
        for model_name, model_info in self.best_params.items():
            print(f"\n🔬 评估模型: {model_name}")
            print(f"   最佳参数: {model_info['params']}")
            
            try:
                # 创建模型
                model = self._create_model(model_name, model_info['params'])
                if model is None:
                    continue
                
                # 数据预处理
                X_processed = self._preprocess_data(X, model_name)
                
                # 85%/15% 划分
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.15, random_state=42
                )
                
                print(f"   📊 训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
                
                # 5折交叉验证
                print(f"   🔄 开始5折交叉验证...")
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
                cv_r2_mean = cv_scores.mean()
                cv_r2_std = cv_scores.std()
                
                print(f"   ✅ 5折CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
                
                # 在完整训练集上训练
                print(f"   🔄 在训练集上训练最终模型...")
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # 在测试集上预测
                y_pred = model.predict(X_test)
                
                # 计算指标
                test_r2 = r2_score(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = mean_absolute_error(y_test, y_pred)
                
                results[model_name] = {
                    'cv_r2_mean': cv_r2_mean,
                    'cv_r2_std': cv_r2_std,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'training_time': training_time,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                print(f"   ✅ 测试集 R²: {test_r2:.4f}")
                print(f"   ✅ 测试集 RMSE: {test_rmse:.4f}")
                print(f"   ✅ 测试集 MAE: {test_mae:.4f}")
                print(f"   ⏱️ 训练时间: {training_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ 模型 {model_name} 评估失败: {str(e)}")
                results[model_name] = None
        
        return results
    
    def _create_model(self, model_name, params):
        """创建模型实例"""
        if model_name == 'RandomForest':
            return RandomForestRegressor(**params, random_state=42)
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            # 创建XGBoost参数的副本
            xgb_params = params.copy()
            # 移除可能冲突的参数
            xgb_params.pop('random_state', None)
            return xgb.XGBRegressor(**xgb_params, random_state=42)
        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            # 创建LightGBM参数的副本
            lgb_params = params.copy()
            # 移除可能冲突的参数并进行调整
            lgb_params.pop('random_state', None)
            if 'num_leaves' in lgb_params:
                max_depth = lgb_params.get('max_depth', 6)
                if max_depth is not None:
                    max_leaves = 2 ** max_depth
                    if lgb_params['num_leaves'] > max_leaves:
                        lgb_params['num_leaves'] = max_leaves
            return lgb.LGBMRegressor(**lgb_params, random_state=42)
        else:
            print(f"   ⚠️ 模型 {model_name} 不可用或未安装")
            return None
    
    def _preprocess_data(self, X, model_name):
        """根据模型类型预处理数据"""
        if model_name in ['XGBoost', 'LightGBM']:
            # 对XGBoost和LightGBM进行标准化
            scaler = StandardScaler()
            return pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
        else:
            # RandomForest使用原始数据
            return X
    
    def save_evaluation_results(self, results_500, results_85_15):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果目录
        result_dir = Path("evaluation_results")
        result_dir.mkdir(exist_ok=True)
        
        # 合并结果
        combined_results = {
            'timestamp': timestamp,
            'experiment_type': 'best_model_evaluation',
            'task1_random_500': results_500,
            'task2_85_15_split': results_85_15,
            'best_parameters': self.best_params
        }
        
        # 保存为JSON
        json_path = result_dir / f"best_models_evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        # 创建汇总CSV
        summary_data = []
        
        # 任务1结果
        for model_name, result in results_500.items():
            if result is not None:
                summary_data.append({
                    'Task': 'Random_500',
                    'Model': model_name,
                    'R2': result['r2'],
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'Training_Time': result['training_time'],
                    'Test_Samples': result['test_samples']
                })
        
        # 任务2结果
        for model_name, result in results_85_15.items():
            if result is not None:
                summary_data.append({
                    'Task': '85_15_Split',
                    'Model': model_name,
                    'CV_R2_Mean': result['cv_r2_mean'],
                    'CV_R2_Std': result['cv_r2_std'],
                    'Test_R2': result['test_r2'],
                    'Test_RMSE': result['test_rmse'],
                    'Test_MAE': result['test_mae'],
                    'Training_Time': result['training_time'],
                    'Train_Samples': result['train_samples'],
                    'Test_Samples': result['test_samples']
                })
        
        csv_path = result_dir / f"best_models_summary_{timestamp}.csv"
        pd.DataFrame(summary_data).to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\n💾 评估结果已保存:")
        print(f"   详细结果: {json_path}")
        print(f"   汇总表格: {csv_path}")
        
        return json_path, csv_path
    
    def print_summary(self, results_500, results_85_15):
        """打印评估汇总"""
        print("\n" + "="*80)
        print("📈 评估结果汇总")
        print("="*80)
        
        print("\n🎲 任务1 - 随机500条数据预测:")
        print("-" * 50)
        for model_name, result in results_500.items():
            if result is not None:
                print(f"{model_name:12s}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
            else:
                print(f"{model_name:12s}: 评估失败")
        
        print("\n🔄 任务2 - 85%/15%划分 + 5折CV:")
        print("-" * 50)
        for model_name, result in results_85_15.items():
            if result is not None:
                print(f"{model_name:12s}: CV_R²={result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}, "
                      f"Test_R²={result['test_r2']:.4f}, Test_RMSE={result['test_rmse']:.4f}")
            else:
                print(f"{model_name:12s}: 评估失败")


def main():
    """主函数"""
    print("🚀 FRP钢筋耐久性预测 - 最佳模型评估")
    print("="*80)
    
    try:
        # 初始化评估器
        evaluator = BestModelEvaluator()
        
        # 1. 加载最佳参数
        evaluator.load_best_parameters()
        
        # 2. 加载数据
        X, y, feature_names, sample_weights = evaluator.load_data_from_experiment()
        
        # 3. 任务1：随机500条数据评估
        results_500 = evaluator.evaluate_with_random_500(X, y, feature_names)
        
        # 4. 任务2：85%/15%划分评估
        results_85_15 = evaluator.evaluate_with_85_15_split(X, y, feature_names)
        
        # 5. 保存结果
        evaluator.save_evaluation_results(results_500, results_85_15)
        
        # 6. 打印汇总
        evaluator.print_summary(results_500, results_85_15)
        
        print("\n✅ 评估完成！")
        
    except Exception as e:
        print(f"\n❌ 评估过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()