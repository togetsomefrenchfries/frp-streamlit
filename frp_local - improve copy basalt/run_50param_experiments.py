#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP模型超参数网格搜索实验
Three ML models with ~50 hyperparameter configurations each
5-fold CV with incremental saving every 5 configurations
"""

import time
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import pickle
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

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class FiftyParameterExperiment:
    """50参数超参数网格搜索实验管理器"""
    
    def __init__(self):
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"50param_exp_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存间隔
        self.save_interval = 5
        
    def generate_parameter_grids(self):
        """生成三种模型的50种超参数配置"""
        
        parameter_grids = {}
        
        # ========== Random Forest 50种配置 ==========
        rf_configs = []
        
        # 基础参数范围
        n_estimators_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800]
        max_depth_list = [3, 5, 8, 10, 12, 15, 20, 25, None]
        min_samples_split_list = [2, 5, 10, 15, 20]
        min_samples_leaf_list = [1, 2, 4, 8, 10]
        max_features_list = ['sqrt', 'log2', 0.3, 0.5, 0.7, None]
        
        # 生成50种配置（通过组合和采样）
        import itertools
        import random
        
        # 设置随机种子以确保可重复性
        random.seed(42)
        np.random.seed(42)
        
        # 生成全部组合的子集
        all_combinations = list(itertools.product(
            n_estimators_list[:5],  # 取前5个
            max_depth_list[:5],     # 取前5个
            min_samples_split_list[:2],  # 取前2个
            min_samples_leaf_list[:2],   # 取前2个
            max_features_list[:2]        # 取前2个
        ))
        
        # 随机选择50个组合
        selected_combinations = random.sample(all_combinations, min(50, len(all_combinations)))
        
        # 如果不够50个，补充随机配置
        while len(selected_combinations) < 50:
            config = (
                random.choice(n_estimators_list),
                random.choice(max_depth_list),
                random.choice(min_samples_split_list),
                random.choice(min_samples_leaf_list),
                random.choice(max_features_list)
            )
            if config not in selected_combinations:
                selected_combinations.append(config)
        
        for n_est, max_d, min_split, min_leaf, max_feat in selected_combinations:
            rf_configs.append({
                'n_estimators': n_est,
                'max_depth': max_d,
                'min_samples_split': min_split,
                'min_samples_leaf': min_leaf,
                'max_features': max_feat,
                'random_state': 42
            })
        
        parameter_grids['RandomForest'] = rf_configs
        
        # ========== XGBoost 50种配置 ==========
        if XGBOOST_AVAILABLE:
            xgb_configs = []
            
            n_estimators_xgb = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800]
            max_depth_xgb = [3, 4, 5, 6, 7, 8, 9, 10]
            learning_rate_xgb = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
            subsample_xgb = [0.6, 0.7, 0.8, 0.9, 1.0]
            colsample_bytree_xgb = [0.6, 0.7, 0.8, 0.9, 1.0]
            
            # 生成50种XGBoost配置
            xgb_combinations = list(itertools.product(
                n_estimators_xgb[:4],
                max_depth_xgb[:4],
                learning_rate_xgb[:3],
                subsample_xgb[:2],
                colsample_bytree_xgb[:2]
            ))
            
            selected_xgb = random.sample(xgb_combinations, min(50, len(xgb_combinations)))
            
            while len(selected_xgb) < 50:
                config = (
                    random.choice(n_estimators_xgb),
                    random.choice(max_depth_xgb),
                    random.choice(learning_rate_xgb),
                    random.choice(subsample_xgb),
                    random.choice(colsample_bytree_xgb)
                )
                if config not in selected_xgb:
                    selected_xgb.append(config)
            
            for n_est, max_d, lr, sub, col in selected_xgb:
                xgb_configs.append({
                    'n_estimators': n_est,
                    'max_depth': max_d,
                    'learning_rate': lr,
                    'subsample': sub,
                    'colsample_bytree': col,
                    'random_state': 42
                })
            
            parameter_grids['XGBoost'] = xgb_configs
        
        # ========== LightGBM 50种配置 ==========
        if LIGHTGBM_AVAILABLE:
            lgb_configs = []
            
            n_estimators_lgb = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800]
            max_depth_lgb = [3, 4, 5, 6, 7, 8, 9, 10, -1]
            learning_rate_lgb = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
            num_leaves_lgb = [15, 31, 50, 70, 100, 120, 150, 200]
            subsample_lgb = [0.6, 0.7, 0.8, 0.9, 1.0]
            
            # 生成50种LightGBM配置
            lgb_combinations = list(itertools.product(
                n_estimators_lgb[:4],
                max_depth_lgb[:4],
                learning_rate_lgb[:3],
                num_leaves_lgb[:3],
                subsample_lgb[:2]
            ))
            
            selected_lgb = random.sample(lgb_combinations, min(50, len(lgb_combinations)))
            
            while len(selected_lgb) < 50:
                config = (
                    random.choice(n_estimators_lgb),
                    random.choice(max_depth_lgb),
                    random.choice(learning_rate_lgb),
                    random.choice(num_leaves_lgb),
                    random.choice(subsample_lgb)
                )
                if config not in selected_lgb:
                    selected_lgb.append(config)
            
            for n_est, max_d, lr, leaves, sub in selected_lgb:
                lgb_configs.append({
                    'n_estimators': n_est,
                    'max_depth': max_d,
                    'learning_rate': lr,
                    'num_leaves': leaves,
                    'subsample': sub,
                    'random_state': 42
                })
            
            parameter_grids['LightGBM'] = lgb_configs
        
        return parameter_grids
    
    def train_and_evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test, config_idx):
        """训练和评估单个模型配置"""
        
        start_time = time.time()
        
        # 5折交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        cv_time = time.time() - start_time
        
        # 训练模型并测试
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        total_time = time.time() - start_time
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'config_index': config_idx,
            'parameters': model.get_params(),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_time': cv_time,
            'total_time': total_time,
            'data_shape': {'train': X_train.shape, 'test': X_test.shape}
        }
        
        return result
    
    def save_intermediate_results(self, batch_number):
        """保存中间结果"""
        
        # 保存JSON格式结果
        json_file = self.output_dir / f"results_batch_{batch_number:03d}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式结果
        if self.results:
            df_results = pd.DataFrame(self.results)
            csv_file = self.output_dir / f"results_batch_{batch_number:03d}.csv"
            df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存汇总统计
        summary_file = self.output_dir / f"summary_batch_{batch_number:03d}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"实验ID: {self.experiment_id}\n")
            f.write(f"批次: {batch_number}\n")
            f.write(f"完成配置数: {len(self.results)}\n")
            f.write(f"保存时间: {datetime.now()}\n\n")
            
            if self.results:
                df = pd.DataFrame(self.results)
                f.write("模型性能汇总:\n")
                f.write("="*50 + "\n")
                
                for model_name in df['model_name'].unique():
                    model_results = df[df['model_name'] == model_name]
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  配置数量: {len(model_results)}\n")
                    f.write(f"  最佳CV R²: {model_results['cv_mean'].max():.4f}\n")
                    f.write(f"  最佳测试R²: {model_results['test_r2'].max():.4f}\n")
                    f.write(f"  平均CV时间: {model_results['cv_time'].mean():.2f}s\n")
        
        print(f"✅ 批次 {batch_number} 结果已保存到 {self.output_dir}")
    
    def run_experiment(self):
        """运行完整的50参数实验"""
        
        print("🚀 开始50参数超参数网格搜索实验")
        print(f"实验ID: {self.experiment_id}")
        print(f"结果保存目录: {self.output_dir}")
        print("配置: 8:2训练测试分割, 5折交叉验证")
        print(f"保存间隔: 每{self.save_interval}个配置")
        
        # 加载和预处理数据
        print("\n📊 准备数据...")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from data_loader import DataLoader
            from preprocessor import FRPDataPreprocessor
            from config import config
            
            data_loader = DataLoader()
            df_raw = data_loader.load_data()
            
            if df_raw is not None and len(df_raw) > 0:
                print(f"✅ 成功加载真实数据: {df_raw.shape}")
                
                # 应用预处理
                print("🔧 开始数据预处理...")
                preprocessor = FRPDataPreprocessor()
                df = preprocessor.preprocess_data(df_raw)
                print(f"✅ 预处理完成: {df.shape}")
                
            else:
                raise Exception("数据加载失败")
                
        except Exception as e:
            print(f"⚠️ 加载真实数据失败: {e}")
            print("程序终止")
            return
        
        # 准备训练数据
        target_col = 'Tensile strength retention'
        if target_col not in df.columns:
            print(f"❌ 目标变量 '{target_col}' 不存在")
            return
        
        # 分离特征和目标
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
        
        print(f"📊 最终数据维度: X={X.shape}, y={y.shape}")
        print(f"特征列: {list(X.columns)}")
        
        # 8:2分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 生成超参数网格
        print("\n⚙️ 生成超参数配置...")
        parameter_grids = self.generate_parameter_grids()
        
        total_configs = sum(len(configs) for configs in parameter_grids.values())
        print(f"总配置数: {total_configs}")
        for model_name, configs in parameter_grids.items():
            print(f"  {model_name}: {len(configs)} 种配置")
        
        # 运行实验
        print(f"\n🔬 开始实验...")
        start_time = time.time()
        config_count = 0
        batch_count = 0
        
        for model_name, configs in parameter_grids.items():
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            for i, params in enumerate(configs, 1):
                print(f"\n🔧 {model_name} 配置 {i}/{len(configs)}")
                print(f"参数: {params}")
                
                # 创建模型
                if model_name == 'RandomForest':
                    model = RandomForestRegressor(**params)
                elif model_name == 'XGBoost':
                    model = xgb.XGBRegressor(**params)
                elif model_name == 'LightGBM':
                    model = lgb.LGBMRegressor(**params)
                
                # 训练和评估
                try:
                    result = self.train_and_evaluate_model(
                        model_name, model, X_train, y_train, X_test, y_test, i
                    )
                    self.results.append(result)
                    
                    print(f"✅ CV R²: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")
                    print(f"   测试R²: {result['test_r2']:.4f}")
                    print(f"   用时: {result['total_time']:.2f}s")
                    
                    config_count += 1
                    
                    # 每5个配置保存一次
                    if config_count % self.save_interval == 0:
                        batch_count += 1
                        self.save_intermediate_results(batch_count)
                        
                        elapsed = time.time() - start_time
                        remaining = (total_configs - config_count) * (elapsed / config_count)
                        print(f"\n📊 进度: {config_count}/{total_configs} ({config_count/total_configs*100:.1f}%)")
                        print(f"   已用时间: {elapsed/3600:.2f}小时")
                        print(f"   预计剩余: {remaining/3600:.2f}小时")
                    
                except Exception as e:
                    print(f"❌ 配置失败: {e}")
                    continue
        
        # 保存最终结果
        final_batch = batch_count + 1
        self.save_intermediate_results(final_batch)
        
        # 生成最终报告
        self.generate_final_report()
        
        total_time = time.time() - start_time
        print(f"\n🎉 实验完成!")
        print(f"总用时: {total_time/3600:.2f}小时")
        print(f"完成配置: {len(self.results)}/{total_configs}")
        print(f"结果保存在: {self.output_dir}")
    
    def generate_final_report(self):
        """生成最终实验报告"""
        
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # 详细报告
        report_file = self.output_dir / "final_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FRP 50参数超参数搜索实验报告\n")
            f.write("="*60 + "\n\n")
            f.write(f"实验ID: {self.experiment_id}\n")
            f.write(f"完成时间: {datetime.now()}\n")
            f.write(f"总配置数: {len(self.results)}\n\n")
            
            # 按模型统计
            for model_name in df['model_name'].unique():
                model_df = df[df['model_name'] == model_name]
                f.write(f"{model_name} 结果汇总:\n")
                f.write("-" * 30 + "\n")
                f.write(f"配置数量: {len(model_df)}\n")
                f.write(f"CV R² - 最佳: {model_df['cv_mean'].max():.4f}\n")
                f.write(f"CV R² - 平均: {model_df['cv_mean'].mean():.4f}\n")
                f.write(f"CV R² - 最差: {model_df['cv_mean'].min():.4f}\n")
                f.write(f"测试R² - 最佳: {model_df['test_r2'].max():.4f}\n")
                f.write(f"测试R² - 平均: {model_df['test_r2'].mean():.4f}\n")
                f.write(f"平均训练时间: {model_df['total_time'].mean():.2f}s\n")
                
                # 最佳配置
                best_idx = model_df['cv_mean'].idxmax()
                best_config = model_df.loc[best_idx]
                f.write(f"最佳配置参数: {best_config['parameters']}\n\n")
            
            # 整体最佳
            overall_best_idx = df['cv_mean'].idxmax()
            overall_best = df.loc[overall_best_idx]
            f.write("整体最佳配置:\n")
            f.write("-" * 20 + "\n")
            f.write(f"模型: {overall_best['model_name']}\n")
            f.write(f"CV R²: {overall_best['cv_mean']:.4f}\n")
            f.write(f"测试R²: {overall_best['test_r2']:.4f}\n")
            f.write(f"参数: {overall_best['parameters']}\n")

if __name__ == "__main__":
    experiment = FiftyParameterExperiment()
    experiment.run_experiment()