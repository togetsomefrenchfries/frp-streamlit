#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
750参数超参数优化实验 - Comments=1的数据
750-Parameter Hyperparameter Optimization Experiment - Comments=1 Data Only

特点：
1. 三种机器学习模型：RandomForest, XGBoost, LightGBM
2. 每个模型约250种超参数配置，总计750个配置
3. 5折交叉验证
4. 每5个配置保存一次结果
5. 只使用Comments=1的数据进行训练和验证
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import json
import pickle
from datetime import datetime
import os
import warnings
from itertools import product
import time

warnings.filterwarnings('ignore')

class ValidDataLoader:
    """有效数据加载器 - 只加载Comments=1的数据"""
    
    def __init__(self, file_path=None):
        if file_path is None:
            self.file_path = r'E:\大学\intern\2025-summer-concret\database 4.xlsx'
        else:
            self.file_path = file_path
    
    def load_valid_data(self):
        """加载有效数据（仅Comments=1）"""
        print("🔄 加载原始Excel文件...")
        raw_data = pd.read_excel(self.file_path)
        print(f"原始数据形状: {raw_data.shape}")
        
        # 筛选条件：只检查Comments列为1
        comments_mask = raw_data['Comments'] == 1
        valid_data = raw_data[comments_mask].copy()
        
        print(f"🎯 数据筛选结果:")
        print(f"  原始数据: {len(raw_data)} 行")
        print(f"  Comments=1: {comments_mask.sum()} 行")
        print(f"  最终筛选比例: {len(valid_data)/len(raw_data)*100:.1f}%")
        
        return valid_data
    
    def prepare_features_target(self, data):
        """准备特征和目标变量"""
        # 使用原始高性能特征配置的列索引（排除无效列）
        # feature_name(0), Year(5), no.(7), Value1(34), diameter(18), No.(6)
        # 移除time(50)和temperature(47)因为这些列没有有效数据
        feature_indices = [0, 5, 7, 34, 18, 6]
        
        print(f"📊 使用特征列索引: {feature_indices}")
        
        # 提取特征
        feature_columns = []
        valid_feature_indices = []
        
        for idx in feature_indices:
            if idx < len(data.columns):
                # 检查该列是否有足够的有效数据
                col_values = pd.to_numeric(data.iloc[:, idx], errors='coerce')
                valid_count = len(col_values) - col_values.isnull().sum()
                
                if valid_count > len(data) * 0.5:  # 至少50%的数据有效
                    feature_columns.append(data.columns[idx])
                    valid_feature_indices.append(idx)
                    print(f"  ✓ 列{idx}: {data.columns[idx][:30]}... - 有效数据: {valid_count}")
                else:
                    print(f"  ✗ 列{idx}: {data.columns[idx][:30]}... - 有效数据不足: {valid_count}")
        
        if len(feature_columns) < 3:
            print("❌ 有效特征列不足")
            return None, None
        
        print(f"📊 最终使用特征列: {feature_columns}")
        
        X = data[feature_columns].copy()
        
        # 查找目标变量
        target_column = None
        retention_cols = [col for col in data.columns if 'retention' in str(col).lower()]
        
        if retention_cols:
            target_column = retention_cols[0]
            print(f"🎯 找到retention目标变量: {target_column}")
        else:
            # 使用预定义的好目标变量列
            candidate_indices = [100, 97, 96, 110, 91, 90]
            
            for idx in candidate_indices:
                if idx < len(data.columns):
                    col_values = pd.to_numeric(data.iloc[:, idx], errors='coerce').dropna()
                    if len(col_values) > 1000 and col_values.std() > 0.01:
                        target_column = data.columns[idx]
                        print(f"🎯 使用目标变量: {target_column} (索引{idx}) - 有效值: {len(col_values)}")
                        break
            
            if target_column is None:
                print("❌ 未找到合适的目标变量")
                return None, None
        
        y = data[target_column].copy()
        
        # 数据清理
        print("🧹 数据清理...")
        
        # 处理数值型特征
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        y = pd.to_numeric(y, errors='coerce')
        
        # 删除缺失值 - 使用更灵活的策略
        # 只要目标变量有效，且至少有一半特征有效即可
        y_valid = ~y.isnull()
        
        # 计算每行有效特征的数量
        feature_valid_count = X.notna().sum(axis=1)
        min_features = len(X.columns) // 2  # 至少一半特征有效
        
        # 组合条件
        valid_mask = y_valid & (feature_valid_count >= min_features)
        
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 对特征中的缺失值进行填充
        X = X.fillna(X.median())
        
        # 检查目标变量的分布，移除极端异常值
        if len(y) > 10:
            q1, q3 = y.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outlier_mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[outlier_mask]
            y = y[outlier_mask]
        
        print(f"  清理后数据量: {len(X)} 行")
        print(f"  特征维度: {X.shape[1]}")
        if len(y) > 0:
            print(f"  目标变量范围: [{y.min():.3f}, {y.max():.3f}]")
            print(f"  目标变量标准差: {y.std():.3f}")
        
        return X, y

class HyperparameterGenerator:
    """750超参数配置生成器 - 优化版"""
    
    @staticmethod
    def generate_randomforest_configs(n_configs=250):
        """生成RandomForest配置 - 使用智能采样"""
        configs = []
        np.random.seed(42)
        
        # 参数范围
        param_ranges = {
            'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.5, 0.8],
            'bootstrap': [True, False],
            'criterion': ['squared_error', 'absolute_error'],
            'max_leaf_nodes': [None, 50, 100, 200],
            'min_impurity_decrease': [0.0, 0.01, 0.05],
            'ccp_alpha': [0.0, 0.01, 0.05]
        }
        
        # 随机采样生成配置
        for i in range(n_configs):
            config = {
                'n_estimators': np.random.choice(param_ranges['n_estimators']),
                'max_depth': np.random.choice(param_ranges['max_depth']),
                'min_samples_split': np.random.choice(param_ranges['min_samples_split']),
                'min_samples_leaf': np.random.choice(param_ranges['min_samples_leaf']),
                'max_features': np.random.choice(param_ranges['max_features']),
                'bootstrap': np.random.choice(param_ranges['bootstrap']),
                'criterion': np.random.choice(param_ranges['criterion']),
                'max_leaf_nodes': np.random.choice(param_ranges['max_leaf_nodes']),
                'min_impurity_decrease': np.random.choice(param_ranges['min_impurity_decrease']),
                'ccp_alpha': np.random.choice(param_ranges['ccp_alpha']),
                'random_state': 42,
                'n_jobs': -1
            }
            configs.append(config)
        
        return configs
    
    @staticmethod
    def generate_xgboost_configs(n_configs=250):
        """生成XGBoost配置 - 使用智能采样"""
        configs = []
        np.random.seed(43)
        
        # 参数范围
        param_ranges = {
            'n_estimators': [50, 100, 150, 200, 300, 400],
            'max_depth': [3, 5, 7, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.5],
            'colsample_bylevel': [0.8, 0.9, 1.0]
        }
        
        # 随机采样生成配置
        for i in range(n_configs):
            config = {
                'n_estimators': np.random.choice(param_ranges['n_estimators']),
                'max_depth': np.random.choice(param_ranges['max_depth']),
                'learning_rate': np.random.choice(param_ranges['learning_rate']),
                'subsample': np.random.choice(param_ranges['subsample']),
                'colsample_bytree': np.random.choice(param_ranges['colsample_bytree']),
                'reg_alpha': np.random.choice(param_ranges['reg_alpha']),
                'reg_lambda': np.random.choice(param_ranges['reg_lambda']),
                'min_child_weight': np.random.choice(param_ranges['min_child_weight']),
                'gamma': np.random.choice(param_ranges['gamma']),
                'colsample_bylevel': np.random.choice(param_ranges['colsample_bylevel']),
                'random_state': 42,
                'objective': 'reg:squarederror',
                'n_jobs': -1
            }
            configs.append(config)
        
        return configs
    
    @staticmethod
    def generate_lightgbm_configs(n_configs=250):
        """生成LightGBM配置 - 使用智能采样"""
        configs = []
        np.random.seed(44)
        
        # 参数范围
        param_ranges = {
            'n_estimators': [50, 100, 150, 200, 300, 400],
            'max_depth': [5, 10, 15, 20, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'num_leaves': [31, 50, 100, 150, 200],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
            'min_child_samples': [10, 20, 30, 50],
            'min_split_gain': [0.0, 0.01, 0.05],
            'subsample_freq': [0, 1, 5]
        }
        
        # 随机采样生成配置
        for i in range(n_configs):
            config = {
                'n_estimators': np.random.choice(param_ranges['n_estimators']),
                'max_depth': np.random.choice(param_ranges['max_depth']),
                'learning_rate': np.random.choice(param_ranges['learning_rate']),
                'num_leaves': np.random.choice(param_ranges['num_leaves']),
                'subsample': np.random.choice(param_ranges['subsample']),
                'colsample_bytree': np.random.choice(param_ranges['colsample_bytree']),
                'reg_alpha': np.random.choice(param_ranges['reg_alpha']),
                'reg_lambda': np.random.choice(param_ranges['reg_lambda']),
                'min_child_samples': np.random.choice(param_ranges['min_child_samples']),
                'min_split_gain': np.random.choice(param_ranges['min_split_gain']),
                'subsample_freq': np.random.choice(param_ranges['subsample_freq']),
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_jobs': -1,
                'verbose': -1
            }
            configs.append(config)
        
        return configs

class ValidDataExperimentRunner:
    """有效数据750参数实验运行器"""
    
    def __init__(self, X, y, experiment_name="valid_750param_exp"):
        self.X = X
        self.y = y
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"experiments/{experiment_name}_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 数据分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 数据分割完成:")
        print(f"  训练集: {len(self.X_train)} 样本")
        print(f"  测试集: {len(self.X_test)} 样本")
        
        # 交叉验证设置
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 结果存储
        self.all_results = []
    
    def evaluate_model(self, model, config, model_name, config_id):
        """评估单个模型配置"""
        start_time = time.time()
        
        try:
            # 交叉验证
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=self.cv, scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # 训练并在测试集上评估
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            test_r2 = r2_score(self.y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            test_mae = mean_absolute_error(self.y_test, y_pred)
            
            total_time = time.time() - start_time
            
            result = {
                'model': model_name,
                'config_id': config_id,
                'config': config,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'total_time': total_time
            }
            
            print(f"  配置{config_id}: CV R²={cv_mean:.4f}, 测试R²={test_r2:.4f}")
            
            return result
            
        except Exception as e:
            print(f"  配置{config_id}: 失败 - {str(e)}")
            return None
    
    def save_batch_results(self, batch_num, batch_results):
        """保存批次结果"""
        if not batch_results:
            return
        
        # 保存CSV
        df = pd.DataFrame(batch_results)
        csv_path = os.path.join(self.results_dir, f"results_batch_{batch_num:03d}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 转换数据为JSON可序列化格式
        json_serializable_results = []
        for result in batch_results:
            serializable_result = {}
            for key, value in result.items():
                if key == 'config':
                    # 配置字典需要特殊处理
                    serializable_config = {}
                    for config_key, config_value in value.items():
                        # 转换NumPy类型为Python原生类型
                        if hasattr(config_value, 'item'):
                            serializable_config[config_key] = config_value.item()
                        elif isinstance(config_value, (np.integer, np.floating)):
                            serializable_config[config_key] = config_value.item()
                        elif config_value is None:
                            serializable_config[config_key] = None
                        else:
                            serializable_config[config_key] = config_value
                    serializable_result[key] = serializable_config
                else:
                    # 处理其他值
                    if hasattr(value, 'item'):
                        serializable_result[key] = value.item()
                    elif isinstance(value, (np.integer, np.floating)):
                        serializable_result[key] = value.item()
                    elif pd.isna(value):
                        serializable_result[key] = None
                    else:
                        serializable_result[key] = value
            json_serializable_results.append(serializable_result)
        
        # 保存JSON
        json_path = os.path.join(self.results_dir, f"results_batch_{batch_num:03d}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 批次{batch_num}结果已保存")
    
    def run_experiment(self):
        """运行完整实验"""
        print(f"🚀 开始有效数据750参数实验")
        print(f"📁 结果保存位置: {self.results_dir}")
        
        # 生成超参数配置
        print("\n🔧 生成超参数配置...")
        rf_configs = HyperparameterGenerator.generate_randomforest_configs(250)
        xgb_configs = HyperparameterGenerator.generate_xgboost_configs(250)
        lgb_configs = HyperparameterGenerator.generate_lightgbm_configs(250)
        
        print(f"  RandomForest: {len(rf_configs)} 配置")
        print(f"  XGBoost: {len(xgb_configs)} 配置")  
        print(f"  LightGBM: {len(lgb_configs)} 配置")
        print(f"  总计: {len(rf_configs) + len(xgb_configs) + len(lgb_configs)} 配置")
        
        # 执行实验
        config_id = 1
        batch_num = 1
        batch_results = []
        
        best_results = {'RandomForest': None, 'XGBoost': None, 'LightGBM': None}
        
        # RandomForest实验
        print(f"\n🌲 RandomForest实验 ({len(rf_configs)}个配置)")
        for i, config in enumerate(rf_configs):
            model = RandomForestRegressor(**config)
            result = self.evaluate_model(model, config, 'RandomForest', config_id)
            
            if result:
                batch_results.append(result)
                self.all_results.append(result)
                
                # 更新最佳结果
                if best_results['RandomForest'] is None or result['test_r2'] > best_results['RandomForest']['test_r2']:
                    best_results['RandomForest'] = result
            
            config_id += 1
            
            # 每5个配置保存一次
            if len(batch_results) >= 5:
                self.save_batch_results(batch_num, batch_results)
                batch_results = []
                batch_num += 1
        
        # XGBoost实验
        print(f"\n🚀 XGBoost实验 ({len(xgb_configs)}个配置)")
        for i, config in enumerate(xgb_configs):
            model = xgb.XGBRegressor(**config)
            result = self.evaluate_model(model, config, 'XGBoost', config_id)
            
            if result:
                batch_results.append(result)
                self.all_results.append(result)
                
                # 更新最佳结果
                if best_results['XGBoost'] is None or result['test_r2'] > best_results['XGBoost']['test_r2']:
                    best_results['XGBoost'] = result
            
            config_id += 1
            
            # 每5个配置保存一次
            if len(batch_results) >= 5:
                self.save_batch_results(batch_num, batch_results)
                batch_results = []
                batch_num += 1
        
        # LightGBM实验
        print(f"\n💡 LightGBM实验 ({len(lgb_configs)}个配置)")
        for i, config in enumerate(lgb_configs):
            model = lgb.LGBMRegressor(**config)
            result = self.evaluate_model(model, config, 'LightGBM', config_id)
            
            if result:
                batch_results.append(result)
                self.all_results.append(result)
                
                # 更新最佳结果
                if best_results['LightGBM'] is None or result['test_r2'] > best_results['LightGBM']['test_r2']:
                    best_results['LightGBM'] = result
            
            config_id += 1
            
            # 每5个配置保存一次
            if len(batch_results) >= 5:
                self.save_batch_results(batch_num, batch_results)
                batch_results = []
                batch_num += 1
        
        # 保存剩余结果
        if batch_results:
            self.save_batch_results(batch_num, batch_results)
        
        # 保存完整结果
        self.save_final_results(best_results)
        
        # 打印摘要
        self.print_summary(best_results)
        
        return self.results_dir
    
    def save_final_results(self, best_results):
        """保存最终结果"""
        # 保存完整CSV
        if self.all_results:
            df_all = pd.DataFrame(self.all_results)
            df_all.to_csv(os.path.join(self.results_dir, "complete_results.csv"), 
                         index=False, encoding='utf-8-sig')
            
            # 保存完整JSON
            with open(os.path.join(self.results_dir, "complete_results.json"), 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        
        # 保存最终报告
        report_path = os.path.join(self.results_dir, "final_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("有效数据750参数实验报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"实验时间: {datetime.now()}\n")
            f.write(f"数据筛选条件: Comments=1\n")
            f.write(f"总配置数: {len(self.all_results)}\n\n")
            
            for model_name, result in best_results.items():
                if result:
                    f.write(f"{model_name} 最佳结果:\n")
                    f.write(f"  CV R²: {result['cv_mean']:.4f}±{result['cv_std']:.4f}\n")
                    f.write(f"  测试R²: {result['test_r2']:.4f}\n")
                    f.write(f"  配置: {result['config']}\n\n")
            
            # 找到整体最佳
            if self.all_results:
                best_overall = max(self.all_results, key=lambda x: x['test_r2'])
                f.write("整体最佳配置:\n")
                f.write(f"  模型: {best_overall['model']}\n")
                f.write(f"  CV R²: {best_overall['cv_mean']:.4f}\n")
                f.write(f"  测试R²: {best_overall['test_r2']:.4f}\n")
    
    def print_summary(self, best_results):
        """打印实验摘要"""
        print(f"\n📈 实验摘要:")
        for model_name, result in best_results.items():
            if result:
                print(f"  {model_name}: CV R²={result['cv_mean']:.4f}, 测试R²={result['test_r2']:.4f}")
        
        print(f"\n🎉 实验完成！总配置: {len(self.all_results)}")
        print(f"结果保存在: {self.results_dir}")

def main():
    """主函数"""
    print("🎯 有效数据750参数超参数优化实验")
    print("筛选条件: Comments=1")
    print("=" * 50)
    
    # 加载有效数据
    loader = ValidDataLoader()
    valid_data = loader.load_valid_data()
    X, y = loader.prepare_features_target(valid_data)
    
    if X is None or len(X) == 0:
        print("❌ 没有有效数据，实验终止")
        return
    
    # 运行实验
    runner = ValidDataExperimentRunner(X, y)
    results_dir = runner.run_experiment()
    
    print(f"\n✅ 实验完成！结果保存在: {results_dir}")

if __name__ == "__main__":
    main()