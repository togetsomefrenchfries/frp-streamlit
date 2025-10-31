#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP钢筋耐久性预测 - MLP神经网络实验 (简化版)

特点：
1. 复用已验证的CatBoost数据预处理流程
2. 简化神经网络预处理，避免重复数据分割
3. 基于sklearn MLPRegressor的深度学习模型
4. 多种MLP架构和超参数优化
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import json
import warnings
from datetime import datetime
from pathlib import Path
import sys
import os

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

# 导入MLPRegressor
try:
    from sklearn.neural_network import MLPRegressor
    SKLEARN_MLP_AVAILABLE = True
    print("✅ Sklearn MLPRegressor可用")
except ImportError:
    print("❌ Sklearn MLPRegressor不可用")
    SKLEARN_MLP_AVAILABLE = False
    sys.exit(1)

warnings.filterwarnings('ignore')

# 导入已验证的数据加载器
class ValidDataLoader:
    """已验证的数据加载器 - 与CatBoost实验完全一致"""
    
    def __init__(self, file_path=None):
        if file_path is None:
            # 扩展搜索路径
            possible_paths = [
                "E:/大学/intern/2025-summer-concret/database 4.xlsx",
                "E:\\大学\\intern\\2025-summer-concret\\database 4.xlsx",
                "../database 4.xlsx",
                "../../database 4.xlsx",
                "data/database 4.xlsx",
                "../data/database 4.xlsx",
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    self.file_path = path
                    print(f"✅ 找到数据文件: {path}")
                    break
            else:
                print("❌ 未找到database 4.xlsx文件")
                self.file_path = None
        else:
            self.file_path = file_path
    
    def load_valid_data(self):
        """加载有效数据 - 包含完整的筛选逻辑"""
        if self.file_path is None:
            return None
        
        try:
            print(f"🔄 加载{Path(self.file_path).name}文件...")
            df = pd.read_excel(self.file_path, header=None)
            print(f"✅ Excel读取成功，数据形状: {df.shape}")
            
            print(f"🔍 执行数据筛选...")
            
            # 1. 第一列不为0 (相当于Comments=1的逻辑)
            first_col_valid = df.iloc[:, 0] != 0
            df_step1 = df[first_col_valid].copy()
            print(f"  第一列不为0筛选: {len(df)} -> {len(df_step1)} 行")
            
            # 2. 检查目标变量列是否包含tensile相关数据
            target_candidates = [100, 97, 34]  # retention1, Value1_result, Value1
            valid_target_rows = pd.Series([False] * len(df_step1), index=df_step1.index)
            
            for pos in target_candidates:
                if pos < len(df_step1.columns):
                    col_data = df_step1.iloc[:, pos]
                    # 检查是否为数值型目标变量
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    
                    # 对于retention类目标，值应该在0-2之间（包含百分比形式）
                    if pos == 100:  # retention1位置
                        valid_mask = numeric_data.notna() & (numeric_data >= 0) & (numeric_data <= 2)
                    else:  # Value1等位置
                        valid_mask = numeric_data.notna() & (numeric_data > 0)
                    
                    valid_target_rows = valid_target_rows | valid_mask
            
            df_step2 = df_step1[valid_target_rows].copy()
            print(f"  目标变量有效性筛选: {len(df_step1)} -> {len(df_step2)} 行")
            
            # 3. 纤维类型筛选 - 已移除限制，接受所有纤维类型
            df_filtered = df_step2
            print(f"  纤维类型筛选: 接受所有类型，保留 {len(df_filtered)} 行")
            
            print(f"🎯 最终筛选结果:")
            print(f"  原始数据: {len(df)} 行")
            print(f"  最终保留: {len(df_filtered)} 行")
            print(f"  保留比例: {len(df_filtered)/len(df)*100:.1f}%")
            
            return df_filtered
            
        except Exception as e:
            print(f"❌ 读取Excel文件失败: {e}")
            return None
    
    def prepare_features_target(self, data):
        """准备特征和目标变量 - 简化版"""
        if data is None or len(data) == 0:
            return None
        
        print("\n🔧 开始简化的特征提取...")
        print("=" * 60)
        
        # 数据清理
        data_clean = data.copy()
        
        # 基于database 4.xlsx真实结构的特征定义 (简化版)
        feature_positions = [54, 61, 53, 18, 90, 15, 34, 8, 10, 51, 49, 22, 12]
        feature_names = [
            'pH_environment', 'Chloride_ion', 'concrete', 'diameter',
            'load_value', 'fiber_content', 'tensile_strength',
            'Glass_or_Basalt', 'Vinyl_ester_or_Epoxy', 'condition_time',
            'Temperature', 'surface_treatment', 'glass_transition_temp'
        ]
        
        # 提取特征
        X_data = []
        for i, pos in enumerate(feature_positions):
            if pos < len(data_clean.columns):
                col_data = data_clean.iloc[:, pos].copy()
                
                # 数据清理 - 根据特征类型进行不同处理
                def clean_value(value, feature_name):
                    if pd.isna(value):
                        return np.nan
                    
                    # 纤维类型和树脂类型保持字符串
                    if 'Glass_or_Basalt' in feature_name or 'Vinyl_ester_or_Epoxy' in feature_name:
                        return str(value)
                    
                    # 数值特征转换为float
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        str_val = str(value)
                        if ',' in str_val:
                            try:
                                return float(str_val.split(',')[0])
                            except:
                                pass
                        try:
                            numbers = re.findall(r'\d+\.?\d*', str_val)
                            if numbers:
                                return float(numbers[0])
                        except:
                            pass
                        return np.nan
                
                col_data = col_data.apply(lambda x: clean_value(x, feature_names[i]))
                X_data.append(col_data)
            else:
                print(f"⚠️ 位置{pos}超出范围")
                X_data.append(pd.Series([np.nan] * len(data_clean)))
        
        X = pd.DataFrame(X_data).T
        X.columns = feature_names
        
        # 提取目标变量 - 按优先级选择最佳目标变量
        target_candidates = [100, 97, 34]  # retention1, Value1_result, Value1
        y = None
        target_pos = None
        
        for pos in target_candidates:
            if pos < len(data_clean.columns):
                col_data = data_clean.iloc[:, pos].copy()
                
                # 清理目标变量
                def clean_target(value):
                    if pd.isna(value):
                        return np.nan
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return np.nan
                
                y_candidate = col_data.apply(clean_target)
                
                # 检查有效数据比例
                valid_ratio = y_candidate.notna().sum() / len(y_candidate)
                if valid_ratio > 0.3:  # 至少30%有效数据
                    y = y_candidate
                    target_pos = pos
                    print(f"✅ 选择目标变量: 位置{pos} (有效率: {valid_ratio:.1%})")
                    break
        
        if y is None:
            print("❌ 未找到合适的目标变量")
            return None
        
        # 数据有效性筛选
        print("🔍 数据有效性筛选...")
        
        # 接受所有纤维类型
        fiber_col = 'Glass_or_Basalt'
        
        # 先检查数据是否已经被清理过度
        print(f"纤维类型列的NaN数量: {X[fiber_col].isna().sum()}")
        print(f"纤维类型列的唯一值示例: {X[fiber_col].dropna().head(10).tolist()}")
        
        # 移除纤维类型限制，只要有纤维类型信息就保留
        fiber_series = X[fiber_col].astype(str)
        fiber_mask = fiber_series.notna() & (fiber_series != 'nan') & (fiber_series.str.strip() != '')
        
        # 其他基本条件
        basic_mask = (
            y.notna() &  # 目标变量有值
            X['condition_time'].notna() &  # 条件时间有值
            (X.notna().sum(axis=1) >= 5)  # 至少5个特征有值 (降低要求)
        )
        
        final_mask = fiber_mask & basic_mask
        final_count = final_mask.sum()
        
        print(f"  纤维类型筛选: {fiber_mask.sum()} 行")
        print(f"  基本条件筛选: {basic_mask.sum()} 行")
        print(f"  最终保留: {final_count} 行")
        
        if final_count < 50:
            print("⚠️ 有效数据太少，尝试放宽筛选条件...")
            # 如果数据太少，只筛选基本条件，不筛选纤维类型
            final_mask = basic_mask
            final_count = final_mask.sum()
            print(f"  放宽条件后保留: {final_count} 行")
        
        # 应用筛选
        X_clean = X[final_mask].copy()
        y_clean = y[final_mask].copy()
        
        # 缺失值填充 - 简化版
        print("🔧 缺失值填充...")
        for col in X_clean.columns:
            if X_clean[col].isnull().sum() > 0:
                if 'diameter' in col:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                elif 'load_value' in col:
                    X_clean[col] = X_clean[col].fillna(0)
                elif 'Temperature' in col:
                    X_clean[col] = X_clean[col].fillna(25.0)
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # 字符串特征编码
        print("🔧 字符串特征编码...")
        
        # 纤维类型编码 - 支持多种纤维类型
        if 'Glass_or_Basalt' in X_clean.columns:
            fiber_series = X_clean['Glass_or_Basalt'].astype(str).str.lower()
            
            # 创建数值编码：Glass=0, Basalt=1, Carbon=2, 其他=3
            def encode_fiber_type(x):
                if 'glass' in x:
                    return 0
                elif 'basalt' in x:
                    return 1
                elif 'carbon' in x:
                    return 2
                else:  # 其他类型
                    return 3
            
            X_clean['Glass_or_Basalt'] = fiber_series.apply(encode_fiber_type)
            
            # 打印纤维类型分布
            fiber_counts = X_clean['Glass_or_Basalt'].value_counts().sort_index()
            type_names = {0: 'Glass', 1: 'Basalt', 2: 'Carbon', 3: '其他'}
            print(f"  纤维类型编码完成:")
            for code, count in fiber_counts.items():
                print(f"    {type_names.get(code, f'类型{code}')}: {count} 样本")
        
        # 树脂类型编码
        if 'Vinyl_ester_or_Epoxy' in X_clean.columns:
            resin_series = X_clean['Vinyl_ester_or_Epoxy'].astype(str).str.lower() 
            X_clean['Vinyl_ester_or_Epoxy'] = resin_series.apply(lambda x: 1 if 'vinyl' in x else 0)
            print(f"  树脂类型编码完成")
        
        # diameter开根号变换
        if 'diameter' in X_clean.columns:
            X_clean['diameter'] = np.sqrt(X_clean['diameter'])
            print(f"  直径开根号变换完成")
        
        # 最终清理 - 检查并报告NaN情况
        print(f"清理前样本数: {len(X_clean)}")
        print(f"各列NaN数量:")
        for col in X_clean.columns:
            na_count = X_clean[col].isnull().sum()
            if na_count > 0:
                print(f"  {col}: {na_count}")
        
        y_na_count = y_clean.isnull().sum()
        print(f"目标变量NaN数量: {y_na_count}")
        
        # 只移除目标变量为NaN的行
        y_na_mask = y_clean.isnull()
        if y_na_mask.sum() > 0:
            print(f"移除目标变量NaN的行: {y_na_mask.sum()}")
            X_clean = X_clean[~y_na_mask]
            y_clean = y_clean[~y_na_mask]
        
        # 对于特征中的NaN，已经在前面填充了，这里再次检查
        remaining_na = X_clean.isnull().sum().sum()
        if remaining_na > 0:
            print(f"⚠️ 仍有{remaining_na}个特征NaN值，将用中位数填充")
            for col in X_clean.columns:
                if X_clean[col].isnull().sum() > 0:
                    if X_clean[col].dtype == 'object':
                        # 字符串列用众数填充
                        mode_val = X_clean[col].mode()[0] if len(X_clean[col].mode()) > 0 else 'Glass'
                        X_clean[col] = X_clean[col].fillna(mode_val)
                    else:
                        # 数值列用中位数填充
                        median_val = X_clean[col].median()
                        X_clean[col] = X_clean[col].fillna(median_val)
        
        print(f"✅ 特征提取完成: {len(X_clean)} 样本, {X_clean.shape[1]} 特征")
        
        # 分别处理数值和字符串列
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        string_cols = X_clean.select_dtypes(include=[object]).columns
        
        if len(numeric_cols) > 0:
            print(f"数值特征范围: [{X_clean[numeric_cols].min().min():.3f}, {X_clean[numeric_cols].max().max():.3f}]")
        
        if len(string_cols) > 0:
            print(f"字符串特征: {list(string_cols)}")
        
        print(f"目标范围: [{y_clean.min():.3f}, {y_clean.max():.3f}]")
        
        return X_clean, y_clean

def get_mlp_configs():
    """获取30个MLP参数配置 (简化版)"""
    configs = []
    
    # 定义参数组合
    hidden_layers = [
        (64,), (128,), (256,),
        (64, 32), (128, 64), (256, 128),
        (64, 32, 16), (128, 64, 32), (256, 128, 64),
        (128, 128), (256, 256)
    ]
    
    activations = ['relu', 'tanh', 'logistic']
    alphas = [0.0001, 0.001, 0.01]
    learning_rates = [0.001, 0.01]
    
    config_id = 0
    # 生成配置组合
    for hidden in hidden_layers[:10]:  # 限制为10种网络结构
        for activation in activations:
            if config_id >= 30:  # 限制总数为30
                break
            config_id += 1
            
            alpha = alphas[config_id % len(alphas)]
            lr = learning_rates[config_id % len(learning_rates)]
            
            config = {
                'hidden_layer_sizes': hidden,
                'activation': activation,
                'solver': 'adam',
                'alpha': alpha,
                'learning_rate_init': lr,
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
            
            configs.append({
                'model': 'MLP_Sklearn',
                'config_id': config_id,
                'config': config
            })
        
        if config_id >= 30:
            break
    
    return configs

def train_evaluate_mlp(config_info, X, y):
    """训练和评估MLP模型 - 简化版"""
    config = config_info['config']
    config_id = config_info['config_id']
    
    try:
        print(f"   训练配置 #{config_id}...")
        
        # 最终检查和清理NaN
        from sklearn.impute import SimpleImputer
        
        # 首先检查每列的NaN情况
        nan_counts = X.isnull().sum()
        total_rows = len(X)
        
        # 删除完全为NaN的列
        all_nan_cols = nan_counts[nan_counts == total_rows].index.tolist()
        if all_nan_cols:
            print(f"   删除完全为NaN的列: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols)
        
        # 确保没有NaN
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            print(f"   检测到NaN值，进行最终清理...")
            
            # 对特征使用均值填充
            imputer = SimpleImputer(strategy='mean')
            X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
            
            # 对目标变量移除NaN行
            valid_mask = y.notna()
            X_clean = X_clean[valid_mask]
            y_clean = y[valid_mask]
        else:
            X_clean = X
            y_clean = y
        
        if len(X_clean) < 10:
            print(f"   ❌ 样本数太少: {len(X_clean)}")
            return None
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 创建模型
        model = MLPRegressor(**config)
        
        # 交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='r2')
        
        # 训练最终模型
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 计算指标
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        result = {
            'model': 'MLP_Sklearn',
            'config_id': config_id,
            'config': str(config),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'training_time': training_time,
            'n_samples': len(X_clean),
            'n_features': X_clean.shape[1]
        }
        
        print(f"   ✅ R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
        return result
        
    except Exception as e:
        print(f"   ❌ 失败: {str(e)}")
        return None

def save_results(results, experiment_id):
    """保存实验结果"""
    result_dir = Path("experiments")
    result_dir.mkdir(exist_ok=True)
    
    # CSV文件
    df = pd.DataFrame(results)
    csv_path = result_dir / f"mlp_simple_{experiment_id}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # JSON文件
    json_path = result_dir / f"mlp_simple_{experiment_id}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(results),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 结果已保存: {csv_path}")

def main():
    print("🚀 MLP神经网络实验 (简化版)")
    print("=" * 50)
    
    start_time = time.time()
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 数据加载
    print("📂 加载数据...")
    loader = ValidDataLoader()
    data = loader.load_valid_data()
    
    if data is None or len(data) == 0:
        print("❌ 无有效数据")
        return
    
    # 2. 特征提取
    print("\n🔧 特征提取...")
    result = loader.prepare_features_target(data)
    
    if result is None:
        print("❌ 特征提取失败")
        return
    
    X, y = result
    
    if len(X) < 10:
        print("❌ 样本数太少")
        return
    
    print(f"✅ 数据准备完成: {len(X)} 样本, {X.shape[1]} 特征")
    
    # 3. 模型训练
    print("\n🔬 开始模型训练...")
    configs = get_mlp_configs()
    print(f"总配置数: {len(configs)}")
    
    results = []
    for i, config_info in enumerate(configs):
        print(f"🔄 进度: {i+1}/{len(configs)}")
        result = train_evaluate_mlp(config_info, X, y)
        if result:
            results.append(result)
    
    # 4. 结果分析
    total_time = time.time() - start_time
    print(f"\n🎉 实验完成!")
    print(f"总用时: {total_time:.1f}秒")
    print(f"成功配置: {len(results)}/{len(configs)}")
    
    if results:
        save_results(results, experiment_id)
        
        df_results = pd.DataFrame(results)
        best_config = df_results.loc[df_results['test_r2'].idxmax()]
        
        print(f"\n🏆 最佳配置:")
        print(f"  配置ID: {best_config['config_id']}")
        print(f"  测试R²: {best_config['test_r2']:.4f}")
        print(f"  测试RMSE: {best_config['test_rmse']:.4f}")
        print(f"  交叉验证: {best_config['cv_mean']:.4f}±{best_config['cv_std']:.4f}")
        
        r2_values = df_results['test_r2']
        print(f"\n📈 R²分布:")
        print(f"  平均: {r2_values.mean():.4f}")
        print(f"  最大: {r2_values.max():.4f}")
        print(f"  最小: {r2_values.min():.4f}")
        
    else:
        print("❌ 没有成功的配置")

if __name__ == "__main__":
    main()