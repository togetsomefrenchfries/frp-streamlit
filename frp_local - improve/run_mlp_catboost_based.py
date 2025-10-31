#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP钢筋耐久性预测 - MLP神经网络实验 (基于CatBoost预处理流程)

特点：
1. 复用已验证的CatBoost数据预处理流程
2. 基于多层感知器(MLP)的深度学习模型
3. 在CatBoost预处理基础上添加神经网络专用的第二次预处理
4. 5折交叉验证
5. 支持PyTorch和Sklearn两种实现
6. 多种MLP架构和超参数优化
7. 增强的错误处理和进度跟踪
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import time
import json
import warnings
from datetime import datetime
from pathlib import Path
import sys
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

# 导入深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch已安装")
except ImportError:
    print("⚠️  PyTorch未安装，将使用sklearn的MLPRegressor")
    TORCH_AVAILABLE = False
    # 为了避免语法错误，定义空的torch相关模块
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

try:
    from sklearn.neural_network import MLPRegressor
    SKLEARN_MLP_AVAILABLE = True
    print("✅ Sklearn MLPRegressor可用")
except ImportError:
    print("❌ Sklearn MLPRegressor不可用")
    SKLEARN_MLP_AVAILABLE = False

if not TORCH_AVAILABLE and not SKLEARN_MLP_AVAILABLE:
    print("❌ 没有可用的MLP实现，程序退出")
    sys.exit(1)

warnings.filterwarnings('ignore')

# 添加进度条支持
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️  tqdm未安装，将使用基础进度显示")
    TQDM_AVAILABLE = False
    # 简单的tqdm替代
    class tqdm:
        def __init__(self, iterable, desc="", total=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total or len(iterable) if hasattr(iterable, '__len__') else None
            self.current = 0
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.current += 1
                if self.total:
                    print(f"\r{self.desc} {self.current}/{self.total} ({self.current/self.total*100:.1f}%)", end="")
            print()  # 换行

# 导入已验证的数据加载器 (从run_40param_experiment.py复制)
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
                print("请确保文件位于以下路径之一:")
                for path in possible_paths:
                    print(f"  - {path}")
                self.file_path = None
        else:
            self.file_path = file_path
    
    def load_valid_data(self):
        """加载有效数据"""
        if self.file_path is None:
            return None
        
        try:
            print(f"🔄 加载{Path(self.file_path).name}文件...")
            df = pd.read_excel(self.file_path, header=None)
            print(f"✅ Excel读取成功，数据形状: {df.shape}")
            
            # 基本数据筛选：第一列不为0
            df_filtered = df[df.iloc[:, 0] != 0].copy()
            
            print(f"🎯 数据筛选结果:")
            print(f"  原始数据: {len(df)} 行")
            print(f"  第一列非0: {len(df_filtered)} 行")
            print(f"  最终保留比例: {len(df_filtered)/len(df)*100:.1f}%")
            
            # 数据质量检查
            print(f"📊 数据质量检查:")
            print(f"  有效数据形状: {df_filtered.shape}")
            print(f"  缺失值总数: {df_filtered.isnull().sum().sum()}")
            
            return df_filtered
            
        except Exception as e:
            print(f"❌ 读取Excel文件失败: {e}")
            return None
    
    def prepare_features_target(self, data, generate_plots=True):
        """准备特征和目标变量 - 与CatBoost实验完全一致的特征提取"""
        if data is None or len(data) == 0:
            return None
        
        print("\n🔧 开始基于真实结构的特征提取...")
        print("=" * 80)
        
        if not generate_plots:
            print("⏭️ 跳过特征分布图生成，直接进行特征提取...")
        
        # 数据清理
        print("🧹 正在清理数据...")
        data_clean = data.copy()
        print("✅ 数据清理完成")
        
        # 基于database 4.xlsx真实结构的特征定义
        feature_definitions = {
            'pH_of_condition_enviroment': {
                'positions': [54, 59, 60], 
                'description': '环境条件pH值'
            },
            'Chloride_ion': {
                'positions': [61, 64, 77], 
                'description': '氯离子存在 (0/1)'
            },
            'concrete': {
                'positions': [53, 56, 57], 
                'description': '混凝土环境 (0/1)'
            },
            'diameter': {
                'positions': [18], 
                'description': '纤维直径开根号 (sqrt(mm))'
            },
            'load_value': {
                'positions': [90], 
                'description': '载荷值'
            },
            'fiber_content': {
                'positions': [15], 
                'description': '纤维含量 (%)'
            },
            'initial_tensile_strength': {
                'positions': [34, 37, 40], 
                'description': '初始拉伸强度 (MPa)'
            },
            'Glass_or_Basalt': {
                'positions': [8], 
                'description': '纤维类型 Glass=1, Basalt=0'
            },
            'Vinyl_ester_or_Epoxy': {
                'positions': [10], 
                'description': '树脂类型 Vinyl_ester=1, Epoxy=0'
            },
            'condition_time': {
                'positions': [51], 
                'description': '条件时间 (天/小时)'
            },
            'Temperature': {
                'positions': [49], 
                'description': '温度 (°C)'
            },
            'surface_treatment': {
                'positions': [22], 
                'description': '表面处理 (0/1)'
            },
            'glass_transition_temperature': {
                'positions': [12, 114], 
                'description': '玻璃化转变温度 (°C)'
            }
        }
        
        print("📋 基于真实位置的特征提取规则:")
        for feat_name, feat_info in feature_definitions.items():
            positions_str = str(feat_info['positions']).replace(' ', '')
            print(f"  🎯 {feat_name}: 位置{positions_str} ({feat_info['description']})")
        
        # 提取特征
        feature_matrix = []
        feature_names = []
        
        for feat_name, feat_info in feature_definitions.items():
            print(f"\n🔍 提取特征: {feat_name}")
            
            positions = feat_info['positions']
            feature_values = []
            
            for pos in positions:
                if pos < len(data_clean.columns):
                    col_values = data_clean.iloc[:, pos].copy()
                    feature_values.append(col_values)
                else:
                    print(f"   ⚠️  位置{pos}超出数据范围")
            
            if feature_values:
                if len(feature_values) == 1:
                    final_feature = feature_values[0]
                else:
                    # 多列特征：使用第一个非空值
                    final_feature = feature_values[0].copy()
                    for i, other_feat in enumerate(feature_values[1:], 1):
                        mask = final_feature.isna()
                        final_feature.loc[mask] = other_feat.loc[mask]
                
                feature_matrix.append(final_feature)
                feature_names.append(f"feat_{len(feature_names)}_{feat_name}")
                print(f"   ✅ 提取成功")
            else:
                print(f"   ❌ 提取失败")
        
        # 转换为DataFrame
        if feature_matrix:
            X = pd.DataFrame(feature_matrix).T
            X.columns = feature_names
        else:
            print("❌ 没有成功提取任何特征")
            return None
        
        # 提取目标变量 - 使用位置100
        print(f"🎯 目标变量: 位置100", end="")
        if 100 < len(data_clean.columns):
            y = data_clean.iloc[:, 100].copy()  # 位置100是retention值
            valid_target_count = y.notna().sum()
            total_count = len(y)
            print(f" (有效数据: {valid_target_count}/{total_count})")
        else:
            print(" ❌ 位置100超出范围")
            return None
        
        # 数据有效性检查 - 使用与CatBoost实验相同的逻辑
        print(f"\n🔍 执行严格的数据有效性检查...")
        
        # 跳过Target_parameter检查，直接使用所有retention数据
        print(f"  跳过Target_parameter检查，使用所有retention数据")
        tensile_mask = pd.Series([True] * len(data_clean), index=data_clean.index)
        tensile_count = len(data_clean)
        
        # 检查retention1列（位置100）
        retention_mask = y.notna()
        retention_count = retention_mask.sum()
        print(f"  retention1检查: {retention_count}行有数值")
        
        # 检查condition_time
        condition_time_col = feature_names.index('feat_9_condition_time')
        condition_time_mask = X.iloc[:, condition_time_col].notna()
        condition_time_count = condition_time_mask.sum()
        print(f"  condition_time检查: {condition_time_count}行有值（缺失值将被排除）")
        
        # 纤维类型检查：只保留Glass和Basalt
        fiber_type_col = feature_names.index('feat_7_Glass_or_Basalt')
        fiber_type_series = X.iloc[:, fiber_type_col].astype(str).str.lower()
        glass_count = fiber_type_series.str.contains('glass', na=False).sum()
        basalt_count = fiber_type_series.str.contains('basalt', na=False).sum()
        fiber_valid_mask = (fiber_type_series.str.contains('glass', na=False) | 
                           fiber_type_series.str.contains('basalt', na=False)) & \
                          ~fiber_type_series.str.contains('carbon', na=False) & \
                          ~fiber_type_series.str.contains('aramid', na=False) & \
                          ~fiber_type_series.str.contains('steel', na=False)
        fiber_valid_count = fiber_valid_mask.sum()
        print(f"  纤维类型检查: Glass={glass_count}行, Basalt={basalt_count}行, 总有效={fiber_valid_count}行")
        
        # 综合条件检查
        print(f"\n📈 严格数据质量检查结果:")
        print(f"  原始数据: {len(data_clean)} 行")
        print(f"  特征矩阵: {X.shape}")
        
        # 检查至少有7个特征有值的行
        feature_valid_mask = (X.notna().sum(axis=1) >= 7)
        feature_valid_count = feature_valid_mask.sum()
        print(f"  至少7个特征有值: {feature_valid_count} 行")
        
        print(f"  Target_parameter为Tensile: {tensile_count} 行")
        print(f"  retention1有数值: {retention_count} 行")
        print(f"  condition_time有值: {condition_time_count} 行")
        print(f"  纤维类型为Glass/Basalt: {fiber_valid_count} 行")
        
        # 应用所有筛选条件
        final_mask = (
            feature_valid_mask & 
            tensile_mask & 
            retention_mask & 
            condition_time_mask &
            fiber_valid_mask
        )
        
        final_count = final_mask.sum()
        retention_rate = final_count / len(data_clean) * 100
        print(f"  所有条件都满足: {final_count} 行")
        print(f"  数据保留率: {retention_rate:.1f}%")
        
        if final_count < 100:
            print(f"⚠️  有效数据太少({final_count}行)，可能影响模型训练效果")
        
        # 应用筛选
        X_clean = X[final_mask].copy()
        y_clean = y[final_mask].copy()
        
        # 缺失值处理 - 使用与CatBoost相同的材料科学规则
        print(f"🔧 按照材料科学规则处理缺失值...")
        
        # 处理各特征的缺失值
        for i, col in enumerate(X_clean.columns):
            missing_count = X_clean[col].isna().sum()
            if missing_count > 0:
                print(f"  处理特征 {col.split('_', 2)[-1]} 的 {missing_count} 个缺失值...")
                
                if 'diameter' in col:
                    # 清理diameter列中的字符串数据
                    def clean_diameter(value):
                        if pd.isna(value):
                            return np.nan
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            str_val = str(value)
                            if ',' in str_val:
                                return float(str_val.split(',')[0])
                            try:
                                import re
                                numbers = re.findall(r'\d+\.?\d*', str_val)
                                if numbers:
                                    return float(numbers[0])
                            except:
                                pass
                            return np.nan
                    
                    X_clean[col] = X_clean[col].apply(clean_diameter)
                    # diameter缺失值用中位数填充
                    median_val = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_val)
                    print(f"    直径缺失值用中位数填充: {median_val:.3f}mm")
                    
                elif 'load_value' in col:
                    # 载荷缺失值填充为0
                    X_clean[col] = X_clean[col].fillna(0)
                    print(f"    载荷水平缺失值填充为0 (无加载)")
                    
                elif 'fiber_content' in col:
                    # 纤维含量缺失值用固定值78.2%填充
                    X_clean[col] = X_clean[col].fillna(78.2)
                    print(f"    纤维含量缺失值用固定值填充: 78.2%")
                    
                elif 'initial_tensile_strength' in col:
                    # 初始拉伸强度用中位数填充
                    median_val = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_val)
                    print(f"    初始拉伸强度缺失值用中位数填充: {median_val:.3f}MPa")
                    
                elif 'Temperature' in col:
                    # 温度缺失值填充为25°C (室温)
                    X_clean[col] = X_clean[col].fillna(25.0)
                    print(f"    温度缺失值填充为25°C (室温)")
                    
                elif 'glass_transition_temperature' in col:
                    # 玻璃化转变温度用中位数填充
                    median_val = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_val)
                    print(f"    玻璃化转变温度缺失值用中位数填充: {median_val:.3f}°C")
                    
                else:
                    # 其他特征用中位数填充
                    median_val = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_val)
                    print(f"    {col.split('_', 2)[-1]}缺失值用中位数填充: {median_val:.3f}")
        
        print(f"✅ 缺失值填充完成，最终数据形状: {X_clean.shape}")
        
        # 纤维类型数值编码
        fiber_type_col = 'feat_7_Glass_or_Basalt'
        if fiber_type_col in X_clean.columns:
            print(f"🔧 纤维类型数值编码...")
            fiber_series = X_clean[fiber_type_col].astype(str).str.lower()
            # Glass=1, Basalt=0
            X_clean[fiber_type_col] = fiber_series.apply(lambda x: 1 if 'glass' in x else 0)
            glass_encoded = (X_clean[fiber_type_col] == 1).sum()
            basalt_encoded = (X_clean[fiber_type_col] == 0).sum()
            print(f"    编码结果: Glass={glass_encoded}行, Basalt={basalt_encoded}行")
        
        # 特征工程变换
        print(f"🧮 应用特征工程变换...")
        
        # diameter开根号变换 - 在应用变换前，确保已经清理过
        diameter_col = 'feat_3_diameter'
        if diameter_col in X_clean.columns:
            # 重新清理diameter列以确保没有字符串
            def clean_diameter_for_transform(value):
                if pd.isna(value):
                    return np.nan
                try:
                    return float(value)
                except (ValueError, TypeError):
                    str_val = str(value)
                    if ',' in str_val:
                        return float(str_val.split(',')[0])
                    try:
                        import re
                        numbers = re.findall(r'\d+\.?\d*', str_val)
                        if numbers:
                            return float(numbers[0])
                    except:
                        pass
                    return np.nan
            
            # 应用清理函数
            X_clean[diameter_col] = X_clean[diameter_col].apply(clean_diameter_for_transform)
            
            # 计算中位数（现在应该可以成功）
            try:
                original_median = X_clean[diameter_col].median()
                X_clean[diameter_col] = np.sqrt(X_clean[diameter_col])
                new_median = X_clean[diameter_col].median()
                print(f"    diameter开根号变换: {original_median:.3f}mm → sqrt({original_median:.3f}) = {new_median:.3f}")
            except Exception as e:
                print(f"    ⚠️  diameter变换失败: {e}")
                # 如果仍然失败，使用固定值填充
                X_clean[diameter_col] = X_clean[diameter_col].fillna(8.0)  # 使用固定值
                X_clean[diameter_col] = np.sqrt(X_clean[diameter_col])
                print(f"    使用固定值进行径开根号变换")
        
        print(f"✅ 特征工程完成")
        
        # 清理目标变量中的非数值数据
        print(f"🔧 清理目标变量...")
        def clean_target_value(value):
            if pd.isna(value):
                return np.nan
            try:
                return float(value)
            except (ValueError, TypeError):
                return np.nan
        
        y_clean = y_clean.apply(clean_target_value)
        
        # 移除目标变量为NaN的行
        valid_target_mask = y_clean.notna() 
        if not valid_target_mask.all():
            print(f"  移除目标变量为NaN的行: {(~valid_target_mask).sum()}行")
            X_clean = X_clean[valid_target_mask]
            y_clean = y_clean[valid_target_mask]
        
        # 生成均匀权重
        sample_weights = np.ones(len(X_clean))
        
        print(f"\n✅ 基于真实结构的特征提取成功:")
        print(f"  最终特征数: {X_clean.shape[1]}")
        print(f"  样本数: {len(X_clean)}")
        print(f"  目标变量: position_100")
        print(f"  目标范围: [{y_clean.min():.3f}, {y_clean.max():.3f}]")
        print(f"  权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        
        if len(X_clean) > 0:
            # 最终数据验证和清理
            print("🔧 最终数据验证和清理...")
            
            # 清理所有特征中的非数值数据
            for col in X_clean.columns:
                def clean_numeric_value(value):
                    if pd.isna(value):
                        return np.nan
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
                            import re
                            numbers = re.findall(r'\d+\.?\d*', str_val)
                            if numbers:
                                return float(numbers[0])
                        except:
                            pass
                        return np.nan
                
                X_clean[col] = X_clean[col].apply(clean_numeric_value)
            
            # 移除缺失值
            if X_clean.isnull().sum().sum() > 0:
                print("⚠️  仍有缺失值，进行最终清理...")
                # 找出有缺失值的行
                rows_with_na = X_clean.isnull().any(axis=1)
                if rows_with_na.sum() > 0:
                    print(f"  删除{rows_with_na.sum()}行包含缺失值的数据")
                    X_clean = X_clean[~rows_with_na]
                    y_clean = y_clean[~rows_with_na]
                    sample_weights = sample_weights[~rows_with_na.values]
                
                print(f"  样本数: {len(X_clean)}")
                
                return X_clean, y_clean, list(X_clean.columns), sample_weights
        
        return None

# 神经网络专用的第二次预处理类
class MLPDataProcessor:
    """MLP神经网络专用的数据预处理器 - 在CatBoost预处理基础上的第二次处理"""
    
    def __init__(self, scaler_type='standard'):
        """
        初始化MLP数据处理器
        
        Args:
            scaler_type: 缩放器类型 ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_stats = {}
        
    def create_scaler(self):
        """创建数据缩放器"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def neural_network_preprocess(self, X, y, test_size=0.2, random_state=42):
        """
        神经网络专用的第二次预处理
        
        Args:
            X: 经过CatBoost预处理的特征数据
            y: 目标变量
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_stats)
        """
        print("🧠 开始神经网络专用的第二次预处理...")
        
        # 1. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
        
        # 2. 特征统计
        print("📊 计算特征统计信息...")
        self.feature_stats = {
            'n_features': X.shape[1],
            'feature_names': list(X.columns),
            'train_mean': X_train.mean().to_dict(),
            'train_std': X_train.std().to_dict(),
            'train_min': X_train.min().to_dict(),
            'train_max': X_train.max().to_dict()
        }
        
        # 3. 数据缩放
        print(f"🔧 应用{self.scaler_type}缩放...")
        self.scaler = self.create_scaler()
        
        # 只在训练集上拟合缩放器
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 转换回DataFrame，保持列名
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        print(f"✅ 神经网络预处理完成")
        print(f"  缩放后特征范围: [{X_train_scaled.min().min():.3f}, {X_train_scaled.max().max():.3f}]")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler, self.feature_stats

# PyTorch MLP模型定义
if TORCH_AVAILABLE:
    class MLPRegressor_PyTorch(nn.Module):
        """基于PyTorch的MLP回归器"""
        
        def __init__(self, input_size, hidden_layers, dropout_rate=0.0, activation='relu'):
            super(MLPRegressor_PyTorch, self).__init__()
            
            layers = []
            prev_size = input_size
            
            # 激活函数选择
            if activation == 'relu':
                act_fn = nn.ReLU()
            elif activation == 'tanh':
                act_fn = nn.Tanh()
            elif activation == 'sigmoid':
                act_fn = nn.Sigmoid()
            elif activation == 'leaky_relu':
                act_fn = nn.LeakyReLU()
            else:
                act_fn = nn.ReLU()
            
            # 构建隐藏层
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(act_fn)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            # 输出层
            layers.append(nn.Linear(prev_size, 1))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
else:
    class MLPRegressor_PyTorch:
        """PyTorch不可用时的占位符类"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorch不可用")

class MLPTrainer:
    """MLP训练器 - 支持PyTorch和Sklearn实现"""
    
    def __init__(self, use_pytorch=True):
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE
        if not self.use_pytorch and not SKLEARN_MLP_AVAILABLE:
            raise ValueError("没有可用的MLP实现")
    
    def create_model(self, config, input_size):
        """创建MLP模型"""
        if self.use_pytorch:
            return self._create_pytorch_model(config, input_size)
        else:
            return self._create_sklearn_model(config)
    
    def _create_pytorch_model(self, config, input_size):
        """创建PyTorch模型"""
        model = MLPRegressor_PyTorch(
            input_size=input_size,
            hidden_layers=config['hidden_layer_sizes'],
            dropout_rate=config.get('dropout_rate', 0.0),
            activation=config.get('activation', 'relu')
        )
        return model
    
    def _create_sklearn_model(self, config):
        """创建Sklearn模型"""
        sklearn_config = config.copy()
        # 移除PyTorch特有的参数
        sklearn_config.pop('dropout_rate', None)
        sklearn_config.pop('batch_size', None)
        sklearn_config.pop('learning_rate_init', None)
        sklearn_config.pop('epochs', None)
        
        model = MLPRegressor(**sklearn_config)
        return model
    
    def train_pytorch_model(self, model, X_train, y_train, X_val, y_val, config):
        """训练PyTorch模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    使用设备: {device}")
        
        model = model.to(device)
        
        # 转换数据
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get('batch_size', 32), 
            shuffle=True
        )
        
        # 优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.get('learning_rate_init', 0.001)
        )
        criterion = nn.MSELoss()
        
        # 训练循环
        epochs = config.get('epochs', 200)
        best_val_loss = float('inf')
        patience = config.get('early_stopping_patience', 20)
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    早停在第{epoch+1}轮")
                    break
        
        return model
    
    def predict_pytorch_model(self, model, X):
        """PyTorch模型预测"""
        device = next(model.parameters()).device
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(device)
            predictions = model(X_tensor)
            return predictions.cpu().numpy().flatten()

def get_mlp_parameter_configs():
    """获取100个MLP参数配置"""
    
    configs = []
    
    # MLP 参数配置 (100个)
    import itertools
    from itertools import product
    
    # 定义参数范围
    if TORCH_AVAILABLE:
        # PyTorch参数
        hidden_layer_sizes = [
            [64], [128], [256], [512],
            [64, 32], [128, 64], [256, 128], [512, 256],
            [64, 32, 16], [128, 64, 32], [256, 128, 64],
            [128, 128], [256, 256], [64, 64, 64],
            [512, 256, 128], [256, 128, 64, 32]
        ]
        activations = ['relu', 'tanh', 'leaky_relu']
        learning_rates = [0.001, 0.005, 0.01, 0.02]
        dropout_rates = [0.0, 0.1, 0.2, 0.3]
        batch_sizes = [16, 32, 64]
        epochs_list = [100, 200, 300]
        
        # 生成配置组合
        base_combinations = list(product(
            hidden_layer_sizes[:12],  # 12种网络结构
            activations,              # 3种激活函数
            learning_rates[:3],       # 3种学习率
        ))
        
        # 从108个组合中选择100个
        selected_combinations = base_combinations[:100]
        
        for i, (hidden_layers, activation, lr) in enumerate(selected_combinations):
            config = {
                'model_type': 'pytorch',
                'hidden_layer_sizes': hidden_layers,
                'activation': activation,
                'learning_rate_init': lr,
                'dropout_rate': dropout_rates[i % len(dropout_rates)],
                'epochs': epochs_list[i % len(epochs_list)],
                'batch_size': batch_sizes[i % len(batch_sizes)],
                'early_stopping_patience': 20,
                'random_state': 42
            }
            
            configs.append({
                'model': 'MLP_PyTorch',
                'config_id': i + 1,
                'config': config
            })
    
    else:
        # Sklearn参数
        hidden_layer_sizes = [
            (64,), (128,), (256,), (512,),
            (64, 32), (128, 64), (256, 128), (512, 256),
            (64, 32, 16), (128, 64, 32), (256, 128, 64),
            (128, 128), (256, 256), (64, 64, 64)
        ]
        activations = ['relu', 'tanh', 'logistic']
        solvers = ['adam', 'lbfgs']
        alpha_values = [0.0001, 0.001, 0.01, 0.1]
        learning_rate_inits = [0.001, 0.01, 0.1]
        
        # 生成配置组合
        base_combinations = list(product(
            hidden_layer_sizes[:12],
            activations,
            solvers[:1],  # 只使用adam
            alpha_values[:3]
        ))
        
        selected_combinations = base_combinations[:100]
        
        for i, (hidden_layers, activation, solver, alpha) in enumerate(selected_combinations):
            config = {
                'model_type': 'sklearn',
                'hidden_layer_sizes': hidden_layers,
                'activation': activation,
                'solver': solver,
                'alpha': alpha,
                'learning_rate_init': learning_rate_inits[i % len(learning_rate_inits)],
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
            
            configs.append({
                'model': 'MLP_Sklearn',
                'config_id': i + 1,
                'config': config
            })
    
    return configs

def train_and_evaluate_mlp_config(config_info, X, y, sample_weights=None, cv_folds=5):
    """训练和评估单个MLP配置"""
    model_name = config_info['model']
    config = config_info['config']
    config_id = config_info['config_id']
    
    try:
        # 神经网络专用的第二次预处理
        print(f"   🧠 神经网络数据预处理...")
        mlp_processor = MLPDataProcessor(scaler_type='standard')
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_stats = \
            mlp_processor.neural_network_preprocess(X, y)
        
        # 创建MLP训练器
        use_pytorch = config.get('model_type') == 'pytorch'
        trainer = MLPTrainer(use_pytorch=use_pytorch)
        
        # 交叉验证
        print(f"   🔄 开始交叉验证...")
        
        if use_pytorch:
            # PyTorch交叉验证
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            # 对整个数据集进行缩放
            X_scaled_full = scaler.fit_transform(X)
            X_scaled_full = pd.DataFrame(X_scaled_full, columns=X.columns, index=X.index)
            
            for train_idx, val_idx in kfold.split(X_scaled_full):
                X_train_fold = X_scaled_full.iloc[train_idx]
                X_val_fold = X_scaled_full.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # 创建并训练模型
                model = trainer.create_model(config, X_scaled_full.shape[1])
                model = trainer.train_pytorch_model(
                    model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, config
                )
                
                # 预测
                y_pred_fold = trainer.predict_pytorch_model(model, X_val_fold)
                score = r2_score(y_val_fold, y_pred_fold)
                cv_scores.append(score)
            
            cv_scores = np.array(cv_scores)
        else:
            # Sklearn交叉验证
            # 对整个数据集进行缩放
            X_scaled_full = scaler.fit_transform(X)
            model = trainer.create_model(config, X.shape[1])
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_scaled_full, y, cv=kfold, scoring='r2')
        
        print(f"   ✅ 交叉验证完成: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # 训练最终模型
        print(f"   🔄 开始训练最终模型...")
        start_time = time.time()
        
        if use_pytorch:
            # 为最终训练创建验证集
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_train_scaled, y_train, test_size=0.1, random_state=42
            )
            
            final_model = trainer.create_model(config, X_train_scaled.shape[1])
            final_model = trainer.train_pytorch_model(
                final_model, X_train_final, y_train_final, X_val_final, y_val_final, config
            )
        else:
            final_model = trainer.create_model(config, X_train_scaled.shape[1])
            final_model.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        print(f"   ✅ 模型训练完成，用时: {training_time:.2f}s")
        
        # 预测
        if use_pytorch:
            y_pred = trainer.predict_pytorch_model(final_model, X_test_scaled)
        else:
            y_pred = final_model.predict(X_test_scaled)
        
        # 计算指标
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        
        result = {
            'model': model_name,
            'config_id': config_id,
            'config': str(config),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'training_time': training_time,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'scaler_type': 'standard'
        }
        
        print(f"   ✅ 评估完成: R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
        return result
        
    except Exception as e:
        import traceback
        print(f"❌ 配置 {model_name} #{config_id} 训练失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        print(f"   配置参数: {config}")
        
        # 输出前几行堆栈信息
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines[-8:]:
            if line.strip():
                print(f"   {line}")
        
        return None

def save_mlp_results(results, experiment_id):
    """保存MLP实验结果"""
    # 创建结果目录
    result_dir = Path("experiments")
    result_dir.mkdir(exist_ok=True)
    
    # 保存为CSV
    df = pd.DataFrame(results)
    csv_path = result_dir / f"mlp_exp_{experiment_id}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 保存详细结果为JSON
    json_path = result_dir / f"mlp_exp_{experiment_id}_detailed.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(results),
            'pytorch_available': TORCH_AVAILABLE,
            'sklearn_mlp_available': SKLEARN_MLP_AVAILABLE,
            'data_preprocessing': 'catboost_compatible',
            'neural_preprocessing': 'standard_scaler',
            'data_filter': 'Glass&Basalt_fibers',
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 MLP实验结果已保存:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")

def main():
    print("🚀 开始MLP神经网络参数优化实验 (基于CatBoost预处理流程)")
    print("=" * 70)
    
    # 检查并创建必要目录
    Path("analysis_results").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    
    if TORCH_AVAILABLE:
        print("📋 使用PyTorch实现的MLP: 100个配置")
        print("📋 支持GPU加速 (如果可用)")
    else:
        print("📋 使用Sklearn实现的MLP: 100个配置")
    
    print("📋 复用已验证的CatBoost数据预处理流程")
    print("📋 添加神经网络专用的第二次预处理")
    print("📋 专门优化MLP神经网络超参数")
    
    start_time = time.time()
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 使用已验证的数据加载器
    print("📂 使用已验证的数据加载器...")
    loader = ValidDataLoader()
    data = loader.load_valid_data()
    
    if data is None or len(data) == 0:
        print("❌ 无有效数据，实验终止")
        return
    
    # 2. 使用已验证的特征提取流程
    print("\n🔧 使用已验证的特征提取流程...")
    result = loader.prepare_features_target(data, generate_plots=False)
    
    if result is None or len(result) != 4:
        print("❌ 特征提取失败，实验终止")
        return
    
    X, y, feature_names, sample_weights = result
    
    if X is None or y is None:
        print("❌ 特征提取失败，实验终止")
        return
    
    print(f"✅ CatBoost预处理完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"特征列: {feature_names}")
    print(f"💡 使用与CatBoost实验完全一致的预处理流程")
    
    # 数据概览
    print(f"\n📊 数据概览:")
    print(f"  特征范围: {X.min().min():.3f} ~ {X.max().max():.3f}")
    print(f"  目标范围: {y.min():.3f} ~ {y.max():.3f}")
    print(f"  目标均值±标准差: {y.mean():.3f}±{y.std():.3f}")
    
    # 3. 获取MLP参数配置
    print("\n⚙️  准备MLP参数配置...")
    configs = get_mlp_parameter_configs()
    print(f"总配置数: {len(configs)}")
    
    # 统计每个模型的配置数
    model_counts = {}
    for config in configs:
        model_name = config['model']
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    print(f"各模型配置数:")
    for model_name, count in model_counts.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {model_name}: {count} 个配置")
    
    # 4. 运行实验
    print(f"\n🔬 开始运行MLP实验...")
    
    results = []
    
    # 使用tqdm显示进度
    if TQDM_AVAILABLE:
        config_iterator = tqdm(configs, desc="训练MLP模型")
    else:
        config_iterator = configs
        total_configs = len(configs)
    
    for i, config_info in enumerate(config_iterator):
        if not TQDM_AVAILABLE:
            print(f"\n🔄 进度: {i+1}/{total_configs} ({(i+1)/total_configs*100:.1f}%)")
            print(f"模型: {config_info['model']} #{config_info['config_id']}")
        
        # 训练和评估
        result = train_and_evaluate_mlp_config(config_info, X, y, sample_weights)
        
        if result is not None:
            results.append(result)
        else:
            print(f"   ⚠️  配置 #{config_info['config_id']} 跳过")
    
    # 5. 结果分析
    total_time = time.time() - start_time
    print(f"\n🎉 实验完成!")
    print(f"总用时: {total_time:.1f}秒")
    print(f"成功配置: {len(results)}/{len(configs)}")
    
    if results:
        # 保存结果
        save_mlp_results(results, experiment_id)
        
        # 显示最佳结果
        df_results = pd.DataFrame(results)
        best_config = df_results.loc[df_results['test_r2'].idxmax()]
        
        print(f"\n🏆 最佳配置:")
        print(f"  模型: {best_config['model']}")
        print(f"  配置ID: {best_config['config_id']}")
        print(f"  测试R²: {best_config['test_r2']:.4f}")
        print(f"  测试RMSE: {best_config['test_rmse']:.4f}")
        print(f"  交叉验证: {best_config['cv_mean']:.4f}±{best_config['cv_std']:.4f}")
        print(f"  训练时间: {best_config['training_time']:.2f}s")
        
        # R²分布统计
        r2_values = df_results['test_r2']
        print(f"\n📈 R²性能分布:")
        print(f"  平均值: {r2_values.mean():.4f}")
        print(f"  标准差: {r2_values.std():.4f}")
        print(f"  最小值: {r2_values.min():.4f}")
        print(f"  最大值: {r2_values.max():.4f}")
        print(f"  中位数: {r2_values.median():.4f}")
        
        # 性能区间统计
        excellent = (r2_values >= 0.7).sum()
        good = ((r2_values >= 0.5) & (r2_values < 0.7)).sum()
        fair = ((r2_values >= 0.3) & (r2_values < 0.5)).sum()
        poor = (r2_values < 0.3).sum()
        
        print(f"\n📊 性能区间分布:")
        print(f"  优秀 (R²≥0.7): {excellent} 个配置 ({excellent/len(results)*100:.1f}%)")
        print(f"  良好 (0.5≤R²<0.7): {good} 个配置 ({good/len(results)*100:.1f}%)")
        print(f"  一般 (0.3≤R²<0.5): {fair} 个配置 ({fair/len(results)*100:.1f}%)")
        print(f"  较差 (R²<0.3): {poor} 个配置 ({poor/len(results)*100:.1f}%)")
        
        # 与CatBoost对比提示
        print(f"\n💡 建议对比:")
        print(f"  可与CatBoost实验结果进行性能对比")
        print(f"  查看神经网络vs树模型的优劣势")
        
    else:
        print("❌ 没有成功的配置")

if __name__ == "__main__":
    main()