#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP钢筋耐久性预测 - 40参数优化实验 (预定义特征版)

特点：
1. 三种机器学习模型：RandomForest, XGBoost, LightGBM
2. 每个模型约13-14种参数配置，总计40个配置
3. 5折交叉验证
4. 只使用Comments=1的数据进行训练和验证
5. 使用与app.py相同的13个预定义工程特征进行训练
6. 快速高效的参数搜索策略
7. 增强的错误处理和进度跟踪

13个预定义工程特征：
- pH_of_condition_enviroment, Chloride_ion, concrete
- diameter, load_value, fiber_content, Glass_or_Basalt
- Vinyl_ester_or_Epoxy, condition_time, Temperature
- Tensile_strength_retention, surface_treatment
- max_strength, glass_transition_temperature
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
from tqdm import tqdm

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost未安装，将跳过XGBoost实验")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM未安装，将跳过LightGBM实验")
    LIGHTGBM_AVAILABLE = False

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

class ValidDataLoader:
    """有效数据加载器 - 只加载Comments=1的数据 (预定义特征版)"""
    
    def __init__(self, file_path=None):
        if file_path is None:
            # 扩展搜索路径
            possible_paths = [
                "E:/大学/intern/2025-summer-concret/database 4.xlsx",
                "E:\\大学\\intern\\2025-summer-concret\\database 4.xlsx",
                "../database 4.xlsx",
                "../../database 4.xlsx", 
                "../../../database 4.xlsx",
                "data/database 4.xlsx",
                "../data/database 4.xlsx",
                "data/research_data.xlsx",
                "../data/research_data.xlsx", 
                "../../data/research_data.xlsx",
                "data/train_data.xlsx",
                "../data/train_data.xlsx",
                "../frp_local/data/research_data.xlsx",
                "../frp_local/data/train_data.xlsx",
                "data/research_data.csv",
                "../data/research_data.csv",
                "../frp_local/data/research_data.csv",
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    self.file_path = path
                    print(f"✅ 找到数据文件: {path}")
                    break
            else:
                print("⚠️  未找到数据文件，将创建模拟数据进行演示")
                self.file_path = None
                self.use_mock_data = True
                return
        else:
            self.file_path = file_path
        
        self.use_mock_data = False
    
    def load_valid_data(self):
        """加载有效数据（仅Comments=1）- 增强版"""
        
        if self.use_mock_data:
            return self._create_mock_data()
        
        print("🔄 加载原始Excel文件...")
        try:
            file_path = Path(self.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {self.file_path}")
            
            # 根据文件扩展名选择读取方法
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                # 尝试读取第一个工作表
                try:
                    raw_data = pd.read_excel(self.file_path, sheet_name=0)
                except Exception as e:
                    print(f"读取第一个工作表失败，尝试默认读取: {e}")
                    raw_data = pd.read_excel(self.file_path)
            elif file_path.suffix.lower() == '.csv':
                raw_data = pd.read_csv(self.file_path, encoding='utf-8')
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")
                
        except Exception as e:
            print(f"❌ 读取数据文件失败: {e}")
            print("将使用模拟数据继续实验...")
            return self._create_mock_data()
        
        print(f"原始数据形状: {raw_data.shape}")
        print(f"原始数据列数: {len(raw_data.columns)}")
        
        # 检查是否有Comments列
        comments_col = None
        for col in raw_data.columns:
            if 'comment' in str(col).lower():
                comments_col = col
                break
        
        if comments_col is None:
            print("⚠️  未找到Comments列，将使用所有数据")
            valid_data = raw_data.copy()
        else:
            # 筛选条件：只检查Comments列为1
            try:
                comments_mask = pd.to_numeric(raw_data[comments_col], errors='coerce') == 1
                valid_data = raw_data[comments_mask].copy()
                
                print(f"🎯 数据筛选结果:")
                print(f"  原始数据: {len(raw_data)} 行")
                print(f"  Comments=1: {comments_mask.sum()} 行")
                if len(raw_data) > 0:
                    print(f"  最终筛选比例: {len(valid_data)/len(raw_data)*100:.1f}%")
            except Exception as e:
                print(f"⚠️  Comments列处理失败: {e}")
                valid_data = raw_data.copy()
        
        # 数据质量检查
        print(f"📊 数据质量检查:")
        print(f"  有效数据形状: {valid_data.shape}")
        print(f"  缺失值总数: {valid_data.isnull().sum().sum()}")
        
        return valid_data
    
    def _create_mock_data(self):
        """创建模拟数据用于演示 - 包含app.py需要的列"""
        print("📝 创建模拟数据...")
        
        np.random.seed(42)
        n_samples = 500
        
        # 创建包含app.py所需列的模拟数据
        mock_data = pd.DataFrame({
            'Comments': np.ones(n_samples),  # 所有数据Comments=1
            'feature_name': ['FRP_' + str(i) for i in range(n_samples)],
            'Year': np.random.randint(2000, 2024, n_samples),
            'no.': np.arange(n_samples),
            'Value1_1': np.random.normal(1000, 200, n_samples),  # 目标变量
            'diameter': np.random.normal(10, 2, n_samples),
            'No.': np.random.randint(1, 100, n_samples),
            
            # app.py预处理需要的列
            'solution_condition': np.random.choice(['tap water', 'sea water', 'distilled water'], n_samples),
            'temperature': np.random.normal(25, 5, n_samples),
            'time_field': np.random.normal(100, 30, n_samples),
            'pH_1': np.random.normal(7, 1, n_samples),
            'retention1': np.random.normal(0.8, 0.1, n_samples),
            'Fiber_content_weight': np.random.normal(60, 10, n_samples),
            'fiber_type': np.random.choice(['Glass', 'Basalt'], n_samples),
            'resin_type': np.random.choice(['Vinyl_ester', 'Epoxy'], n_samples),
            'surface_treatment': np.random.choice(['Yes', 'No'], n_samples),
            'ultimate_tensile_strength': np.random.normal(800, 100, n_samples),
            'stress_or_strain': np.random.choice(['stress', 'strain'], n_samples),
            'value_load': np.random.normal(0.5, 0.2, n_samples),
            'glass_transition_temperature': np.random.normal(120, 20, n_samples),
            'concrete': np.random.choice([0, 1], n_samples),
            'crack': np.random.choice([0, 1], n_samples),
            'cover': np.random.choice([0, 1], n_samples),
            'nominal_area': np.random.normal(78.5, 10, n_samples),
            'type_of_load': np.random.choice(['normal', 'preloading'], n_samples),
            'tensile_modulus': np.random.normal(40000, 5000, n_samples),
        })
        
        # 确保有足够的列（至少40列）
        for i in range(len(mock_data.columns), 50):
            mock_data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        print(f"✅ 模拟数据创建完成: {mock_data.shape}")
        print(f"   包含app.py所需的预处理列")
        return mock_data
    
    def prepare_features_target(self, data):
        """准备特征和目标变量 - 预定义特征版本，与app.py保持一致"""
        print("🔧 开始预定义特征工程（与app.py保持一致）...")
        
        # 定义13个预定义工程特征的映射关系
        print("📋 使用预定义的13个工程特征...")
        
        # 预定义特征映射 - 基于app.py的特征工程
        predefined_features = {
            'pH_of_condition_enviroment': ['pH_1', 'pH', 'ph', 'PH'],
            'Chloride_ion': ['Chloride', 'chloride', 'Cl', 'cl'],
            'concrete': ['concrete', 'Concrete'],
            'diameter': ['diameter', 'Diameter', 'dia'],
            'load_value': ['value_load', 'load', 'Load', 'stress_value'],
            'fiber_content': ['Fiber_content_weight', 'fiber_content', 'fiber'],
            'Glass_or_Basalt': ['fiber_type', 'Glass_or_Basalt'],
            'Vinyl_ester_or_Epoxy': ['resin_type', 'Vinyl_ester_or_Epoxy'],
            'condition_time': ['time_field', 'condition_time', 'time'],
            'Temperature': ['temperature', 'Temperature', 'temp'],
            'Tensile_strength_retention': ['retention1', 'retention', 'strength_retention'],
            'surface_treatment': ['surface_treatment', 'treatment'],
            'glass_transition_temperature': ['glass_transition_temperature', 'Tg1', 'Tg']
        }
        
        # 目标变量候选列
        target_candidates = [
            'retention1', 'Tensile_strength_retention', 'strength_retention',
            'normally stress, MPa', 'stress', 'Value1_1'
        ]
        
        print(f"🎯 预定义特征映射:")
        for feat_name, col_candidates in predefined_features.items():
            print(f"  {feat_name}: {col_candidates}")
        
        # 查找匹配的列
        feature_mapping = {}
        found_features = []
        
        for feature_name, candidates in predefined_features.items():
            found_col = None
            for candidate in candidates:
                # 精确匹配
                if candidate in data.columns:
                    found_col = candidate
                    break
                # 模糊匹配
                for col in data.columns:
                    if candidate.lower() in str(col).lower():
                        found_col = col
                        break
                if found_col:
                    break
            
            if found_col:
                feature_mapping[feature_name] = found_col
                found_features.append(feature_name)
                print(f"  ✅ {feature_name} -> {found_col}")
            else:
                print(f"  ❌ {feature_name} -> 未找到匹配列")
        
        print(f"\n📊 特征匹配结果:")
        print(f"  找到特征: {len(found_features)}/13")
        print(f"  匹配特征: {found_features}")
        
        # 查找目标变量
        target_col = None
        for candidate in target_candidates:
            if candidate in data.columns:
                target_col = candidate
                break
            # 模糊匹配
            for col in data.columns:
                if candidate.lower() in str(col).lower():
                    target_col = col
                    break
            if target_col:
                break
        
        if target_col:
            print(f"  🎯 目标变量: {target_col}")
        else:
            print(f"  ⚠️  未找到目标变量，使用备用方法...")
            return self._fallback_predefined_extraction(data)
        
        # 检查最小特征要求
        if len(found_features) < 3:
            print(f"  ⚠️  找到的特征太少({len(found_features)}个)，使用备用方法...")
            return self._fallback_predefined_extraction(data)
        
        # 提取特征数据
        print(f"\n🔧 提取特征数据...")
        feature_data = {}
        
        for feature_name in found_features:
            col_name = feature_mapping[feature_name]
            
            # 特殊处理分类特征
            if feature_name in ['Glass_or_Basalt', 'Vinyl_ester_or_Epoxy', 'surface_treatment', 'concrete']:
                # 编码分类特征
                feature_data[feature_name] = self._encode_categorical_feature(data[col_name], feature_name)
            else:
                # 数值特征
                feature_data[feature_name] = pd.to_numeric(data[col_name], errors='coerce')
        
        # 目标变量
        y_raw = pd.to_numeric(data[target_col], errors='coerce')
        
        # 创建特征DataFrame
        X_raw = pd.DataFrame(feature_data)
        
        # 数据质量检查
        print(f"📈 数据质量检查:")
        print(f"  原始数据: {len(data)} 行")
        print(f"  特征矩阵: {X_raw.shape}")
        
        # 显示各特征的有效数据量
        for feature_name in X_raw.columns:
            valid_count = X_raw[feature_name].count()
            valid_ratio = valid_count / len(X_raw) * 100
            print(f"    {feature_name}: {valid_count}/{len(X_raw)} ({valid_ratio:.1f}%)")
        
        # 合并数据并清理缺失值
        combined = pd.concat([X_raw, y_raw], axis=1)
        combined_clean = combined.dropna()
        
        print(f"  清理后数据: {len(combined_clean)} 行")
        print(f"  数据保留率: {len(combined_clean)/len(data)*100:.1f}%")
        
        if len(combined_clean) < 50:
            print(f"  ⚠️  清理后数据不足，使用备用方法...")
            return self._fallback_predefined_extraction(data)
        
        # 分离特征和目标
        X_clean = combined_clean.iloc[:, :-1]
        y_clean = combined_clean.iloc[:, -1]
        
        # 重命名特征列以避免特殊字符问题
        feature_name_map = {col: f"feat_{i}_{col.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')}" 
                           for i, col in enumerate(X_clean.columns)}
        X_clean = X_clean.rename(columns=feature_name_map)
        
        # 数据统计
        print(f"\n✅ 预定义特征提取成功:")
        print(f"  最终特征数: {X_clean.shape[1]}")
        print(f"  样本数: {len(X_clean)}")
        print(f"  目标变量: {target_col}")
        print(f"  目标范围: [{y_clean.min():.3f}, {y_clean.max():.3f}]")
        print(f"  目标标准差: {y_clean.std():.4f}")
        
        # 显示最终特征列表
        print(f"  最终特征列表:")
        for i, (old_name, new_name) in enumerate(feature_name_map.items(), 1):
            print(f"    {i:2d}. {old_name} -> {new_name}")
        
        return X_clean, y_clean, list(X_clean.columns)
    
    def _encode_categorical_feature(self, series, feature_name):
        """编码分类特征"""
        if feature_name == 'Glass_or_Basalt':
            # 纤维类型编码：Glass=1, Basalt=0
            return series.map({'Glass': 1, 'Basalt': 0, 'glass': 1, 'basalt': 0}).fillna(0)
        
        elif feature_name == 'Vinyl_ester_or_Epoxy':
            # 树脂类型编码：Vinyl_ester=1, Epoxy=0
            return series.map({'Vinyl_ester': 1, 'Epoxy': 0, 'vinyl_ester': 1, 'epoxy': 0}).fillna(0)
        
        elif feature_name == 'surface_treatment':
            # 表面处理编码：Yes/是=1, No/否=0
            return series.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, '是': 1, '否': 0}).fillna(0)
        
        elif feature_name == 'concrete':
            # 混凝土环境编码：有=1, 无=0
            return pd.to_numeric(series, errors='coerce').fillna(0)
        
        else:
            # 默认数值转换
            return pd.to_numeric(series, errors='coerce').fillna(0)
    
    def _fallback_predefined_extraction(self, data):
        """备用的预定义特征提取方法"""
        print("🔄 使用备用预定义特征提取...")
        
        # 简化的预定义特征映射
        simple_features = {
            'temperature': ['temperature', 'Temperature', 'temp'],
            'time': ['time_field', 'condition_time', 'time'],
            'pH': ['pH_1', 'pH', 'ph'],
            'fiber_content': ['Fiber_content_weight', 'fiber_content'],
            'diameter': ['diameter', 'Diameter'],
            'load': ['value_load', 'load', 'Load']
        }
        
        # 简化目标变量
        simple_targets = ['retention1', 'Value1_1', 'normally stress, MPa']
        
        feature_data = {}
        found_count = 0
        
        # 查找特征
        for feat_name, candidates in simple_features.items():
            for candidate in candidates:
                if candidate in data.columns:
                    numeric_data = pd.to_numeric(data[candidate], errors='coerce')
                    if numeric_data.count() > len(data) * 0.1:  # 至少10%有效数据
                        feature_data[f"feat_{found_count}_{feat_name}"] = numeric_data
                        found_count += 1
                        print(f"  ✅ {feat_name} -> {candidate}")
                        break
        
        # 查找目标变量
        target_col = None
        for candidate in simple_targets:
            if candidate in data.columns:
                target_data = pd.to_numeric(data[candidate], errors='coerce')
                if target_data.count() > len(data) * 0.3:  # 至少30%有效数据
                    target_col = candidate
                    break
        
        if len(feature_data) >= 3 and target_col is not None:
            X_simple = pd.DataFrame(feature_data)
            y_simple = pd.to_numeric(data[target_col], errors='coerce')
            
            # 清理数据
            combined = pd.concat([X_simple, y_simple], axis=1)
            combined_clean = combined.dropna()
            
            if len(combined_clean) >= 50:
                X_clean = combined_clean.iloc[:, :-1]
                y_clean = combined_clean.iloc[:, -1]
                
                print(f"✅ 备用预定义特征提取成功:")
                print(f"  特征数: {X_clean.shape[1]}")
                print(f"  样本数: {len(X_clean)}")
                print(f"  目标变量: {target_col}")
                
                return X_clean, y_clean, list(X_clean.columns)
        
        print("❌ 备用预定义特征提取也失败，使用最简单的备用方案...")
        return self._fallback_feature_extraction(data)
    
    def _create_mock_data(self):
        """创建模拟数据用于演示 - 包含app.py需要的列"""
        print("📝 创建模拟数据...")
        
        np.random.seed(42)
        n_samples = 500
        
        # 创建包含app.py所需列的模拟数据
        mock_data = pd.DataFrame({
            'Comments': np.ones(n_samples),  # 所有数据Comments=1
            'feature_name': ['FRP_' + str(i) for i in range(n_samples)],
            'Year': np.random.randint(2000, 2024, n_samples),
            'no.': np.arange(n_samples),
            'Value1_1': np.random.normal(1000, 200, n_samples),  # 目标变量
            'diameter': np.random.normal(10, 2, n_samples),
            'No.': np.random.randint(1, 100, n_samples),
            
            # app.py预处理需要的列
            'solution_condition': np.random.choice(['tap water', 'sea water', 'distilled water'], n_samples),
            'temperature': np.random.normal(25, 5, n_samples),
            'time_field': np.random.normal(100, 30, n_samples),
            'pH_1': np.random.normal(7, 1, n_samples),
            'retention1': np.random.normal(0.8, 0.1, n_samples),
            'Fiber_content_weight': np.random.normal(60, 10, n_samples),
            'fiber_type': np.random.choice(['Glass', 'Basalt'], n_samples),
            'resin_type': np.random.choice(['Vinyl_ester', 'Epoxy'], n_samples),
            'surface_treatment': np.random.choice(['Yes', 'No'], n_samples),
            'ultimate_tensile_strength': np.random.normal(800, 100, n_samples),
            'stress_or_strain': np.random.choice(['stress', 'strain'], n_samples),
            'value_load': np.random.normal(0.5, 0.2, n_samples),
            'glass_transition_temperature': np.random.normal(120, 20, n_samples),
            'concrete': np.random.choice([0, 1], n_samples),
            'crack': np.random.choice([0, 1], n_samples),
            'cover': np.random.choice([0, 1], n_samples),
            'nominal_area': np.random.normal(78.5, 10, n_samples),
            'type_of_load': np.random.choice(['normal', 'preloading'], n_samples),
            'tensile_modulus': np.random.normal(40000, 5000, n_samples),
        })
        
        # 确保有足够的列（至少40列）
        for i in range(len(mock_data.columns), 50):
            mock_data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        print(f"✅ 模拟数据创建完成: {mock_data.shape}")
        print(f"   包含app.py所需的预处理列")
        return mock_data

def get_40_parameter_configs():
    """获取40个参数配置（每个模型13-14个）"""
    
    configs = []
    
    # RandomForest 参数配置 (14个)
    rf_configs = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 7, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 3, 'random_state': 42},
        {'n_estimators': 150, 'max_depth': 8, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': 6, 'min_samples_split': 4, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 9, 'min_samples_split': 3, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5, 'random_state': 42},
        {'n_estimators': 150, 'max_depth': 4, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 250, 'max_depth': 7, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 6, 'random_state': 42},
        {'n_estimators': 400, 'max_depth': 8, 'min_samples_split': 3, 'random_state': 42},
        {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 4, 'random_state': 42},
    ]
    
    for i, config in enumerate(rf_configs):
        configs.append({
            'model': 'RandomForest',
            'config_id': i + 1,
            'config': config
        })
    
    # XGBoost 参数配置 (13个)
    if XGBOOST_AVAILABLE:
        xgb_configs = [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.2, 'random_state': 42},
            {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42},
            {'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.08, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.1, 'random_state': 42},
            {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.03, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.15, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42},
            {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.02, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.1, 'random_state': 42},
        ]
        
        for i, config in enumerate(xgb_configs):
            configs.append({
                'model': 'XGBoost',
                'config_id': i + 1,
                'config': config
            })
    
    # LightGBM 参数配置 (13个)
    if LIGHTGBM_AVAILABLE:
        lgb_configs = [
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 63, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'num_leaves': 15, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.2, 'num_leaves': 127, 'random_state': 42},
            {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'num_leaves': 7, 'random_state': 42},
            {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.08, 'num_leaves': 31, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 255, 'random_state': 42},
            {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.03, 'num_leaves': 15, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.15, 'num_leaves': 1023, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 63, 'random_state': 42},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': 42},
            {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.02, 'num_leaves': 15, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 12, 'learning_rate': 0.1, 'num_leaves': 4095, 'random_state': 42},
        ]
        
        for i, config in enumerate(lgb_configs):
            configs.append({
                'model': 'LightGBM',
                'config_id': i + 1,
                'config': config
            })
    
    return configs

def train_and_evaluate_config(config_info, X, y, cv_folds=5):
    """训练和评估单个配置 - 增强版"""
    model_name = config_info['model']
    config = config_info['config']
    config_id = config_info['config_id']
    
    try:
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 创建模型
        if model_name == 'RandomForest':
            model = RandomForestRegressor(**config)
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            # 添加早停参数
            config_xgb = config.copy()
            config_xgb.update({'verbosity': 0, 'early_stopping_rounds': 10})
            model = xgb.XGBRegressor(**config_xgb)
        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            # 添加早停参数
            config_lgb = config.copy()
            config_lgb.update({'verbosity': -1, 'early_stopping_rounds': 10})
            model = lgb.LGBMRegressor(**config_lgb)
        else:
            return None
        
        # 交叉验证 - 使用标准化数据
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # 为XGBoost和LightGBM使用未标准化数据（它们内部处理）
        if model_name in ['XGBoost', 'LightGBM']:
            cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
            X_for_split = X
        else:
            cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
            X_for_split = X_scaled
        
        # 训练测试分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_for_split, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        
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
            'n_features': X.shape[1]
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 配置 {model_name} #{config_id} 训练失败: {e}")
        return None

def save_results(results, experiment_id):
    """保存实验结果"""
    # 创建结果目录
    result_dir = Path("experiments")
    result_dir.mkdir(exist_ok=True)
    
    # 保存为CSV
    df = pd.DataFrame(results)
    csv_path = result_dir / f"40param_exp_{experiment_id}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 保存详细结果为JSON
    json_path = result_dir / f"40param_exp_{experiment_id}_detailed.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(results),
            'data_filter': 'Comments=1',
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 结果已保存:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")

def main():
    print("🚀 开始40参数优化实验 (预定义特征版)")
    print("=" * 60)
    print("📋 使用预定义工程特征，与app.py保持一致")
    
    start_time = time.time()
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 加载数据
    print("📂 加载数据...")
    loader = ValidDataLoader()
    data = loader.load_valid_data()
    
    if data is None or len(data) == 0:
        print("❌ 无有效数据，实验终止")
        return
    
    # 2. 准备特征和目标变量
    print("\n🔧 准备预定义特征和目标变量...")
    result = loader.prepare_features_target(data)
    
    if result is None or len(result) != 3:
        print("❌ 特征提取失败，实验终止")
        return
    
    X, y, feature_names = result
    
    if X is None or y is None:
        print("❌ 特征提取失败，实验终止")
        return
    
    print(f"✅ 预定义特征准备完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"特征列: {feature_names}")
    print(f"💡 使用与app.py相同的预定义工程特征")
    
    # 数据概览
    print(f"\n📊 数据概览:")
    print(f"  特征范围: {X.min().min():.3f} ~ {X.max().max():.3f}")
    print(f"  目标范围: {y.min():.3f} ~ {y.max():.3f}")
    print(f"  目标均值±标准差: {y.mean():.3f}±{y.std():.3f}")
    
    # 3. 获取参数配置
    print("\n⚙️  准备参数配置...")
    configs = get_40_parameter_configs()
    print(f"总配置数: {len(configs)}")
    
    # 统计每个模型的配置数
    model_counts = {}
    for config in configs:
        model_name = config['model']
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    print("各模型配置数:")
    for model, count in model_counts.items():
        available = "✅" if (
            model == 'RandomForest' or 
            (model == 'XGBoost' and XGBOOST_AVAILABLE) or 
            (model == 'LightGBM' and LIGHTGBM_AVAILABLE)
        ) else "❌"
        print(f"  {available} {model}: {count} 个配置")
    
    # 4. 运行实验
    print(f"\n🔬 开始运行实验...")
    results = []
    failed_configs = []
    
    # 使用进度条
    config_iterator = tqdm(configs, desc="训练模型") if TQDM_AVAILABLE else configs
    
    for i, config_info in enumerate(config_iterator, 1):
        model_name = config_info['model']
        config_id = config_info['config_id']
        
        if not TQDM_AVAILABLE:
            print(f"\n[{i:2d}/{len(configs)}] 训练 {model_name} 配置 #{config_id}...")
        
        result = train_and_evaluate_config(config_info, X, y)
        
        if result:
            results.append(result)
            if not TQDM_AVAILABLE:
                print(f"   ✅ R²: {result['test_r2']:.6f}, "
                      f"CV: {result['cv_mean']:.6f}±{result['cv_std']:.6f}, "
                      f"时间: {result['training_time']:.2f}s")
        else:
            failed_configs.append(f"{model_name} #{config_id}")
            if not TQDM_AVAILABLE:
                print(f"   ❌ 配置失败")
        
        # 每10个配置保存一次
        if i % 10 == 0 and results:
            save_results(results, experiment_id)
            if not TQDM_AVAILABLE:
                print(f"   💾 已保存前 {len(results)} 个结果")
    
    # 5. 最终保存和分析
    total_time = time.time() - start_time
    print(f"\n📊 实验完成！总用时: {total_time/60:.1f} 分钟")
    
    if results:
        save_results(results, experiment_id)
        
        # 详细分析结果
        df = pd.DataFrame(results)
        print(f"\n🎯 实验结果总结:")
        print(f"   成功配置: {len(results)}/{len(configs)}")
        print(f"   失败配置: {len(failed_configs)}")
        print(f"   平均训练时间: {df['training_time'].mean():.2f}s")
        print(f"   总训练时间: {df['training_time'].sum():.1f}s")
        
        # 性能统计
        print(f"\n📈 性能统计:")
        print(f"   最佳R²: {df['test_r2'].max():.6f}")
        print(f"   平均R²: {df['test_r2'].mean():.6f}")
        print(f"   R²标准差: {df['test_r2'].std():.6f}")
        print(f"   最低RMSE: {df['test_rmse'].min():.6f}")
        print(f"   平均RMSE: {df['test_rmse'].mean():.6f}")
        
        # 各模型最佳结果
        print(f"\n🏆 各模型最佳结果:")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            if len(model_df) > 0:
                best = model_df.loc[model_df['test_r2'].idxmax()]
                print(f"   {model}: R²={best['test_r2']:.6f}, "
                      f"RMSE={best['test_rmse']:.6f} (配置#{best['config_id']})")
        
        # TOP10 配置
        print(f"\n🥇 TOP10 配置:")
        top10 = df.nlargest(10, 'test_r2')
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"   {i:2d}. {row['model']} #{row['config_id']}: "
                  f"R²={row['test_r2']:.6f}, RMSE={row['test_rmse']:.6f}")
        
        # 失败的配置
        if failed_configs:
            print(f"\n❌ 失败的配置:")
            for config in failed_configs:
                print(f"   {config}")
    
    else:
        print("❌ 没有成功的配置")
    
    print(f"\n🎉 实验完成! 结果已保存到 experiments/ 目录")

if __name__ == "__main__":
    main()