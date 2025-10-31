# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 数据预处理模块
Data Preprocessing Module for FRP Rebar Durability Prediction

包含：
- FRP材料数据的专业预处理
- 特征工程
- 数据清理和转换
"""

import pandas as pd
import numpy as np
import re
import hashlib
import json
from typing import Optional, Dict, List, Any, Tuple
import warnings

try:
    from .config import config
except ImportError:
    from config import config
from utils import validate_dataframe, safe_convert_to_numeric, clean_column_names

class FRPDataPreprocessor:
    """FRP数据预处理器 - 专门针对FRP材料数据的预处理"""
    
    def __init__(self, enable_caching: bool = False):
        """
        初始化预处理器
        
        Args:
            enable_caching: 是否启用缓存功能
        """
        self.enable_caching = enable_caching
        self.data_ori = None
        self.processed_data = None
        self.feature_columns = None
        
        # 材料属性配置
        self.material_props = config.MATERIAL_PROPERTIES
        
    def preprocess_data(self, df: pd.DataFrame, cache_key: Optional[str] = None) -> pd.DataFrame:
        """
        完整的数据预处理流程
        
        Args:
            df: 原始数据
            cache_key: 缓存键（如果启用缓存）
            
        Returns:
            预处理后的数据
        """
        
        print("🚀 Starting FRP data preprocessing...")
        
        # 验证输入数据
        validate_dataframe(df, name="Input data")
        
        # 保存原始数据
        self.data_ori = df.copy()
        
        # 步骤1: 基础数据清理
        df_clean = self.change_smd_to_nan(df)
        
        # 步骤2: 范围值解析
        df_parsed = self.parse_range_to_mean(df_clean)
        
        # 步骤3: 特征工程
        df_features = self.create_selected_features(df_parsed)
        
        # 步骤4: 创建模型数据集
        df_model = self.create_model_dataset(df_features)
        
        # 保存处理结果
        self.processed_data = df_model
        self.feature_columns = list(df_model.columns)
        
        print(f"✅ Preprocessing completed: {df_model.shape}")
        
        return df_model
    
    def change_smd_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将缺失值标记转换为NaN
        Convert 'SMD' to NaN, 'Notreported' to 'Unknown'
        """
        print("🧹 Processing missing value markers...")
        
        df_new = df.copy()
        
        # 批量处理缺失值标记
        missing_markers = {
            'SMD': np.nan,
            'smd': np.nan,
            'Notreported': 'Unknown',
            'not reported': 'Unknown',
            'Not reported': 'Unknown',
            'NOT REPORTED': 'Unknown'
        }
        
        for col in df_new.columns:
            df_new[col] = df_new[col].replace(missing_markers)
        
        print("✅ Missing value marker processing completed")
        return df_new
    
    def parse_range_to_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        解析范围值到平均值
        Process range strings like "20,30" to mean value
        """
        print("📏 Parsing range values to mean...")
        
        # 需要处理的数值列（基于原代码）
        numeric_columns = [
            'glass_transition_temperature', 'glass_transition_temperature_run_2',
            'cure_ratio', 'Fiber_content_weight', 'Fiber_content_volume',
            'Void_content', 'diameter', 'average_area', 'nominal_area',
            'num_1', 'temperature', 'pH_of_concrete', 'strength_of_concrete',
            'crack', 'pH_1', 'pHafter', 'RH_1', 'field_average_humidity',
            'field_average_temperature', 'temp', 'temp2', 'value_load',
            'Value1_1', 'COV1_1', 'Value2_1', 'COV2_1', 'Value3_1', 'COV3_1'
        ]
        
        df_new = df.copy()
        
        for col in numeric_columns:
            if col in df_new.columns:
                for idx in df_new.index:
                    value = df_new.loc[idx, col]
                    if isinstance(value, str):
                        # 检查是否包含逗号且无冒号（原代码条件）
                        if ',' in value and ':' not in value:
                            try:
                                # 提取数字并计算平均值
                                numbers = re.findall(r"\\d+\\.?\\d*", value)
                                if numbers:
                                    new_value = np.mean([float(x) for x in numbers])
                                    if not np.isnan(new_value):
                                        df_new.loc[idx, col] = new_value
                            except (ValueError, TypeError):
                                continue
        
        print("✅ Range value parsing completed")
        return df_new
    
    def create_selected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建选定特征
        基于原代码构建13个关键特征用于模型训练
        """
        print("🎯 Creating selected features...")
        
        # 保留原始重要列
        original_important_cols = ['Target_parameter', 'retention1', 'Value1_1', 'temperature', 'time_field']
        
        # 创建新特征列
        feature_columns = [
            'pH_of_condition_enviroment', 'Chloride_ion', 'concrete',
            'diameter', 'load_value', 'fiber_content', 'Glass_or_Basalt',
            'Vinyl_ester_or_Epoxy', 'condition_time', 'Temperature',
            'Tensile_strength_retention', 'surface_treatment',
            'max_strength', 'glass_transition_temperature'
        ]
        
        df_new = df.copy()
        
        # 初始化新特征列
        for col in feature_columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
        
        length = len(df_new)
        print(f"Processing feature engineering for {length} rows of data...")
        
        for i, idx in enumerate(df_new.index):
            try:
                # 1. pH和氯离子处理
                self._process_ph_and_chloride(df_new, idx)
                
                # 2. 混凝土指示器
                self._process_concrete_indicator(df_new, idx)
                
                # 3. 直径处理
                self._process_diameter(df_new, idx)
                
                # 4. 载荷处理
                self._process_load(df_new, idx)
                
                # 5. 纤维含量处理
                self._process_fiber_content(df_new, idx)
                
                # 6. 纤维和基体类型编码
                self._process_material_types(df_new, idx)
                
                # 7. 表面处理
                self._process_surface_treatment(df_new, idx)
                
                # 8. 其他特征
                self._process_other_features(df_new, idx)
                
            except Exception as e:
                continue
        
        print("✅ Selected feature creation completed")
        return df_new
    
    def _process_ph_and_chloride(self, df: pd.DataFrame, idx: int):
        """严格按照论文要求处理pH和氯离子特征"""
        
        # 初始化
        df.loc[idx, 'Chloride_ion'] = 0
        final_ph = 7.0  # 默认值
        
        # 步骤1: 确定环境类型
        is_concrete_environment = False
        
        # 检查Condition_environment字段
        if 'Condition_environment' in df.columns:
            condition_env = str(df.loc[idx, 'Condition_environment']).lower()
            concrete_keywords = ['concrete', 'cover', 'crack', 'cement', 'mortar']
            if any(keyword in condition_env for keyword in concrete_keywords):
                is_concrete_environment = True
        
        # 备用检查
        if not is_concrete_environment:
            concrete_cols = ['concrete', 'crack', 'cover', 'cement']
            for col in concrete_cols:
                if col in df.columns:
                    value = df.loc[idx, col]
                    if isinstance(value, str) or (isinstance(value, (int, float)) and not pd.isna(value)):
                        is_concrete_environment = True
                        break
        
        # 步骤2: 混凝土环境中的pH处理
        if is_concrete_environment:
            if 'pH_of_concrete' in df.columns:
                ph_concrete = df.loc[idx, 'pH_of_concrete']
                if isinstance(ph_concrete, (int, float)) and not pd.isna(ph_concrete):
                    final_ph = float(ph_concrete)
                else:
                    final_ph = 13.0  # 默认混凝土碱性
            else:
                final_ph = 13.0  # 默认混凝土碱性
        
        # 步骤3: 溶液环境中的pH处理
        else:
            ph_found = False
            
            # 检查solution_condition中的pH值
            if 'solution_condition' in df.columns:
                solution_condition = df.loc[idx, 'solution_condition']
                if isinstance(solution_condition, (int, float)) and not pd.isna(solution_condition):
                    final_ph = float(solution_condition)
                    ph_found = True
            
            # 备用: 检查pH相关字段
            if not ph_found:
                ph_columns = ['pH_1', 'pH', 'ph', 'PH']
                for ph_col in ph_columns:
                    if ph_col in df.columns:
                        ph_value = df.loc[idx, ph_col]
                        if isinstance(ph_value, (int, float)) and not pd.isna(ph_value):
                            final_ph = float(ph_value)
                            ph_found = True
                            break
            
            # 基于溶液类型描述分配值
            if not ph_found:
                solution_text = ''
                if 'solution_condition' in df.columns:
                    solution_text = str(df.loc[idx, 'solution_condition']).lower()
                
                if not solution_text and 'ingredient_1' in df.columns:
                    solution_text = str(df.loc[idx, 'ingredient_1']).lower()
                
                water_types = ['tap water', 'sea water', 'seawater', 'distilled water', 
                              'deionized water', 'di water', 'pure water']
                
                if any(water_type in solution_text for water_type in water_types):
                    final_ph = 7.0
                    
                    # 海水特殊处理
                    if 'sea' in solution_text:
                        df.loc[idx, 'Chloride_ion'] = 1
        
        # 步骤4: 考虑pHafter
        if 'pHafter' in df.columns:
            ph_after = df.loc[idx, 'pHafter']
            if isinstance(ph_after, (int, float)) and not pd.isna(ph_after):
                final_ph = (final_ph + float(ph_after)) / 2.0
        
        # 设置最终pH值
        df.loc[idx, 'pH_of_condition_enviroment'] = final_ph
        
        # 额外的氯离子检查
        if 'ingredient_1' in df.columns:
            ingredient = str(df.loc[idx, 'ingredient_1']).lower()
            chloride_keywords = ['cl', 'chloride', 'nacl', 'cacl2', 'mgcl2', 'salt']
            if any(keyword in ingredient for keyword in chloride_keywords):
                df.loc[idx, 'Chloride_ion'] = 1
    
    def _process_concrete_indicator(self, df: pd.DataFrame, idx: int):
        """处理混凝土指示器"""
        concrete_indicator = 0
        
        concrete_cols = ['concrete', 'crack', 'cover']
        for col in concrete_cols:
            if col in df.columns:
                value = df.loc[idx, col]
                if isinstance(value, str) or (isinstance(value, (int, float)) and not pd.isna(value)):
                    concrete_indicator = 1
                    break
        
        df.loc[idx, 'concrete'] = concrete_indicator
    
    def _process_diameter(self, df: pd.DataFrame, idx: int):
        """处理直径特征"""
        # 优先使用直接测量的直径
        if 'diameter' in df.columns:
            diameter_value = df.loc[idx, 'diameter']
            if isinstance(diameter_value, (int, float)) and not pd.isna(diameter_value):
                df.loc[idx, 'diameter'] = diameter_value
                return
        
        # 从nominal_area计算直径
        if 'nominal_area' in df.columns:
            area_value = df.loc[idx, 'nominal_area']
            if isinstance(area_value, (int, float)) and not pd.isna(area_value) and area_value > 0:
                calculated_diameter = 2 * np.sqrt(area_value / np.pi)
                df.loc[idx, 'diameter'] = calculated_diameter
    
    def _process_load(self, df: pd.DataFrame, idx: int):
        """处理载荷特征"""
        load_value = 0
        
        # 检查预载荷
        if 'type_of_load' in df.columns:
            if df.loc[idx, 'type_of_load'] == 'preloading':
                df.loc[idx, 'load_value'] = 0
                return
        
        # 处理应力/应变
        if 'stress_or_strain' in df.columns and 'value_load' in df.columns:
            stress_strain = df.loc[idx, 'stress_or_strain']
            value = df.loc[idx, 'value_load']
            
            if isinstance(value, (int, float)) and not pd.isna(value):
                if stress_strain == 'stress':
                    # 应力情况：需要除以极限拉伸强度
                    if 'ultimate_tensile_strength' in df.columns:
                        uts = df.loc[idx, 'ultimate_tensile_strength']
                        if isinstance(uts, (int, float)) and uts > 0:
                            load_value = value / uts
                elif stress_strain == 'strain':
                    # 应变情况：转换为相对应力
                    if 'tensile_modulus' in df.columns and 'ultimate_tensile_strength' in df.columns:
                        modulus = df.loc[idx, 'tensile_modulus']
                        uts = df.loc[idx, 'ultimate_tensile_strength']
                        if all(isinstance(x, (int, float)) and x > 0 for x in [modulus, uts]):
                            load_value = value * 0.001 * modulus / uts
        
        df.loc[idx, 'load_value'] = load_value
    
    def _process_fiber_content(self, df: pd.DataFrame, idx: int):
        """处理纤维含量特征"""
        # 优先使用重量百分比
        if 'Fiber_content_weight' in df.columns:
            weight_content = df.loc[idx, 'Fiber_content_weight']
            if isinstance(weight_content, (int, float)) and not pd.isna(weight_content):
                df.loc[idx, 'fiber_content'] = weight_content
                return
        
        # 从体积百分比转换
        if 'Fiber_content_volume' in df.columns:
            volume_content = df.loc[idx, 'Fiber_content_volume']
            if isinstance(volume_content, (int, float)) and not pd.isna(volume_content):
                # 获取密度
                fiber_type = df.loc[idx, 'Fiber_type'] if 'Fiber_type' in df.columns else 'Unknown'
                matrix_type = df.loc[idx, 'Matrix_type'] if 'Matrix_type' in df.columns else 'Unknown'
                
                # 纤维密度
                density_fiber = self.material_props['fiber_densities'].get(fiber_type, 2.0)
                
                # 基体密度
                density_matrix = self.material_props['matrix_densities'].get(matrix_type, 1.2)
                
                # 体积分数转重量分数
                weight_content = (100.0 * volume_content * density_fiber) / (
                    volume_content * density_fiber + (100.0 - volume_content) * density_matrix
                )
                df.loc[idx, 'fiber_content'] = weight_content
    
    def _process_material_types(self, df: pd.DataFrame, idx: int):
        """处理材料类型编码"""
        # 纤维类型编码 (Glass fiber=1, Basalt fiber=0)
        if 'Fiber_type' in df.columns:
            fiber_type = df.loc[idx, 'Fiber_type']
            if fiber_type == 'Glass':
                df.loc[idx, 'Glass_or_Basalt'] = 1
            elif fiber_type == 'Basalt':
                df.loc[idx, 'Glass_or_Basalt'] = 0
        
        # 基体类型编码 (Vinyl ester=1, Epoxy=0)
        if 'Matrix_type' in df.columns:
            matrix_type = df.loc[idx, 'Matrix_type']
            if matrix_type == 'Vinyl ester':
                df.loc[idx, 'Vinyl_ester_or_Epoxy'] = 1
            elif matrix_type == 'Epoxy':
                df.loc[idx, 'Vinyl_ester_or_Epoxy'] = 0
    
    def _process_surface_treatment(self, df: pd.DataFrame, idx: int):
        """处理表面处理特征"""
        if 'surface_treatment' in df.columns:
            treatment = df.loc[idx, 'surface_treatment']
            if treatment == 'sand coated':
                df.loc[idx, 'surface_treatment'] = 0
            elif treatment == 'Smooth':
                df.loc[idx, 'surface_treatment'] = 1
    
    def _process_other_features(self, df: pd.DataFrame, idx: int):
        """处理其他特征"""
        # 直接复制的特征映射
        feature_mappings = {
            'condition_time': 'time_field',
            'Temperature': 'temperature',
            'Tensile_strength_retention': 'retention1',
            'Target_parameter': 'Target_parameter',
            'max_strength': 'Value1_1',
            'glass_transition_temperature': 'glass_transition_temperature'
        }
        
        for new_col, old_col in feature_mappings.items():
            if old_col in df.columns:
                value = df.loc[idx, old_col]
                # 对于Target_parameter，直接复制
                if new_col == 'Target_parameter':
                    df.loc[idx, new_col] = value
                # 对于其他数值特征，检查是否为有效数值
                elif isinstance(value, (int, float)) and not pd.isna(value):
                    df.loc[idx, new_col] = value
                # 对于字符串类型的数值，尝试转换
                elif isinstance(value, str) and value.strip() != '':
                    try:
                        numeric_value = float(value)
                        if not np.isnan(numeric_value):
                            df.loc[idx, new_col] = numeric_value
                    except (ValueError, TypeError):
                        # 如果无法转换为数值，对某些字段仍保留原始值
                        if new_col in ['Target_parameter']:
                            df.loc[idx, new_col] = value
    
    def create_model_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建模型数据集
        遵循原代码的i0_data_pick到i010_data_pick流程
        """
        print("📊 Creating model dataset...")
        
        # 选择关键特征列
        model_columns = [
            'Title', 'Target_parameter', 'Tensile_strength_retention',
            'pH_of_condition_enviroment', 'condition_time', 'fiber_content',
            'Temperature', 'diameter', 'Chloride_ion', 'concrete',
            'load_value', 'Glass_or_Basalt', 'Vinyl_ester_or_Epoxy',
            'surface_treatment', 'max_strength'
        ]
        
        # 创建模型dataframe
        model_data = pd.DataFrame(index=df.index)
        
        for col in model_columns:
            if col == 'Title':
                model_data[col] = df['Title'] if 'Title' in df.columns else df.index
            else:
                model_data[col] = df[col] if col in df.columns else np.nan
        
        print(f"Initial model data shape: {model_data.shape}")
        
        # 使用所有数据，不按Target_parameter过滤
        tensile_data = model_data.copy()
        
        # 选择最终特征
        final_columns = [
            'Title', 'Tensile_strength_retention', 'pH_of_condition_enviroment',
            'condition_time', 'fiber_content', 'Temperature', 'diameter',
            'concrete', 'load_value', 'Chloride_ion', 'Glass_or_Basalt',
            'Vinyl_ester_or_Epoxy', 'surface_treatment', 'max_strength'
        ]
        
        # 检查数据完整性
        print("Feature data completeness analysis:")
        for col in final_columns:
            if col in tensile_data.columns:
                non_null_count = tensile_data[col].count()
                total_count = len(tensile_data)
                percentage = (non_null_count / total_count * 100) if total_count > 0 else 0
                print(f"   {col}: {non_null_count}/{total_count} ({percentage:.1f}%)")
        
        # 使用更宽松的dropna策略
        final_data = tensile_data[final_columns].copy()
        
        # 只移除完全空的行
        before_drop = len(final_data)
        final_data = final_data.dropna(how='all')  # 只移除所有值都是NaN的行
        after_drop = len(final_data)
        print(f"Removed completely empty rows: {before_drop} -> {after_drop}")
        
        # 进一步检查：如果数据仍然太少，使用更宽松的策略
        if len(final_data) < 100:  # 如果少于100行
            print("Data volume too small, using more lenient filtering strategy...")
            
            # 只要有Tensile_strength_retention（目标变量）就保留
            if 'Tensile_strength_retention' in tensile_data.columns:
                has_target = tensile_data['Tensile_strength_retention'].notna()
                final_data = tensile_data[has_target][final_columns].copy()
                print(f"After filtering by target variable: {len(final_data)} rows")
            
            # 如果仍然太少，只移除目标变量为空的行
            if len(final_data) < 50:
                final_data = tensile_data[final_columns].copy()
                if 'Tensile_strength_retention' in final_data.columns:
                    final_data = final_data.dropna(subset=['Tensile_strength_retention'])
                    print(f"Kept data with target variable: {len(final_data)} rows")
        
        # 重命名列以匹配原代码
        final_data.columns = [
            'Title', 'Tensile strength retention', 'pH of condition environment',
            'Exposure time', 'Fibre content', 'Exposure temperature', 'Diameter',
            'Presence of concrete', 'Load', 'Presence of chloride ion', 'Fibre type',
            'Matrix type', 'Surface treatment', 'Strength of unconditioned rebar'
        ]
        
        print(f"✅ Model dataset creation completed, final data shape: {final_data.shape}")
        
        return final_data
    
    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息"""
        if self.processed_data is None:
            return {}
        
        # 分离数值和分类特征
        numeric_features = []
        categorical_features = []
        
        for col in self.processed_data.columns:
            if col in ['Title', 'Tensile strength retention']:
                continue
            
            if self.processed_data[col].dtype in ['int64', 'float64']:
                # 检查是否为二进制分类特征
                unique_values = self.processed_data[col].dropna().unique()
                if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values if not pd.isna(v)):
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        return {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'target_variable': 'Tensile strength retention',
            'all_features': list(self.processed_data.columns),
            'training_columns': [col for col in self.processed_data.columns if col not in ['Title', 'Tensile strength retention']]
        }

# 便捷函数
def preprocess_frp_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """便捷的FRP数据预处理函数"""
    preprocessor = FRPDataPreprocessor()
    processed_data = preprocessor.preprocess_data(df)
    feature_info = preprocessor.get_feature_info()
    
    return processed_data, feature_info