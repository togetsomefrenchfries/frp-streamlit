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
                
                # 9. 样品权重处理
                self._process_sample_weight(df_new, idx)
                
            except Exception as e:
                continue
        
        print("✅ Selected feature creation completed")
        return df_new
    
    def _process_ph_and_chloride(self, df: pd.DataFrame, idx: int):
        """
        📋 特征1&2: pH_of_condition_enviroment & Chloride_ion
        
        🔍 什么是"源列"？
        源列 = 原始Excel/CSV文件中的列名，是数据提取的来源
        目标特征 = 我们要生成的新特征列名，用于机器学习模型
        
        📊 源列到目标特征的映射过程：
        
        原始数据表格列名(源列)           →    处理逻辑    →    目标特征名
        ├─ 'pH_of_concrete'             →    直接读取    →    'pH_of_condition_enviroment'
        ├─ 'solution_condition'         →    数值提取    →    'pH_of_condition_enviroment'  
        ├─ 'pH', 'pH_1', 'pH.1'        →    数值转换    →    'pH_of_condition_enviroment'
        └─ 'ingredient', 'cycle_pH'     →    关键词搜索  →    'Chloride_ion'
        
        🔍 具体示例：
        如果原始Excel表格有这些列：
        - 第15列：名为 'pH_1'，包含数据 [7.2, 8.5, 6.8, ...]
        - 第23列：名为 'ingredient'，包含文本 ['NaCl solution', 'tap water', ...]
        
        处理过程：
        1. 查找源列 'pH_1' → 读取数值7.2 → 赋值给目标特征 'pH_of_condition_enviroment'
        2. 查找源列 'ingredient' → 检测到'NaCl' → 设置目标特征 'Chloride_ion' = 1
        """
        
        # 🎯 源列查找优先级列表
        ph_source_columns = [
            'pH_of_concrete',      # 源列1：混凝土pH专用列
            'solution_condition',  # 源列2：溶液条件列  
            'pH', 'pH_1', 'pH.1', 'pH.2', 'ph', 'PH'  # 源列3-8：各种pH列变体
        ]
        
        chloride_source_columns = [
            'ingredient',          # 源列1：成分描述
            'ingredient.1',        # 源列2：成分描述备用
            'cycle_ingredient',    # 源列3：循环成分
            'cycle_pH',           # 源列4：循环pH
            'pHafter',            # 源列5：pH后续值
            'concrete',           # 源列6：混凝土描述
            'note_of_concrete'    # 源列7：混凝土备注
        ]
        
        print(f"  🔍 正在处理第{idx}行数据...")
        print(f"  📋 pH源列候选: {ph_source_columns}")
        print(f"  📋 氯离子源列候选: {chloride_source_columns}")
        
        # 初始化
        df.loc[idx, 'Chloride_ion'] = 0
        final_ph = 7.0  # 默认中性
        
        # 步骤1: 判断是否为混凝土环境
        is_concrete_environment = False
        
        # 检查Condition_environment字段
        if 'Condition_environment' in df.columns:
            condition_env = str(df.loc[idx, 'Condition_environment']).lower()
            concrete_keywords = ['concrete', 'cover', 'crack', 'cement', 'mortar']
            if any(keyword in condition_env for keyword in concrete_keywords):
                is_concrete_environment = True
        
        # 备用检查concrete相关列
        if not is_concrete_environment:
            concrete_cols = ['concrete', 'crack', 'cover', 'cement']
            for col in concrete_cols:
                if col in df.columns:
                    value = df.loc[idx, col]
                    if isinstance(value, str) or (isinstance(value, (int, float)) and not pd.isna(value)):
                        is_concrete_environment = True
                        break
        
        # 步骤2: 根据环境类型处理pH
        if is_concrete_environment:
            # 混凝土环境：使用pH_of_concrete或默认13.0
            if 'pH_of_concrete' in df.columns:
                ph_concrete = df.loc[idx, 'pH_of_concrete']
                if isinstance(ph_concrete, (int, float)) and not pd.isna(ph_concrete):
                    final_ph = float(ph_concrete)
                else:
                    final_ph = 13.0  # 混凝土碱性环境
            else:
                final_ph = 13.0
        else:
            # 溶液环境：多级检查
            ph_found = False
            
            # 优先：solution_condition列中的数值
            if 'solution_condition' in df.columns:
                solution_condition = df.loc[idx, 'solution_condition']
                if isinstance(solution_condition, (int, float)) and not pd.isna(solution_condition):
                    final_ph = float(solution_condition)
                    ph_found = True
            
            # 备用：pH相关列
            if not ph_found:
                ph_columns = ['pH', 'pH_1', 'pH.1', 'pH.2', 'ph', 'PH']
                for ph_col in ph_columns:
                    if ph_col in df.columns:
                        ph_value = df.loc[idx, ph_col]
                        if isinstance(ph_value, (int, float)) and not pd.isna(ph_value):
                            final_ph = float(ph_value)
                            ph_found = True
                            break
                        elif isinstance(ph_value, str) and ph_value.strip():
                            try:
                                numeric_ph = float(ph_value.strip())
                                final_ph = numeric_ph
                                ph_found = True
                                break
                            except ValueError:
                                continue
            
            # 最后：根据水类型描述推断
            if not ph_found:
                solution_text = ''
                water_description_columns = [
                    'solution_condition', 'ingredient', 'cycle_pH', 
                    'pHafter', 'concrete', 'note_of_concrete'
                ]
                
                for col in water_description_columns:
                    if col in df.columns and not solution_text:
                        col_value = df.loc[idx, col]
                        if col_value is not None and not pd.isna(col_value):
                            solution_text = str(col_value).lower()
                            if solution_text and solution_text != 'nan':
                                break
                
                water_types = ['tap water', 'sea water', 'seawater', 'distilled water', 
                              'deionized water', 'di water', 'pure water']
                
                if any(water_type in solution_text for water_type in water_types):
                    final_ph = 7.0
                    # 海水特殊处理
                    if 'sea' in solution_text:
                        df.loc[idx, 'Chloride_ion'] = 1
        
        # 步骤3: 考虑pHafter进行平均
        if 'pHafter' in df.columns:
            ph_after = df.loc[idx, 'pHafter']
            if isinstance(ph_after, (int, float)) and not pd.isna(ph_after):
                final_ph = (final_ph + float(ph_after)) / 2.0
        
        # 设置最终pH值
        df.loc[idx, 'pH_of_condition_enviroment'] = final_ph
        
        # 步骤4: 氯离子检测
        chloride_keywords = ['cl', 'chloride', 'nacl', 'cacl2', 'mgcl2', 'salt', 'seawater', 'sea water', 'artificial sea water']
        chloride_columns = ['ingredient', 'ingredient.1', 'cycle_ingredient', 'cycle_pH', 'pHafter', 'concrete', 'note_of_concrete']
        
        for col in chloride_columns:
            if col in df.columns:
                col_value = df.loc[idx, col]
                if col_value is not None and not pd.isna(col_value):
                    col_text = str(col_value).lower()
                    if any(keyword in col_text for keyword in chloride_keywords):
                        df.loc[idx, 'Chloride_ion'] = 1
                        break
    
    def _process_concrete_indicator(self, df: pd.DataFrame, idx: int):
        """
        📋 特征3: concrete (混凝土环境指示器)
        
        🔍 提取逻辑：
        - 检查列：['concrete', 'crack', 'cover']
        - 任一列有非空值 → concrete = 1
        - 全部为空 → concrete = 0
        
        📊 处理规则：
        - 字符串类型：任何非空字符串 → 1
        - 数值类型：非NaN → 1
        - 空值/NaN → 0
        """
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
        """
        📋 特征4: diameter (纤维直径)
        
        🔍 源列说明：
        源列是原始数据文件中实际存在的列名，我们从这些列中读取原始数据
        
        📊 diameter特征的源列映射：
        
        源列名称                    数据类型        处理方式               目标特征
        ├─ 'diameter'         →    数值型    →    直接复制          →    'diameter'
        ├─ 'nominal_area'     →    数值型    →    几何计算          →    'diameter'
        └─ 'average_area'     →    数值型    →    几何计算          →    'diameter'
        
        🔧 具体处理逻辑：
        1. 优先源列：'diameter' 
           - 如果存在且有数值 → 直接使用
           - 示例：diameter列有值12.5 → 目标特征diameter = 12.5
        
        2. 备用源列1：'nominal_area'
           - 如果diameter列为空，但nominal_area列有值 → 几何计算
           - 公式：diameter = 2 × √(area/π)  
           - 示例：nominal_area = 78.5 → diameter = 2×√(78.5/π) ≈ 10.0
           
        3. 备用源列2：'average_area'
           - 如果前两个都为空，但average_area列有值 → 几何计算
           - 公式：diameter = 2 × √(area/π)  
           - 示例：average_area = 78.5 → diameter = 2×√(78.5/π) ≈ 10.0
        
        💡 为什么需要源列概念？
        - 原始数据列名可能不统一：有的叫'diameter'，有的叫'dia'，有的叫'size'
        - 有些特征需要从多个源列中提取：比如pH可能分布在5-6个不同的列中
        - 有些特征需要计算：比如从面积计算直径
        """
        
        # 🎯 定义源列查找顺序
        diameter_source_columns = [
            'diameter',        # 源列1：直接直径测量值（优先级最高）
            'nominal_area',    # 源列2：横截面积，需要计算转换
            'average_area'     # 源列3：平均横截面积，需要计算转换
        ]
        
        print(f"  🔍 查找diameter源列: {diameter_source_columns}")
        
        # 优先使用直接测量的直径（源列1）
        if 'diameter' in df.columns:
            diameter_value = df.loc[idx, 'diameter']
            print(f"    源列 'diameter' 的值: {diameter_value}")
            if isinstance(diameter_value, (int, float)) and not pd.isna(diameter_value):
                df.loc[idx, 'diameter'] = diameter_value
                print(f"    ✅ 使用直接测量值: {diameter_value}")
                return
        
        # 从nominal_area计算直径（源列2）
        if 'nominal_area' in df.columns:
            area_value = df.loc[idx, 'nominal_area']
            print(f"    源列 'nominal_area' 的值: {area_value}")
            if isinstance(area_value, (int, float)) and not pd.isna(area_value) and area_value > 0:
                calculated_diameter = 2 * np.sqrt(area_value / np.pi)
                df.loc[idx, 'diameter'] = calculated_diameter
                print(f"    ✅ 从nominal_area计算: area={area_value} → diameter={calculated_diameter:.3f}")
                return
        
        # 从average_area计算直径（源列3）
        if 'average_area' in df.columns:
            area_value = df.loc[idx, 'average_area']
            print(f"    源列 'average_area' 的值: {area_value}")
            if isinstance(area_value, (int, float)) and not pd.isna(area_value) and area_value > 0:
                calculated_diameter = 2 * np.sqrt(area_value / np.pi)
                df.loc[idx, 'diameter'] = calculated_diameter
                print(f"    ✅ 从average_area计算: area={area_value} → diameter={calculated_diameter:.3f}")
                return
    
    def _process_fiber_content(self, df: pd.DataFrame, idx: int):
        """
        📋 特征6: fiber_content (纤维含量)
        
        🔍 源列详细说明：
        
        📊 纤维含量的源列体系：
        
        源列名称                     数据内容           单位        处理方式
        ├─ 'Fiber_content_weight' → 重量百分比    →    %     →    直接使用（优先）
        └─ 'Fiber_content_volume' → 体积百分比    →    %     →    密度转换
        
        🔧 为什么需要两个源列？
        1. 纤维含量有两种表示方法：
           - 重量百分比：纤维重量 / 总重量 × 100%
           - 体积百分比：纤维体积 / 总体积 × 100%
        
        2. 转换公式（体积% → 重量%）：
           weight% = (100 × Vf × ρf) / (Vf × ρf + (100-Vf) × ρm)
           其中：Vf=体积分数, ρf=纤维密度, ρm=基体密度
        
        📋 密度数据来源（需要额外的源列）:
        - 纤维类型源列：'Fiber_type' → 确定纤维密度
        - 基体类型源列：'Matrix_type' → 确定基体密度
        
        💡 源列依赖关系：
        主源列: Fiber_content_volume
        辅助源列: Fiber_type, Matrix_type (用于密度查找)
        """
        
        # 🎯 定义源列系统
        primary_source_columns = {
            'weight_percent': 'Fiber_content_weight',  # 主源列1：重量百分比
            'volume_percent': 'Fiber_content_volume'   # 主源列2：体积百分比
        }
        
        auxiliary_source_columns = {
            'fiber_type': 'Fiber_type',      # 辅助源列1：纤维类型（用于密度）
            'matrix_type': 'Matrix_type'     # 辅助源列2：基体类型（用于密度）
        }
        
        print(f"  🔍 纤维含量源列系统:")
        print(f"    主源列: {list(primary_source_columns.values())}")
        print(f"    辅助源列: {list(auxiliary_source_columns.values())}")
        
        # 优先使用重量百分比（主源列1）
        weight_col = primary_source_columns['weight_percent']
        if weight_col in df.columns:
            weight_content = df.loc[idx, weight_col]
            print(f"    源列 '{weight_col}' 的值: {weight_content}")
            if isinstance(weight_content, (int, float)) and not pd.isna(weight_content):
                df.loc[idx, 'fiber_content'] = weight_content
                print(f"    ✅ 使用重量百分比: {weight_content}%")
                return
        
        # 从体积百分比转换（主源列2 + 辅助源列）
        volume_col = primary_source_columns['volume_percent']
        if volume_col in df.columns:
            volume_content = df.loc[idx, volume_col]
            print(f"    源列 '{volume_col}' 的值: {volume_content}")
            if isinstance(volume_content, (int, float)) and not pd.isna(volume_content):
                # 获取材料类型（从辅助源列）
                fiber_type = df.loc[idx, auxiliary_source_columns['fiber_type']] if auxiliary_source_columns['fiber_type'] in df.columns else 'Unknown'
                matrix_type = df.loc[idx, auxiliary_source_columns['matrix_type']] if auxiliary_source_columns['matrix_type'] in df.columns else 'Unknown'
                
                print(f"    辅助源列数据: fiber_type={fiber_type}, matrix_type={matrix_type}")
                
                # 密度查找
                density_fiber = self.material_props['fiber_densities'].get(fiber_type, 2.0)
                density_matrix = self.material_props['matrix_densities'].get(matrix_type, 1.2)
                
                # 体积转重量计算
                weight_content = (100.0 * volume_content * density_fiber) / (
                    volume_content * density_fiber + (100.0 - volume_content) * density_matrix
                )
                df.loc[idx, 'fiber_content'] = weight_content
                print(f"    ✅ 体积转重量: {volume_content}%(vol) → {weight_content:.2f}%(wt)")

    def _process_material_types(self, df: pd.DataFrame, idx: int):
        """
        📋 特征7&8: Glass_or_Basalt & Vinyl_ester_or_Epoxy
        
        🔍 提取逻辑：
        1. Glass_or_Basalt (纤维类型编码)：
           - 检查列：'Fiber_type'
           - 'Glass' → 1
           - 'Basalt' → 0
           - 其他 → NaN
        
        2. Vinyl_ester_or_Epoxy (基体类型编码)：
           - 检查列：'Matrix_type'  
           - 'Vinyl ester' → 1
           - 'Epoxy' → 0
           - 其他 → NaN
        
        📊 编码规则：
        - 二进制编码：特定材料=1，对应材料=0
        - 严格匹配：只处理明确的材料类型
        """
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
        """
        📋 特征9: surface_treatment (表面处理)
        
        🔍 提取逻辑：
        - 检查列：'surface_treatment'
        - 'sand coated' → 0 (粗糙表面)
        - 'Smooth' → 1 (光滑表面)
        - 其他 → NaN
        
        📊 编码规则：
        - 光滑表面 = 1
        - 粗糙表面 = 0
        - 表面处理类型影响粘结性能
        """
        if 'surface_treatment' in df.columns:
            treatment = df.loc[idx, 'surface_treatment']
            if treatment == 'sand coated':
                df.loc[idx, 'surface_treatment'] = 0
            elif treatment == 'Smooth':
                df.loc[idx, 'surface_treatment'] = 1
    
    def _process_other_features(self, df: pd.DataFrame, idx: int):
        """
        📋 特征10-14: 其他特征直接映射
        
        🔍 提取逻辑：
        1. condition_time ← 'time_field' (暴露时间)
        2. Temperature ← 'temperature' (暴露温度)  
        3. Tensile_strength_retention ← 'retention1' (强度保持率)
        4. max_strength ← 'Value1_1' (初始强度)
        5. glass_transition_temperature ← 'glass_transition_temperature' (玻璃化转变温度)
        
        📊 处理规则：
        - 数值列：验证为有效数值后直接复制
        - 字符串数值：尝试转换为float
        - 特殊列(Target_parameter)：直接复制原始值
        - 转换失败：保持NaN
        """
        # 直接复制的特征映射
        feature_mappings = {
            'condition_time': 'time_field',           # 暴露时间 (天/小时)
            'Temperature': 'temperature',             # 暴露温度 (°C)
            'Tensile_strength_retention': 'retention1',  # 强度保持率 (0-1)
            'Target_parameter': 'Target_parameter',   # 目标参数类型
            'max_strength': 'Value1_1',              # 初始强度 (MPa)
            'glass_transition_temperature': 'glass_transition_temperature'  # Tg (°C)
        }
        
        for new_col, old_col in feature_mappings.items():
            if old_col in df.columns:
                value = df.loc[idx, old_col]
                
                # Target_parameter特殊处理：直接复制
                if new_col == 'Target_parameter':
                    df.loc[idx, new_col] = value
                
                # 数值特征：验证后复制
                elif isinstance(value, (int, float)) and not pd.isna(value):
                    df.loc[idx, new_col] = value
                
                # 字符串数值：尝试转换
                elif isinstance(value, str) and value.strip() != '':
                    try:
                        numeric_value = float(value)
                        if not np.isnan(numeric_value):
                            df.loc[idx, new_col] = numeric_value
                    except (ValueError, TypeError):
                        # 转换失败：某些字段保留原始值
                        if new_col in ['Target_parameter']:
                            df.loc[idx, new_col] = value
    
    def _process_sample_weight(self, df: pd.DataFrame, idx: int):
        """
        📋 特征15: sample_weight (样品权重)
        
        🔍 源列说明：
        - 源列：CS列（WPS中显示为cs列）
        - 第3行和第4行：分别是'Mechanical_result'和'num'
        - 提取样品个数，权重 = ln(样品个数)
        
        📊 处理逻辑：
        1. 查找CS列（或类似的列名）
        2. 提取数字值作为样品个数
        3. 如果是'notreported'或'smd'，默认为1
        4. 权重 = ln(样品个数)
        
        💡 权重意义：
        - 样品数量多的数据点更可靠，权重更高
        - 使用对数函数避免权重差异过大
        """
        
        # 查找可能的样品数量列
        possible_columns = []
        for col in df.columns:
            col_str = str(col).lower()
            # 查找包含样品、num、count、cs等关键词的列
            if any(keyword in col_str for keyword in ['cs', 'num', 'count', 'sample', 'specimen']):
                possible_columns.append(col)
        
        sample_count = 1  # 默认样品数
        
        # 查找CS列或类似列
        target_col = None
        for col in possible_columns:
            if str(col).lower() in ['cs', 'num', 'count']:
                target_col = col
                break
        
        if target_col is None and possible_columns:
            target_col = possible_columns[0]  # 使用第一个匹配的列
        
        if target_col and target_col in df.columns:
            value = df.loc[idx, target_col]
            
            if value is not None and not pd.isna(value):
                value_str = str(value).lower().strip()
                
                # 检查是否为缺失值标记
                if value_str in ['notreported', 'not reported', 'smd', 'nan', '']:
                    sample_count = 1
                else:
                    # 尝试提取数字
                    import re
                    numbers = re.findall(r'\d+', value_str)
                    if numbers:
                        try:
                            sample_count = int(numbers[0])
                            if sample_count <= 0:
                                sample_count = 1
                        except (ValueError, TypeError):
                            sample_count = 1
                    else:
                        # 如果是纯数字
                        try:
                            sample_count = int(float(value))
                            if sample_count <= 0:
                                sample_count = 1
                        except (ValueError, TypeError):
                            sample_count = 1
        
        # 计算权重：ln(样品数量)
        sample_weight = np.log(max(sample_count, 1))  # 确保至少为1避免log(0)
        df.loc[idx, 'sample_weight'] = sample_weight
        
        print(f"    样品数量: {sample_count}, 权重: {sample_weight:.3f}")
    
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
            'surface_treatment', 'max_strength', 'sample_weight'
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
            'Vinyl_ester_or_Epoxy', 'surface_treatment', 'max_strength', 'sample_weight'
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
            'Matrix type', 'Surface treatment', 'Strength of unconditioned rebar', 'Sample weight'
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