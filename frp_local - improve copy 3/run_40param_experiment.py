#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP钢筋耐久性预测 - 40参数优化实验 (预定义特征版)

特点：
1. 三种机器学习模型：RandomForest, XGBoost, LightGBM
2. 每个模型约13-14种参数配置，总计40个配置
3. 5折交叉验证
4. 基于database 4.xlsx真实结构的特征提取
5. 使用13个预定义工程特征进行训练
6. 快速高效的参数搜索策略
7. 增强的错误处理和进度跟踪
"""

import pandas as pd
import numpy as np
import re  # 添加缺失的import
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
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
    """有效数据加载器 - 基于database 4.xlsx的真实结构"""
    
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
                print("⚠️  未找到数据文件，将创建模拟数据进行演示")
                self.file_path = None
                self.use_mock_data = True
                return
        else:
            self.file_path = file_path
        
        self.use_mock_data = False
    
    def load_valid_data(self):
        """加载有效数据 - 基于database 4.xlsx格式"""
        
        if self.use_mock_data:
            return self._create_mock_data()
        
        print("🔄 加载database 4.xlsx文件...")
        try:
            # 使用dataset code的读取方式：从第4行开始作为标题
            raw_data = pd.read_excel(self.file_path, header=3, engine='openpyxl')
            print(f"✅ Excel读取成功，数据形状: {raw_data.shape}")
            
        except Exception as e:
            print(f"❌ 读取数据文件失败: {e}")
            print("将使用模拟数据继续实验...")
            return self._create_mock_data()
        
        # 检查第一列（feature_name）来过滤数据
        # 保留第一列不为0的数据（相当于Comments=1的逻辑）
        if len(raw_data) > 0:
            first_col = raw_data.iloc[:, 0]  # feature_name列
            
            # 过滤：排除第一列为0的行，但保留NaN和其他值
            mask = ~(pd.to_numeric(first_col, errors='coerce') == 0)
            valid_data = raw_data[mask].copy()
            
            print(f"🎯 数据筛选结果:")
            print(f"  原始数据: {len(raw_data)} 行")
            print(f"  第一列非0: {mask.sum()} 行")
            print(f"  最终保留比例: {len(valid_data)/len(raw_data)*100:.1f}%")
        else:
            valid_data = raw_data.copy()
        
        # 数据质量检查
        print(f"📊 数据质量检查:")
        print(f"  有效数据形状: {valid_data.shape}")
        print(f"  缺失值总数: {valid_data.isnull().sum().sum()}")
        
        return valid_data
    
    def _create_mock_data(self):
        """创建模拟数据用于演示"""
        print("📝 创建模拟数据...")
        
        np.random.seed(42)
        n_samples = 500
        
        # 创建134列的模拟数据（匹配database 4.xlsx结构）
        mock_data = pd.DataFrame()
        
        # 添加所有需要的列（按真实位置）
        column_data = {}
        
        # 基础列
        column_data[0] = ['FRP_' + str(i) for i in range(1, n_samples + 1)]  # feature_name
        column_data[1] = ['Title_' + str(i) for i in range(n_samples)]  # Title
        column_data[5] = np.random.randint(2000, 2024, n_samples)  # Year
        
        # Rebar信息 - 修正索引，并添加一些SMD和Notreported值用于测试
        column_data[8] = np.random.choice(['Glass', 'Basalt', 'Carbon', 'SMD'], n_samples, p=[0.5, 0.3, 0.15, 0.05])  # 第9列 Fiber_type (索引8)
        column_data[10] = np.random.choice(['Vinyl ester', 'Epoxy', 'Polyester', 'Notreported'], n_samples, p=[0.4, 0.4, 0.15, 0.05])  # 第11列 Matrix_type (索引10)
        
        # 添加一些数值列包含SMD
        tg_values = np.random.normal(120, 20, n_samples).astype(str)
        smd_mask = np.random.random(n_samples) < 0.1  # 10%的SMD
        tg_values[smd_mask] = 'SMD'
        column_data[12] = tg_values  # glass_transition_temperature
        
        # 纤维含量，添加一些Notreported
        fiber_values = np.random.normal(60, 10, n_samples).astype(str)
        notreported_mask = np.random.random(n_samples) < 0.08  # 8%的Notreported
        fiber_values[notreported_mask] = 'Notreported'
        column_data[15] = fiber_values  # Fiber_content_weight
        
        column_data[16] = np.random.normal(50, 8, n_samples)  # Fiber_content_volume (索引16)
        column_data[18] = np.random.normal(10, 2, n_samples)  # 第19列 diameter (索引18)
        column_data[20] = np.random.normal(78.5, 10, n_samples)  # 第21列 nominal_area (索引20)
        column_data[22] = np.random.choice(['Smooth', 'sand coated', 'ribbed'], n_samples)  # 第23列 surface_treatment (索引22)
        
        # 控制组机械性能
        column_data[34] = np.random.normal(1000, 200, n_samples)  # Value1
        
        # 环境条件
        column_data[49] = np.random.normal(25, 5, n_samples)  # temperature
        column_data[51] = np.random.normal(100, 30, n_samples)  # time
        column_data[53] = np.random.choice([0, 1], n_samples)  # concrete
        column_data[54] = np.random.normal(13, 1, n_samples)  # pH_of_concrete
        column_data[56] = np.random.choice([0, 1], n_samples)  # crack
        column_data[57] = np.random.choice([0, 1], n_samples)  # cover
        
        # 溶液条件
        column_data[59] = np.random.normal(7, 1, n_samples)  # pH
        column_data[60] = np.random.normal(7, 1, n_samples)  # pHafter
        column_data[61] = np.random.choice(['tap water', 'NaCl solution', 'sea water'], n_samples)  # ingredient
        
        # 载荷条件
        column_data[88] = np.random.choice(['stress', 'strain', 'no load'], n_samples)  # stress_or_strain
        column_data[89] = np.random.choice(['static', 'cyclic', 'preloading'], n_samples)  # type_of_load
        column_data[90] = np.random.normal(0.5, 0.2, n_samples)  # value_load
        column_data[91] = np.random.normal(800, 100, n_samples)  # ultimate_tensile_strength
        
        # 机械结果
        column_data[100] = np.random.normal(0.8, 0.1, n_samples)  # retention1
        column_data[104] = np.random.normal(0.75, 0.12, n_samples)  # retention2
        column_data[108] = np.random.normal(0.7, 0.15, n_samples)  # retention3
        column_data[114] = np.random.normal(115, 25, n_samples)  # glass_transition_temperature_result
        
        # 创建完整的DataFrame
        max_col = max(column_data.keys()) + 1
        for col_idx in range(max_col):
            if col_idx in column_data:
                mock_data[col_idx] = column_data[col_idx]
            else:
                mock_data[col_idx] = np.nan
        
        print(f"✅ 模拟数据创建完成: {mock_data.shape}")
        print(f"📋 关键位置验证:")
        print(f"   第9列 (索引8) Fiber_type: {np.array(column_data[8])[:5]}")
        print(f"   第11列 (索引10) Matrix_type: {np.array(column_data[10])[:5]}")
        print(f"   第13列 (索引12) Tg with SMD: {np.array(column_data[12])[:5]}")
        return mock_data
    
    def clean_data(self, df):
        """清理数据 - 仿照dataset code的处理方式"""
        print("🧹 正在清理数据...")
        
        df_clean = df.copy()
        
        # 处理特殊值 - 仿照dataset code
        special_values = ['SMD', 'Notreported', 'N/A', '', ' ', 'nan', 'NULL', 'None']
        for val in special_values:
            df_clean = df_clean.replace(val, np.nan)
        
        # 处理范围值到平均值 - 只对数值相关列进行处理
        numeric_positions = [12, 13, 14, 15, 16, 18, 19, 20, 49, 54, 55, 59, 60, 78, 90, 34, 35, 37, 38, 40, 41]
        
        for pos in numeric_positions:
            if pos < len(df_clean.columns):
                # 重置索引以避免索引不连续的问题
                for i in range(len(df_clean)):
                    value = df_clean.iloc[i, pos]
                    if isinstance(value, str):
                        # 检查是否包含逗号且无冒号（范围值）
                        if ',' in value and ':' not in value:
                            try:
                                # 提取数字并计算平均值
                                numbers = re.findall(r"\d+\.?\d*", value)
                                if numbers:
                                    new_value = np.mean([float(x) for x in numbers])
                                    if not np.isnan(new_value):
                                        df_clean.iloc[i, pos] = new_value
                            except (ValueError, TypeError):
                                continue
        
        print("✅ 数据清理完成")
        return df_clean
    
    def analyze_data_quality_and_generate_plots(self, data):
        """
        分析数据质量并生成13个特征的分布图
        包含SMD、Notreported和有效数据的统计
        """
        print("📊 开始数据质量分析和特征分布图生成...")
        print("=" * 80)
        
        # 创建图片保存目录
        plots_dir = Path("analysis_results") / "feature_distribution_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib字体 - 修复字体显示问题
        try:
            # 尝试设置中文字体
            import matplotlib.font_manager as fm
            
            # 查找系统中可用的中文字体
            font_list = [
                'Microsoft YaHei',      # 微软雅黑
                'SimHei',               # 黑体
                'SimSun',               # 宋体
                'KaiTi',                # 楷体
                'FangSong',             # 仿宋
                'DejaVu Sans',          # 备用英文字体
                'Arial Unicode MS'      # 备用字体
            ]
            
            available_font = None
            for font_name in font_list:
                try:
                    font_path = fm.findfont(fm.FontProperties(family=font_name))
                    if font_path and Path(font_path).exists():
                        available_font = font_name
                        break
                except:
                    continue
            
            if available_font:
                plt.rcParams['font.sans-serif'] = [available_font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 使用字体: {available_font}")
            else:
                # 如果没有找到中文字体，使用英文标签
                plt.rcParams['font.family'] = 'DejaVu Sans'
                print("⚠️  未找到中文字体，将使用英文标签")
                
        except Exception as e:
            print(f"⚠️  字体设置失败: {e}，使用默认字体")
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 基于真实结构的13个特征位置映射（修正索引）
        feature_position_mapping = {
            'pH_of_condition_enviroment': {
                'positions': [54, 59, 60],  # pH_of_concrete, pH, pHafter
                'names': ['pH_of_concrete', 'pH', 'pHafter'],
                'description': 'Environmental pH Values'
            },
            'Chloride_ion': {
                'positions': [61, 64, 77],  # ingredient, ingredient_moisture, cycle_ingredient
                'names': ['ingredient', 'ingredient_moisture', 'cycle_ingredient'],
                'description': 'Chloride Ion Related Columns'
            },
            'concrete': {
                'positions': [53, 56, 57],  # concrete, crack, cover
                'names': ['concrete', 'crack', 'cover'],
                'description': 'Concrete Environment'
            },
            'diameter': {
                'positions': [18, 20],  # diameter, nominal_area
                'names': ['diameter', 'nominal_area'],
                'description': 'Fiber Diameter'
            },
            'load_value': {
                'positions': [90],  # value
                'names': ['value_load'],
                'description': 'Load Value'
            },
            'fiber_content': {
                'positions': [15, 16],  # Fiber_content_weight, Fiber_content_volume
                'names': ['Fiber_content_weight', 'Fiber_content_volume'],
                'description': 'Fiber Content'
            },
            'Glass_or_Basalt': {
                'positions': [8],  # 第9列 Fiber_type (索引8)
                'names': ['Fiber_type'],
                'description': 'Fiber Type'
            },
            'Vinyl_ester_or_Epoxy': {
                'positions': [10],  # 第11列 Matrix_type (索引10)
                'names': ['Matrix_type'],
                'description': 'Matrix Type'
            },
            'condition_time': {
                'positions': [51, 84],  # time, time_in_cycle
                'names': ['time', 'time_in_cycle'],
                'description': 'Condition Time'
            },
            'Temperature': {
                'positions': [49, 69, 78],  # temperature, field_average_temperature, temp
                'names': ['temperature', 'field_average_temperature', 'temp'],
                'description': 'Temperature'
            },
            'Tensile_strength_retention': {
                'positions': [100, 104, 108],  # retention1, retention2, retention3
                'names': ['retention1', 'retention2', 'retention3'],
                'description': 'Tensile Strength Retention'
            },
            'surface_treatment': {
                'positions': [22],  # surface_treatment
                'names': ['surface_treatment'],
                'description': 'Surface Treatment'
            },
            'glass_transition_temperature': {
                'positions': [12, 114, 13],  # glass_transition_temperature, result, run_2
                'names': ['glass_transition_temperature', 'glass_transition_temperature_result', 'glass_transition_temperature_run_2'],
                'description': 'Glass Transition Temperature'
            }
        }
        
        # 为每个特征生成分布图
        for i, (feature_name, config) in enumerate(feature_position_mapping.items(), 1):
            print(f"\n📈 生成第{i}个特征分布图: {feature_name}")
            
            # 创建图形
            n_positions = len(config['positions'])
            fig, axes = plt.subplots(2, n_positions, figsize=(5*n_positions, 10))
            
            # 确保axes总是二维数组格式，便于统一处理
            if n_positions == 1:
                # 当只有一列时，axes是1维数组，需要转换为2维
                axes = axes.reshape(2, 1)
            elif n_positions > 1:
                # 多列时，axes已经是正确的2维格式
                pass
            
            fig.suptitle(f'{i:02d}. {feature_name} - {config["description"]}', fontsize=16, fontweight='bold')
            
            # 为每个位置生成数据质量分析
            for j, (pos, col_name) in enumerate(zip(config['positions'], config['names'])):
                if pos < len(data.columns):
                    col_data = data.iloc[:, pos]
                    
                    # 统计数据质量
                    quality_stats = self._analyze_column_quality(col_data, col_name)
                    
                    # 第一行：数据质量分布饼图
                    ax1 = axes[0, j]  # 现在axes总是2维的，可以安全使用
                    self._plot_data_quality_pie(ax1, quality_stats, col_name)
                    
                    # 第二行：有效数据分布图
                    ax2 = axes[1, j]  # 现在axes总是2维的，可以安全使用
                    self._plot_data_distribution(ax2, col_data, col_name, quality_stats)
                    
                else:
                    # 如果列不存在，显示空图
                    ax1 = axes[0, j] if len(config['positions']) > 1 else axes[0]
                    ax2 = axes[1, j] if len(config['positions']) > 1 else axes[1]
                    
                    ax1.text(0.5, 0.5, f'Column {pos}\nNot Found', ha='center', va='center', fontsize=12)
                    ax1.set_title(f'{col_name} (Position {pos})')
                    ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                    
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax2.set_xticks([])
                    ax2.set_yticks([])
            
            # 调整布局并保存
            plt.tight_layout()
            plot_path = plots_dir / f"{i:02d}_{feature_name}_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ 已保存: {plot_path}")
        
        # 生成总体数据质量报告
        self._generate_quality_summary_report(data, feature_position_mapping, plots_dir)
        
        print(f"\n🎉 数据质量分析完成！")
        print(f"📁 图片保存位置: {plots_dir}")
        print(f"📊 共生成 {len(feature_position_mapping)} 张特征分布图")
    
    def _plot_data_quality_pie(self, ax, quality_stats, col_name):
        """绘制数据质量饼图 - 修复显示问题"""
        labels = []
        sizes = []
        colors = []
        
        # 数据质量分类 - 使用英文标签避免字体问题
        categories = [
            ('Valid Data', quality_stats['valid_total'], '#2E8B57'),      # 海绿色
            ('SMD', quality_stats['smd'], '#FF6B6B'),                     # 红色
            ('Not Reported', quality_stats['notreported'], '#FFB347'),   # 橙色
            ('Missing/NaN', quality_stats['nan_empty'], '#87CEEB'),       # 天蓝色
        ]
        
        # 只添加数量大于0的类别
        for label, count, color in categories:
            if count > 0:
                percentage = count / quality_stats['total'] * 100
                labels.append(f'{label}\n({count}, {percentage:.1f}%)')
                sizes.append(count)
                colors.append(color)
        
        if len(sizes) > 0:
            # 绘制饼图
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90,
                textprops={'fontsize': 8}
            )
            
            # 美化百分比文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(7)
                
            # 美化标签文本
            for text in texts:
                text.set_fontsize(8)
                text.set_fontweight('normal')
                
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
        
        ax.set_title(f'{col_name}\nData Quality Distribution', fontsize=10, fontweight='bold')
    
    def _plot_data_distribution(self, ax, col_data, col_name, quality_stats):
        """绘制有效数据分布图 - 修复显示问题"""
        valid_data = quality_stats['valid_data']
        
        if len(valid_data) == 0:
            ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', fontsize=12)
            ax.set_title(f'{col_name} - Data Distribution')
            return
        
        # 尝试数值分布
        numeric_data = pd.to_numeric(valid_data, errors='coerce')
        numeric_valid = numeric_data.dropna()
        
        if len(numeric_valid) > 0 and len(numeric_valid) >= len(valid_data) * 0.5:  # 降低阈值到50%
            # 绘制数值分布直方图
            n_bins = min(15, max(5, len(numeric_valid.unique())))  # 动态调整bins数量
            
            try:
                ax.hist(numeric_valid, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
                ax.set_title(f'{col_name} - Numeric Distribution\n(n={len(numeric_valid)})', fontsize=9)
                ax.set_xlabel('Values', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                
                # 添加统计信息
                mean_val = numeric_valid.mean()
                std_val = numeric_valid.std()
                
                if not (np.isnan(mean_val) or np.isnan(std_val)):
                    stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)
                
                # 设置刻度标签字体大小
                ax.tick_params(axis='both', which='major', labelsize=7)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Plot Error:\n{str(e)[:50]}', ha='center', va='center', fontsize=8)
                
        else:
            # 绘制分类数据条形图
            value_counts = valid_data.value_counts().head(8)  # 减少显示数量
            
            if len(value_counts) > 0:
                try:
                    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                                 color='lightcoral', alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    ax.set_xticks(range(len(value_counts)))
                    
                    # 简化标签显示
                    labels = []
                    for x in value_counts.index:
                        label_str = str(x)
                        if len(label_str) > 8:
                            label_str = label_str[:8] + '..'
                        labels.append(label_str)
                    
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                    ax.set_title(f'{col_name} - Categorical Distribution\n(n={len(valid_data)}, Top {len(value_counts)})', fontsize=9)
                    ax.set_ylabel('Count', fontsize=8)
                    
                    # 在柱子上显示数值
                    for bar, count in zip(bars, value_counts.values):
                        height = bar.get_height()
                        if height > 0:  # 只在有高度的柱子上显示
                            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{count}', ha='center', va='bottom', fontsize=6)
                    
                    # 设置刻度标签字体大小
                    ax.tick_params(axis='both', which='major', labelsize=7)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Plot Error:\n{str(e)[:50]}', ha='center', va='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{col_name} - Data Distribution')

    def _analyze_column_quality(self, col_data, col_name):
        """分析单列的数据质量 - 修复检测逻辑"""
        total_count = len(col_data)
        
        # 转换为字符串进行分析
        str_data = col_data.astype(str).str.lower().str.strip()
        
        # 统计各种数据状态 - 修复检测逻辑
        smd_count = sum(str_data.isin(['smd']))
        notreported_count = sum(str_data.isin(['notreported', 'not reported', 'not_reported']))
        
        # NaN检测
        nan_count = sum(col_data.isna())
        
        # 空值检测
        empty_count = sum(str_data.isin(['', 'nan', 'none', 'null']))
        
        # 计算有效数据 - 排除所有无效类型
        invalid_values = ['smd', 'notreported', 'not reported', 'not_reported', '', 'nan', 'none', 'null']
        invalid_mask = str_data.isin(invalid_values) | col_data.isna()
        valid_count = total_count - invalid_mask.sum()
        
        # 分析有效数据的类型
        valid_data = col_data[~invalid_mask]
        numeric_count = 0
        text_count = 0
        
        if len(valid_data) > 0:
            numeric_data = pd.to_numeric(valid_data, errors='coerce')
            numeric_count = numeric_data.count()
            text_count = len(valid_data) - numeric_count
        
        # 调试信息
        print(f"    列 {col_name}: 总数={total_count}, SMD={smd_count}, NotReported={notreported_count}, NaN={nan_count}, Empty={empty_count}, Valid={valid_count}")
        
        return {
            'total': total_count,
            'smd': smd_count,
            'notreported': notreported_count,
            'nan_empty': nan_count + empty_count,
            'valid_total': valid_count,
            'valid_numeric': numeric_count,
            'valid_text': text_count,
            'valid_data': valid_data
        }
    
    def _generate_quality_summary_report(self, data, feature_mapping, plots_dir):
        """生成数据质量总结报告"""
        report_path = plots_dir / "data_quality_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("13个特征数据质量分析报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总样本数: {len(data)}\n\n")
            
            # 总体统计
            total_smd = 0
            total_notreported = 0
            total_valid = 0
            total_cells = 0
            
            for feature_name, config in feature_mapping.items():
                f.write(f"🎯 {feature_name} ({config['description']}):\n")
                f.write("-" * 50 + "\n")
                
                for pos, col_name in zip(config['positions'], config['names']):
                    if pos < len(data.columns):
                        col_data = data.iloc[:, pos]
                        quality_stats = self._analyze_column_quality(col_data, col_name)
                        
                        f.write(f"  📋 {col_name} (位置{pos}):\n")
                        f.write(f"     总数据: {quality_stats['total']}\n")
                        f.write(f"     SMD: {quality_stats['smd']} ({quality_stats['smd']/quality_stats['total']*100:.1f}%)\n")
                        f.write(f"     Notreported: {quality_stats['notreported']} ({quality_stats['notreported']/quality_stats['total']*100:.1f}%)\n")
                        f.write(f"     空值/NaN: {quality_stats['nan_empty']} ({quality_stats['nan_empty']/quality_stats['total']*100:.1f}%)\n")
                        f.write(f"     有效数据: {quality_stats['valid_total']} ({quality_stats['valid_total']/quality_stats['total']*100:.1f}%)\n")
                        f.write(f"       - 数值型: {quality_stats['valid_numeric']}\n")
                        f.write(f"       - 文本型: {quality_stats['valid_text']}\n")
                        f.write("\n")
                        
                        # 累计统计
                        total_smd += quality_stats['smd']
                        total_notreported += quality_stats['notreported']
                        total_valid += quality_stats['valid_total']
                        total_cells += quality_stats['total']
                    else:
                        f.write(f"  ❌ {col_name} (位置{pos}): 列不存在\n\n")
                
                f.write("\n")
            
            # 总体统计
            f.write("📊 总体统计:\n")
            f.write("=" * 30 + "\n")
            f.write(f"总数据单元格: {total_cells}\n")
            f.write(f"SMD总数: {total_smd} ({total_smd/total_cells*100:.2f}%)\n")
            f.write(f"Notreported总数: {total_notreported} ({total_notreported/total_cells*100:.2f}%)\n")
            f.write(f"有效数据总数: {total_valid} ({total_valid/total_cells*100:.2f}%)\n")
            f.write(f"数据缺失率: {(total_cells-total_valid)/total_cells*100:.2f}%\n")
        
        print(f"📄 数据质量报告已保存: {report_path}")

    def prepare_features_target(self, data):
        """
        基于database 4.xlsx真实结构的特征提取
        使用固定位置读取，结合app.py的处理逻辑
        """
        print("🔧 开始基于真实结构的特征提取...")
        print("=" * 80)
        
        # 先进行数据质量分析并生成分布图
        self.analyze_data_quality_and_generate_plots(data)
        
        # 然后进行正常的特征提取
        data_clean = self.clean_data(data)
        
        # 13个预定义特征的位置映射（修正索引）
        feature_position_mapping = {
            'pH_of_condition_enviroment': {
                'primary_positions': [54, 59, 60],  # 第55列, 第60列, 第61列
                'secondary_positions': [62, 70, 75, 76],  
                'type': 'numerical',
                'description': '环境条件pH值'
            },
            'Chloride_ion': {
                'primary_positions': [61, 64, 77],  # 第62列, 第65列, 第78列
                'type': 'categorical_binary',
                'description': '氯离子存在 (0/1)',
                'keywords': ['cl', 'chloride', 'nacl', 'cacl2', 'mgcl2', 'salt', 'seawater', 'sea water']
            },
            'concrete': {
                'primary_positions': [53, 56, 57],  # 第54列, 第57列, 第58列
                'type': 'categorical_binary',
                'description': '混凝土环境 (0/1)'
            },
            'diameter': {
                'primary_positions': [18],  # 第19列 diameter
                'secondary_positions': [20],  # 第21列 nominal_area
                'type': 'numerical',
                'description': '纤维直径 (mm)'
            },
            'load_value': {
                'primary_positions': [90],  # 第91列 value
                'auxiliary_positions': [88, 89, 91, 92],  
                'type': 'numerical',
                'description': '载荷值'
            },
            'fiber_content': {
                'primary_positions': [15],  # 第16列 Fiber_content_weight
                'secondary_positions': [16],  # 第17列 Fiber_content_volume
                'auxiliary_positions': [8, 10],  # 第9列 Fiber_type, 第11列 Matrix_type
                'type': 'numerical',
                'description': '纤维含量 (%)'
            },
            'Glass_or_Basalt': {
                'primary_positions': [8],  # 第9列 Fiber_type (索引8)
                'type': 'categorical_binary',
                'description': '纤维类型 Glass=1, Basalt=0'
            },
            'Vinyl_ester_or_Epoxy': {
                'primary_positions': [10],  # 第11列 Matrix_type (索引10)
                'type': 'categorical_binary',
                'description': '树脂类型 Vinyl_ester=1, Epoxy=0'
            },
            'condition_time': {
                'primary_positions': [51],  # 第52列 time
                'secondary_positions': [84],  # 第85列 time_in_cycle
                'type': 'numerical',
                'description': '条件时间 (天/小时)'
            },
            'Temperature': {
                'primary_positions': [49],  # 第50列 temperature
                'secondary_positions': [69, 78],  # field_average_temperature, temp
                'type': 'numerical',
                'description': '温度 (°C)'
            },
            'Tensile_strength_retention': {
                'primary_positions': [100],  # 第101列 retention1
                'secondary_positions': [104, 108],  # retention2, retention3
                'type': 'numerical',
                'description': '拉伸强度保持率 (0-1)'
            },
            'surface_treatment': {
                'primary_positions': [22],  # 第23列 surface_treatment
                'type': 'categorical_binary',
                'description': '表面处理 (0/1)'
            },
            'glass_transition_temperature': {
                'primary_positions': [12, 114],  # 第13列, 第115列
                'secondary_positions': [13],  # 第14列 glass_transition_temperature_run_2
                'type': 'numerical',
                'description': '玻璃化转变温度 (°C)'
            }
        }
        
        # 目标变量候选位置
        target_position_candidates = [100, 97, 34]  # retention1, Value1_result, Value1
        
        print(f"📋 基于真实位置的特征提取规则:")
        for feature_name, info in feature_position_mapping.items():
            print(f"  🎯 {feature_name}: 位置{info['primary_positions']} ({info['description']})")
        
        # 开始特征提取
        feature_data = {}
        found_features = []
        
        for feature_name, config in feature_position_mapping.items():
            print(f"\n🔍 提取特征: {feature_name}")
            
            success = False
            
            if config['type'] == 'numerical':
                success = self._extract_numerical_feature(
                    data_clean, feature_name, config, feature_data
                )
            elif config['type'] == 'categorical_binary':
                success = self._extract_categorical_feature(
                    data_clean, feature_name, config, feature_data
                )
            
            if success:
                found_features.append(feature_name)
                print(f"   ✅ 提取成功")
            else:
                print(f"   ❌ 提取失败")
        
        # 查找目标变量
        target_col = None
        target_data = None
        
        for pos in target_position_candidates:
            if pos < len(data_clean.columns):
                col_data = data_clean.iloc[:, pos]
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                
                if numeric_data.count() > len(data_clean) * 0.3:  # 至少30%有效数据
                    target_data = numeric_data
                    target_col = f"position_{pos}"
                    print(f"🎯 目标变量: 位置{pos} (有效数据: {numeric_data.count()}/{len(data_clean)})")
                    break
        
        if target_data is None:
            print("⚠️  未找到合适的目标变量，使用备用方法...")
            return self._fallback_extraction(data_clean)
        
        # 检查最小特征要求
        if len(found_features) < 3:
            print(f"⚠️  找到的特征太少({len(found_features)}个)，使用备用方法...")
            return self._fallback_extraction(data_clean)
        
        # 创建特征DataFrame
        X_raw = pd.DataFrame(feature_data)
        
        # 合并数据并清理缺失值
        combined = pd.concat([X_raw, target_data], axis=1)
        combined_clean = combined.dropna()
        
        print(f"\n📈 数据质量检查:")
        print(f"  原始数据: {len(data_clean)} 行")
        print(f"  特征矩阵: {X_raw.shape}")
        print(f"  清理后数据: {len(combined_clean)} 行")
        print(f"  数据保留率: {len(combined_clean)/len(data_clean)*100:.1f}%")
        
        if len(combined_clean) < 50:
            print(f"  ⚠️  清理后数据不足，使用备用方法...")
            return self._fallback_extraction(data_clean)
        
        # 分离特征和目标
        X_clean = combined_clean.iloc[:, :-1]
        y_clean = combined_clean.iloc[:, -1]
        
        # 重命名特征列
        feature_name_map = {col: f"feat_{i}_{col}" for i, col in enumerate(X_clean.columns)}
        X_clean = X_clean.rename(columns=feature_name_map)
        
        print(f"\n✅ 基于真实结构的特征提取成功:")
        print(f"  最终特征数: {X_clean.shape[1]}")
        print(f"  样本数: {len(X_clean)}")
        print(f"  目标变量: {target_col}")
        print(f"  目标范围: [{y_clean.min():.3f}, {y_clean.max():.3f}]")
        
        return X_clean, y_clean, list(X_clean.columns)
    
    def _extract_numerical_feature(self, data, feature_name, config, feature_data):
        """提取数值特征"""
        primary_positions = config['primary_positions']
        secondary_positions = config.get('secondary_positions', [])
        
        # 尝试主要位置
        for pos in primary_positions:
            if pos < len(data.columns):
                col_data = data.iloc[:, pos]
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                
                if numeric_data.count() > len(data) * 0.1:  # 至少10%有效数据
                    # 应用特征特定的处理逻辑
                    processed_data = self._process_numerical_by_feature(
                        numeric_data, feature_name, data, pos
                    )
                    feature_data[feature_name] = processed_data
                    return True
        
        # 尝试次要位置
        for pos in secondary_positions:
            if pos < len(data.columns):
                col_data = data.iloc[:, pos]
                
                if feature_name == 'diameter' and pos == 20:  # nominal_area计算直径
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if numeric_data.count() > 0:
                        # 从面积计算直径
                        diameter_data = 2 * np.sqrt(numeric_data / np.pi)
                        feature_data[feature_name] = diameter_data
                        return True
                
                elif feature_name == 'fiber_content' and pos == 16:  # 体积转重量
                    volume_data = pd.to_numeric(col_data, errors='coerce')
                    if volume_data.count() > 0:
                        # 简化转换（实际需要密度信息）
                        weight_data = volume_data * 0.8  # 简化换算系数
                        feature_data[feature_name] = weight_data
                        return True
                
                else:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if numeric_data.count() > len(data) * 0.1:
                        processed_data = self._process_numerical_by_feature(
                            numeric_data, feature_name, data, pos
                        )
                        feature_data[feature_name] = processed_data
                        return True
        
        return False
    
    def _extract_categorical_feature(self, data, feature_name, config, feature_data):
        """提取分类特征"""
        primary_positions = config['primary_positions']
        
        if feature_name == 'Chloride_ion':
            # 氯离子：关键词搜索
            keywords = config['keywords']
            chloride_found = pd.Series(0, index=data.index)
            
            for pos in primary_positions:
                if pos < len(data.columns):
                    col_data = data.iloc[:, pos].astype(str).str.lower()
                    for keyword in keywords:
                        mask = col_data.str.contains(keyword, na=False)
                        chloride_found.loc[mask] = 1
            
            feature_data[feature_name] = chloride_found
            return True
        
        elif feature_name == 'concrete':
            # 混凝土环境：任一列有值
            concrete_indicator = pd.Series(0, index=data.index)
            
            for pos in primary_positions:
                if pos < len(data.columns):
                    col_data = data.iloc[:, pos]
                    has_value = col_data.notna() & (col_data != '') & (col_data != 0)
                    concrete_indicator.loc[has_value] = 1
            
            feature_data[feature_name] = concrete_indicator
            return True
        
        elif feature_name in ['Glass_or_Basalt', 'Vinyl_ester_or_Epoxy']:
            # 材料类型编码
            pos = primary_positions[0]
            if pos < len(data.columns):
                col_data = data.iloc[:, pos].astype(str).str.lower()
                
                if feature_name == 'Glass_or_Basalt':
                    encoded = pd.Series(0, index=data.index)
                    glass_mask = col_data.str.contains('glass', na=False)
                    encoded.loc[glass_mask] = 1
                else:  # Vinyl_ester_or_Epoxy
                    encoded = pd.Series(0, index=data.index)
                    vinyl_mask = col_data.str.contains('vinyl', na=False)
                    encoded.loc[vinyl_mask] = 1
                
                feature_data[feature_name] = encoded
                return True
        
        elif feature_name == 'surface_treatment':
            # 表面处理
            pos = primary_positions[0]
            if pos < len(data.columns):
                col_data = data.iloc[:, pos].astype(str).str.lower()
                
                encoded = pd.Series(0, index=data.index)
                smooth_mask = col_data.str.contains('smooth', na=False)
                encoded.loc[smooth_mask] = 1
                
                feature_data[feature_name] = encoded
                return True
        
        return False
    
    def _process_numerical_by_feature(self, numeric_data, feature_name, data, position):
        """根据特征类型进行数值处理"""
        if feature_name == 'pH_of_condition_enviroment':
            # pH处理：包含混凝土环境逻辑
            if position == 54:  # pH_of_concrete
                # 混凝土环境，默认值13.0
                return numeric_data.fillna(13.0).clip(0, 14)
            else:
                # 溶液环境，默认值7.0
                return numeric_data.fillna(7.0).clip(0, 14)
        
        elif feature_name == 'fiber_content':
            # 纤维含量：转换为0-1范围
            if numeric_data.max() > 1:
                return (numeric_data / 100).clip(0, 1)
            return numeric_data.clip(0, 1)
        
        elif feature_name == 'Tensile_strength_retention':
            # 强度保持率：转换为0-1范围
            if numeric_data.max() > 1:
                return (numeric_data / 100).clip(0, 1.2)
            return numeric_data.clip(0, 1.2)
        
        elif feature_name == 'Temperature':
            # 温度：华氏度转换
            if numeric_data.max() > 100:
                fahrenheit_mask = numeric_data > 50
                numeric_data.loc[fahrenheit_mask] = (numeric_data.loc[fahrenheit_mask] - 32) * 5/9
            return numeric_data.clip(-50, 200)
        
        elif feature_name == 'condition_time':
            # 时间：小时转天
            if numeric_data.max() > 365:
                return (numeric_data / 24).clip(0, 10000)
            return numeric_data.clip(0, 10000)
        
        else:
            # 其他数值特征：基本处理
            return numeric_data
    
    def _fallback_extraction(self, data):
        """备用特征提取方法"""
        print("🔄 使用备用特征提取...")
        
        # 基本特征位置
        basic_features = {
            'temperature': 49,
            'time': 51,
            'diameter': 18,
            'fiber_content': 15,
            'retention': 100
        }
        
        feature_data = {}
        found_count = 0
        
        for feat_name, pos in basic_features.items():
            if pos < len(data.columns):
                col_data = data.iloc[:, pos]
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if numeric_data.count() > len(data) * 0.1:
                    feature_data[f"feat_{found_count}_{feat_name}"] = numeric_data
                    found_count += 1
        
        # 目标变量
        target_data = pd.to_numeric(data.iloc[:, 100], errors='coerce')  # retention1
        
        if len(feature_data) >= 3 and target_data.count() > len(data) * 0.3:
            X_simple = pd.DataFrame(feature_data)
            
            # 清理数据
            combined = pd.concat([X_simple, target_data], axis=1)
            combined_clean = combined.dropna()
            
            if len(combined_clean) >= 50:
                X_clean = combined_clean.iloc[:, :-1]
                y_clean = combined_clean.iloc[:, -1]
                
                print(f"✅ 备用特征提取成功:")
                print(f"  特征数: {X_clean.shape[1]}")
                print(f"  样本数: {len(X_clean)}")
                
                return X_clean, y_clean, list(X_clean.columns)
        
        return None

def get_40_parameter_configs():
    """获取40个快速参数配置"""
    
    configs = []
    
    # RandomForest 参数配置 (15个)
    rf_configs = [
        {'n_estimators': 50, 'max_depth': 5, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 7, 'random_state': 42},
        {'n_estimators': 150, 'max_depth': 10, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': None, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 5, 'random_state': 42},
        {'n_estimators': 150, 'max_depth': 5, 'min_samples_split': 10, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': 7, 'min_samples_leaf': 2, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 10, 'max_features': 'sqrt', 'random_state': 42},
        {'n_estimators': 250, 'max_depth': 5, 'max_features': 'log2', 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 8, 'min_samples_split': 3, 'random_state': 42},
        {'n_estimators': 80, 'max_depth': 6, 'min_samples_leaf': 4, 'random_state': 42},
        {'n_estimators': 120, 'max_depth': 12, 'max_features': None, 'random_state': 42},
        {'n_estimators': 180, 'max_depth': 4, 'min_samples_split': 8, 'random_state': 42},
        {'n_estimators': 350, 'max_depth': 9, 'min_samples_leaf': 1, 'random_state': 42},
        {'n_estimators': 400, 'max_depth': 15, 'max_features': 'sqrt', 'random_state': 42}
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
            {'n_estimators': 180, 'max_depth': 5, 'learning_rate': 0.12, 'random_state': 42},
            {'n_estimators': 320, 'max_depth': 4, 'learning_rate': 0.07, 'random_state': 42},
            {'n_estimators': 120, 'max_depth': 9, 'learning_rate': 0.09, 'random_state': 42}
        ]
        
        for i, config in enumerate(xgb_configs):
            configs.append({
                'model': 'XGBoost',
                'config_id': i + 1,
                'config': config
            })
    
    # LightGBM 参数配置 (12个)
    if LIGHTGBM_AVAILABLE:
        lgb_configs = [
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 63, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'num_leaves': 15, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.2, 'num_leaves': 127, 'random_state': 42},
            {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'num_leaves': 7, 'random_state': 42},
            {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.08, 'num_leaves': 31, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 255, 'random_state': 42},
            {'n_estimators': 180, 'max_depth': 6, 'learning_rate': 0.12, 'num_leaves': 63, 'random_state': 42},
            {'n_estimators': 220, 'max_depth': 4, 'learning_rate': 0.07, 'num_leaves': 15, 'random_state': 42},
            {'n_estimators': 350, 'max_depth': 5, 'learning_rate': 0.06, 'num_leaves': 31, 'random_state': 42},
            {'n_estimators': 120, 'max_depth': 7, 'learning_rate': 0.15, 'num_leaves': 127, 'random_state': 42},
            {'n_estimators': 280, 'max_depth': 6, 'learning_rate': 0.04, 'num_leaves': 63, 'random_state': 42}
        ]
        
        for i, config in enumerate(lgb_configs):
            configs.append({
                'model': 'LightGBM',
                'config_id': i + 1,
                'config': config
            })
    
    return configs

def train_and_evaluate_config(config_info, X, y, cv_folds=5):
    """训练和评估单个配置 - 完全修复版"""
    model_name = config_info['model']
    config = config_info['config']
    config_id = config_info['config_id']
    
    try:
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 创建模型 - 完全移除early stopping
        if model_name == 'RandomForest':
            model = RandomForestRegressor(**config)
            X_for_training = X_scaled  # RandomForest使用标准化数据
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            # 简化配置，完全移除early stopping相关参数
            config_xgb = config.copy()
            config_xgb.update({
                'verbosity': 0,
                'objective': 'reg:squarederror',
                'enable_categorical': False
            })
            model = xgb.XGBRegressor(**config_xgb)
            X_for_training = X  # XGBoost使用原始数据
        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            # 简化配置，完全移除early stopping相关参数
            config_lgb = config.copy()
            config_lgb.update({
                'verbosity': -1,
                'objective': 'regression',
                'metric': 'rmse',
                'force_col_wise': True
            })
            
            # 检查和修复num_leaves参数
            if 'num_leaves' in config_lgb:
                max_depth = config_lgb.get('max_depth', 6)
                if max_depth is not None:
                    max_leaves = 2 ** max_depth
                    if config_lgb['num_leaves'] > max_leaves:
                        config_lgb['num_leaves'] = max_leaves
                        print(f"⚠️  调整num_leaves从{config['num_leaves']}到{max_leaves}")
            
            model = lgb.LGBMRegressor(**config_lgb)
            X_for_training = X  # LightGBM使用原始数据
        else:
            print(f"❌ 未知模型类型或模块不可用: {model_name}")
            return None
        
        # 交叉验证
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        print(f"   🔄 开始交叉验证...")
        cv_scores = cross_val_score(model, X_for_training, y, cv=kfold, scoring='r2')
        print(f"   ✅ 交叉验证完成: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # 训练测试分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_for_training, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        print(f"   🔄 开始训练模型...")
        start_time = time.time()
        
        # 直接训练，不使用任何early stopping
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"   ✅ 模型训练完成，用时: {training_time:.2f}s")
        
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
        
        print(f"   ✅ 评估完成: R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
        return result
        
    except Exception as e:
        import traceback
        print(f"❌ 配置 {model_name} #{config_id} 训练失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        print(f"   配置参数: {config}")
        
        # 打印详细的错误堆栈（调试用）
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"   根本原因: {e.__cause__}")
        
        # 输出前几行堆栈信息
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines[-10:]:  # 只显示最后10行
            if line.strip():
                print(f"   {line}")
        
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
    print("🚀 开始大规模参数优化实验 (预定义特征版)")
    print("=" * 60)
    
    # 检查并创建必要目录
    Path("analysis_results").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    
    print("📋 RandomForest: 15个配置")
    print("📋 XGBoost: 13个配置") 
    print("📋 LightGBM: 12个配置")
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
    print(f"\n🔬 开始运行大规模实验...")
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
        
        # 每50个配置保存一次
        if i % 50 == 0 and results:
            save_results(results, experiment_id)
            if not TQDM_AVAILABLE:
                print(f"   💾 已保存前 {len(results)} 个结果")
    
    # 5. 最终保存和分析
    total_time = time.time() - start_time
    print(f"\n📊 大规模实验完成！总用时: {total_time/60:.1f} 分钟")
    
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
        
        # TOP20 配置
        print(f"\n🥇 TOP20 配置:")
        top20 = df.nlargest(20, 'test_r2')
        for i, (_, row) in enumerate(top20.iterrows(), 1):
            print(f"   {i:2d}. {row['model']} #{row['config_id']}: "
                  f"R²={row['test_r2']:.6f}, RMSE={row['test_rmse']:.6f}")
        
        # 失败的配置统计
        if failed_configs:
            print(f"\n❌ 失败的配置: {len(failed_configs)} 个")
            # 只显示前10个失败配置
            for i, config in enumerate(failed_configs[:10], 1):
                print(f"   {i:2d}. {config}")
            if len(failed_configs) > 10:
                print(f"   ... 还有 {len(failed_configs) - 10} 个失败配置")
    
    else:
        print("❌ 没有成功的配置")
    
    print(f"\n🎉 大规模实验完成! 结果已保存到 experiments/ 目录")

if __name__ == "__main__":
    main()