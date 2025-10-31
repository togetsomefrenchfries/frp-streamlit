#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP钢筋耐久性预测 - 13个预定义特征数据分布分析

功能：
1. 分析13个预定义工程特征的原始数据分布
2. 对数值特征生成直方图和分布统计
3. 对分类特征生成条形图和频率统计
4. 处理SMD、NotReported等特殊值
5. 生成详细的数据质量报告
6. 保存所有图表到analysis_results文件夹

13个分析特征：
- pH_of_condition_enviroment, Chloride_ion, concrete
- diameter, load_value, fiber_content, Glass_or_Basalt
- Vinyl_ester_or_Epoxy, condition_time, Temperature
- Tensile_strength_retention, surface_treatment
- glass_transition_temperature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import json
import matplotlib

# 修复中文字体显示问题
def setup_chinese_fonts():
    """设置中文字体显示"""
    try:
        # 尝试设置中文字体
        import matplotlib.font_manager as fm
        
        # 查找系统中可用的中文字体
        font_list = [font.name for font in fm.fontManager.ttflist]
        
        # 常见的中文字体列表（按优先级排序）
        chinese_fonts = [
            'Microsoft YaHei',
            'SimHei', 
            'SimSun',
            'KaiTi',
            'FangSong',
            'STSong',
            'STKaiti',
            'STFangsong',
            'Dengxian',
            'PingFang SC',
            'Hiragino Sans GB',
            'WenQuanYi Micro Hei',
            'Source Han Sans CN',
            'Noto Sans CJK SC'
        ]
        
        # 寻找可用的中文字体
        available_font = None
        for font in chinese_fonts:
            if font in font_list:
                available_font = font
                break
        
        if available_font:
            # 设置字体
            plt.rcParams['font.sans-serif'] = [available_font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 中文字体设置成功: {available_font}")
        else:
            # 备用方案：使用英文标签
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            print("⚠️  未找到中文字体，将使用英文标签")
            return False
            
    except Exception as e:
        print(f"❌ 字体设置失败: {e}")
        # 使用默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False
    
    return True

# 设置中文字体
use_chinese = setup_chinese_fonts()

# 图表样式设置
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['figure.dpi'] = 100

warnings.filterwarnings('ignore')

class FeatureDistributionAnalyzer:
    """13个特征分布分析器"""
    
    def __init__(self, data_file_path=None):
        self.data_file_path = data_file_path
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        self.use_chinese = use_chinese
        
        # 中英文标签字典
        self.labels = {
            'title_numerical': 'Numerical Feature Analysis' if not use_chinese else '数值特征分析',
            'title_categorical': 'Categorical Feature Analysis' if not use_chinese else '分类特征分析',
            'histogram': 'Value Distribution Histogram' if not use_chinese else '数值分布直方图',
            'boxplot': 'Box Plot' if not use_chinese else '箱线图',
            'special_values': 'Special Values Statistics' if not use_chinese else '特殊值统计',
            'data_quality': 'Data Quality Distribution' if not use_chinese else '数据质量分布',
            'category_dist': 'Category Distribution (Top 10)' if not use_chinese else '主要类别分布 (前10个)',
            'category_ratio': 'Category Ratio' if not use_chinese else '主要类别比例',
            'data_overview': 'Data Overview' if not use_chinese else '数据概览',
            'value': 'Value' if not use_chinese else '数值',
            'frequency': 'Frequency' if not use_chinese else '频次',
            'category': 'Category' if not use_chinese else '类别',
            'count': 'Count' if not use_chinese else '数量',
            'special_type': 'Special Value Type' if not use_chinese else '特殊值类型',
            'valid_values': 'Valid Values' if not use_chinese else '有效数值',
            'special_vals': 'Special Values' if not use_chinese else '特殊值',
            'others': 'Others' if not use_chinese else '其他',
            'no_special': 'No Special Values' if not use_chinese else '无特殊值',
            'samples': 'Samples' if not use_chinese else '样本数',
            'mean': 'Mean' if not use_chinese else '均值',
            'std': 'Std' if not use_chinese else '标准差',
            'median': 'Median' if not use_chinese else '中位数',
            'indicator': 'Indicator' if not use_chinese else '指标',
            'value_col': 'Value' if not use_chinese else '数值'
        }
        
        # 基于固定列位置的13个预定义特征映射 - 直接使用索引
        self.feature_mappings = {
            'pH_of_condition_enviroment': {
                'column_indices': [54, 59, 60],  # 第55列, 第60列, 第61列 (索引54, 59, 60)
                'type': 'numerical',
                'description': '环境条件pH值 (1-14)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'Chloride_ion': {
                'column_indices': [61, 64, 77],  # 第62列, 第65列, 第78列 (索引61, 64, 77)
                'type': 'categorical', 
                'description': '氯离子浓度',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'concrete': {
                'column_indices': [53, 56, 57],  # 第54列, 第57列, 第58列 (索引53, 56, 57)
                'type': 'categorical',
                'description': '混凝土环境 (0/1)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'diameter': {
                'column_indices': [18, 20],  # 第19列, 第21列 (索引18, 20)
                'type': 'numerical',
                'description': '纤维直径 (mm)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'load_value': {
                'column_indices': [90],  # 第91列 (索引90)
                'type': 'numerical',
                'description': '载荷值',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'fiber_content': {
                'column_indices': [15, 16],  # 第16列, 第17列 (索引15, 16)
                'type': 'numerical',
                'description': '纤维含量 (%)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'Glass_or_Basalt': {
                'column_indices': [8],  # 第9列 (索引8) - 纤维类型
                'type': 'categorical',
                'description': '纤维类型 (Glass/Basalt)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'Vinyl_ester_or_Epoxy': {
                'column_indices': [10],  # 第11列 (索引10) - 树脂类型
                'type': 'categorical',
                'description': '树脂类型 (Vinyl_ester/Epoxy)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'condition_time': {
                'column_indices': [51, 84],  # 第52列, 第85列 (索引51, 84)
                'type': 'numerical',
                'description': '条件时间 (days/hours)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'Temperature': {
                'column_indices': [49, 69, 78],  # 第50列, 第70列, 第79列 (索引49, 69, 78)
                'type': 'numerical',
                'description': '温度 (°C)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'Tensile_strength_retention': {
                'column_indices': [100, 104, 108],  # 第101列, 第105列, 第109列 (索引100, 104, 108)
                'type': 'numerical',
                'description': '拉伸强度保持率 (0-1)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'surface_treatment': {
                'column_indices': [22],  # 第23列 (索引22)
                'type': 'categorical',
                'description': '表面处理 (Yes/No)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            },
            'glass_transition_temperature': {
                'column_indices': [12, 114, 13],  # 第13列, 第115列, 第14列 (索引12, 114, 13)
                'type': 'numerical',
                'description': '玻璃化转变温度 (°C)',
                'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']
            }
        }
        
        self.analysis_report = {
            'timestamp': datetime.now().isoformat(),
            'features_analyzed': {},
            'summary_statistics': {}
        }
    
    def load_data(self):
        """加载数据文件"""
        print("🔄 加载数据文件...")
        
        # 搜索数据文件
        possible_paths = [
            "E:/大学/intern/2025-summer-concret/database 4.xlsx",
            "E:\\大学\\intern\\2025-summer-concret\\database 4.xlsx",
            "../database 4.xlsx",
            "../../database 4.xlsx",
            "data/database 4.xlsx",
            "../data/database 4.xlsx",
        ]
        
        if self.data_file_path:
            possible_paths.insert(0, self.data_file_path)
        
        for path in possible_paths:
            if Path(path).exists():
                try:
                    if Path(path).suffix.lower() in ['.xlsx', '.xls']:
                        # 使用和run_40param_experiment.py相同的读取方式
                        self.data = pd.read_excel(path, header=3, engine='openpyxl')
                    elif Path(path).suffix.lower() == '.csv':
                        self.data = pd.read_csv(path, encoding='utf-8')
                    else:
                        continue
                    
                    print(f"✅ 成功加载数据文件: {path}")
                    print(f"   数据形状: {self.data.shape}")
                    
                    # 显示关键列位置的列名用于验证
                    print(f"📋 关键列位置验证:")
                    key_positions = [8, 10, 15, 18, 22, 49, 51, 54, 59, 90, 100]
                    for pos in key_positions:
                        if pos < len(self.data.columns):
                            print(f"   位置{pos:3d}: {self.data.columns[pos]}")
                        else:
                            print(f"   位置{pos:3d}: <超出范围>")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ 读取文件失败 {path}: {e}")
                    continue
        
        print("❌ 未找到可用的数据文件")
        return False
    
    def display_all_columns(self):
        """显示所有数据列名用于手动检查"""
        print(f"\n📋 数据文件中的所有列名 (共{len(self.data.columns)}个):")
        print("=" * 80)
        
        for i, col in enumerate(self.data.columns, 1):
            print(f"{i:3d}. {col}")
        
        # 保存列名到文件
        columns_file = self.results_dir / "all_columns.txt"
        with open(columns_file, 'w', encoding='utf-8') as f:
            f.write("数据文件中的所有列名\n")
            f.write("=" * 40 + "\n\n")
            for i, col in enumerate(self.data.columns, 1):
                f.write(f"{i:3d}. {col}\n")
        
        print(f"\n💾 所有列名已保存到: {columns_file}")
    
    def find_feature_column(self, feature_name):
        """直接使用列位置索引获取特征数据"""
        column_indices = self.feature_mappings[feature_name]['column_indices']
        
        print(f"  🔍 使用固定列位置: {column_indices}")
        
        # 尝试每个候选位置，找到第一个有效的
        for col_idx in column_indices:
            if col_idx < len(self.data.columns):
                col_name = self.data.columns[col_idx]
                print(f"  ✅ 使用位置 {col_idx}: {col_name}")
                return col_idx  # 返回列索引而不是列名
            else:
                print(f"  ❌ 位置 {col_idx} 超出数据范围 (总列数: {len(self.data.columns)})")
        
        print(f"  ❌ 所有候选位置都无效")
        return None

    def convert_to_json_serializable(self, obj):
        """转换数据为JSON可序列化格式"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def analyze_single_feature(self, feature_name):
        """分析单个特征的数据分布 - 使用列索引"""
        print(f"📊 分析特征: {feature_name}")
        
        # 查找对应的数据列索引
        column_index = self.find_feature_column(feature_name)
        if column_index is None:
            print(f"  ❌ 未找到特征 {feature_name} 对应的数据列")
            return None
        
        column_name = self.data.columns[column_index]
        print(f"  ✅ 找到对应列: 位置{column_index} - {column_name}")
        
        # 获取原始数据 - 直接使用列索引
        raw_data = self.data.iloc[:, column_index].copy()
        feature_info = self.feature_mappings[feature_name]
        
        # 数据预处理 - 识别特殊值
        special_values = feature_info['special_values']
        
        # 统计特殊值
        special_counts = {}
        for special_val in special_values:
            # 检查各种可能的特殊值表示
            mask = raw_data.astype(str).str.contains(special_val, case=False, na=False)
            count = mask.sum()
            if count > 0:
                special_counts[special_val] = int(count)  # 转换为Python int
        
        # 处理缺失值
        null_count = raw_data.isnull().sum()
        if null_count > 0:
            special_counts['Missing'] = int(null_count)  # 转换为Python int
        
        # 提取数值数据
        if feature_info['type'] == 'numerical':
            # 尝试转换为数值
            numeric_data = pd.to_numeric(raw_data, errors='coerce')
            valid_numeric = numeric_data.dropna()
            
            analysis_result = {
                'feature_name': feature_name,
                'column_index': column_index,
                'column_name': column_name,
                'data_type': 'numerical',
                'description': feature_info['description'],
                'total_samples': int(len(raw_data)),
                'valid_numeric_samples': int(len(valid_numeric)),
                'special_values': special_counts,
                'statistics': {}
            }
            
            if len(valid_numeric) > 0:
                analysis_result['statistics'] = {
                    'count': int(len(valid_numeric)),
                    'mean': float(valid_numeric.mean()),
                    'std': float(valid_numeric.std()),
                    'min': float(valid_numeric.min()),
                    'max': float(valid_numeric.max()),
                    'median': float(valid_numeric.median()),
                    'q25': float(valid_numeric.quantile(0.25)),
                    'q75': float(valid_numeric.quantile(0.75))
                }
            
            # 生成数值特征图表
            self._plot_numerical_feature(feature_name, valid_numeric, special_counts, analysis_result)
            
        else:  # categorical
            # 分类特征分析
            value_counts = raw_data.value_counts(dropna=False)
            
            # 转换value_counts为可序列化格式
            value_distribution = {}
            for key, value in value_counts.items():
                # 确保键是字符串，值是Python int
                str_key = str(key) if not pd.isna(key) else 'NaN'
                value_distribution[str_key] = int(value)
            
            analysis_result = {
                'feature_name': feature_name,
                'column_index': column_index,
                'column_name': column_name,
                'data_type': 'categorical',
                'description': feature_info['description'],
                'total_samples': int(len(raw_data)),
                'unique_values': int(len(value_counts)),
                'special_values': special_counts,
                'value_distribution': value_distribution
            }
            
            # 生成分类特征图表
            self._plot_categorical_feature(feature_name, value_counts, special_counts, analysis_result)
        
        # 保存分析结果
        self.analysis_report['features_analyzed'][feature_name] = analysis_result
        
        return analysis_result
    
    def _plot_numerical_feature(self, feature_name, numeric_data, special_counts, analysis_result):
        """绘制数值特征分布图 - 中英文兼容版"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 标题使用英文（避免中文显示问题）
        title = f'{feature_name} - {analysis_result["description"]}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 直方图
        if len(numeric_data) > 0:
            axes[0, 0].hist(numeric_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title(self.labels['histogram'])
            axes[0, 0].set_xlabel(self.labels['value'])
            axes[0, 0].set_ylabel(self.labels['frequency'])
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加统计信息
            stats = analysis_result['statistics']
            stats_text = f"{self.labels['samples']}: {stats['count']}\n{self.labels['mean']}: {stats['mean']:.3f}\n{self.labels['std']}: {stats['std']:.3f}\n{self.labels['median']}: {stats['median']:.3f}"
            axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. 箱线图
        if len(numeric_data) > 0:
            axes[0, 1].boxplot(numeric_data, vert=True)
            axes[0, 1].set_title(self.labels['boxplot'])
            axes[0, 1].set_ylabel(self.labels['value'])
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 特殊值统计
        if special_counts:
            special_names = list(special_counts.keys())
            special_values = list(special_counts.values())
            
            bars = axes[1, 0].bar(special_names, special_values, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title(self.labels['special_values'])
            axes[1, 0].set_xlabel(self.labels['special_type'])
            axes[1, 0].set_ylabel(self.labels['count'])
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, special_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(value), ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, self.labels['no_special'], ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(self.labels['special_values'])
        
        # 4. 数据质量概览
        total_samples = analysis_result['total_samples']
        valid_samples = len(numeric_data)
        special_total = sum(special_counts.values()) if special_counts else 0
        
        quality_data = {
            self.labels['valid_values']: valid_samples,
            self.labels['special_vals']: special_total,
            self.labels['others']: total_samples - valid_samples - special_total
        }
        
        colors = ['lightgreen', 'lightcoral', 'lightgray']
        wedges, texts, autotexts = axes[1, 1].pie(quality_data.values(), labels=quality_data.keys(), 
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 1].set_title(self.labels['data_quality'])
        
        plt.tight_layout()
        
        # 保存图表
        save_path = self.results_dir / f"{feature_name}_numerical_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 数值特征图表已保存: {save_path}")
    
    def _plot_categorical_feature(self, feature_name, value_counts, special_counts, analysis_result):
        """绘制分类特征分布图 - 中英文兼容版"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 标题使用英文（避免中文显示问题）
        title = f'{feature_name} - {analysis_result["description"]}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 主要类别分布
        main_categories = value_counts.head(10)  # 显示前10个最常见的类别
        
        bars1 = axes[0, 0].bar(range(len(main_categories)), main_categories.values, 
                               color='lightblue', alpha=0.7)
        axes[0, 0].set_title(self.labels['category_dist'])
        axes[0, 0].set_xlabel(self.labels['category'])
        axes[0, 0].set_ylabel(self.labels['count'])
        axes[0, 0].set_xticks(range(len(main_categories)))
        axes[0, 0].set_xticklabels([str(x)[:10] for x in main_categories.index], rotation=45, ha='right')
        
        # 添加数值标签
        for bar, value in zip(bars1, main_categories.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(value), ha='center', va='bottom')
        
        # 2. 饼图显示主要类别比例
        if len(main_categories) > 0:
            wedges, texts, autotexts = axes[0, 1].pie(main_categories.values, 
                                                      labels=[str(x)[:8] for x in main_categories.index],
                                                      autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title(self.labels['category_ratio'])
        
        # 3. 特殊值统计
        if special_counts:
            special_names = list(special_counts.keys())
            special_values = list(special_counts.values())
            
            bars2 = axes[1, 0].bar(special_names, special_values, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title(self.labels['special_values'])
            axes[1, 0].set_xlabel(self.labels['special_type'])
            axes[1, 0].set_ylabel(self.labels['count'])
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars2, special_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(value), ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, self.labels['no_special'], ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(self.labels['special_values'])
        
        # 4. 数据概览表格
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        # 创建统计表格 - 使用英文标签
        table_data = [
            ['Total Samples', analysis_result['total_samples']],
            ['Unique Values', analysis_result['unique_values']],
            ['Most Frequent', str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A'],
            ['Most Freq Count', value_counts.iloc[0] if len(value_counts) > 0 else 0],
            ['Special Values', sum(special_counts.values()) if special_counts else 0]
        ]
        
        table = axes[1, 1].table(cellText=table_data, colLabels=[self.labels['indicator'], self.labels['value_col']],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title(self.labels['data_overview'])
        
        plt.tight_layout()
        
        # 保存图表
        save_path = self.results_dir / f"{feature_name}_categorical_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 分类特征图表已保存: {save_path}")
    
    def generate_summary_report(self):
        """生成汇总报告"""
        print("\n📋 生成汇总报告...")
        
        # 创建汇总统计
        total_features = len(self.feature_mappings)
        analyzed_features = len(self.analysis_report['features_analyzed'])
        
        numerical_features = 0
        categorical_features = 0
        
        for feature_name, result in self.analysis_report['features_analyzed'].items():
            if result['data_type'] == 'numerical':
                numerical_features += 1
            else:
                categorical_features += 1
        
        self.analysis_report['summary_statistics'] = {
            'total_features_defined': total_features,
            'features_analyzed': analyzed_features,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'analysis_success_rate': analyzed_features / total_features * 100 if total_features > 0 else 0
        }
        
        # 转换整个报告为JSON可序列化格式
        serializable_report = self.convert_to_json_serializable(self.analysis_report)
        
        # 保存详细报告为JSON
        report_path = self.results_dir / "feature_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        # 生成可读的汇总报告
        summary_path = self.results_dir / "analysis_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("FRP钢筋耐久性预测 - 13个特征数据分布分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"分析时间: {self.analysis_report['timestamp']}\n")
            f.write(f"总特征数: {total_features}\n")
            f.write(f"成功分析: {analyzed_features}\n")
            f.write(f"数值特征: {numerical_features}\n")
            f.write(f"分类特征: {categorical_features}\n")
            f.write(f"分析成功率: {analyzed_features/total_features*100:.1f}%\n\n")
            
            f.write("各特征分析结果:\n")
            f.write("-" * 30 + "\n")
            
            for feature_name, result in self.analysis_report['features_analyzed'].items():
                f.write(f"\n{feature_name}:\n")
                f.write(f"  数据列: {result['column_name']}\n")
                f.write(f"  类型: {result['data_type']}\n")
                f.write(f"  描述: {result['description']}\n")
                f.write(f"  总样本: {result['total_samples']}\n")
                
                if result['data_type'] == 'numerical':
                    if 'valid_numeric_samples' in result:
                        f.write(f"  有效数值: {result['valid_numeric_samples']}\n")
                    if 'statistics' in result and result['statistics']:
                        stats = result['statistics']
                        f.write(f"  统计: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}\n")
                else:
                    f.write(f"  唯一值: {result['unique_values']}\n")
                
                if result['special_values']:
                    f.write(f"  特殊值: {result['special_values']}\n")
            
            # 添加未找到的特征列表
            f.write(f"\n未分析的特征:\n")
            f.write("-" * 20 + "\n")
            for feature_name in self.feature_mappings.keys():
                if feature_name not in self.analysis_report['features_analyzed']:
                    f.write(f"  ❌ {feature_name}\n")
        
        print(f"  💾 详细报告已保存: {report_path}")
        print(f"  📄 汇总报告已保存: {summary_path}")
    
    def run_analysis(self):
        """运行完整的特征分析"""
        print("🚀 开始13个特征数据分布分析")
        print("=" * 50)
        
        # 加载数据
        if not self.load_data():
            print("❌ 数据加载失败，分析终止")
            return
        
        # 显示所有列名
        self.display_all_columns()
        
        print(f"\n📊 开始分析13个预定义特征...")
        
        # 分析每个特征
        success_count = 0
        for i, feature_name in enumerate(self.feature_mappings.keys(), 1):
            print(f"\n[{i:2d}/13] {feature_name}")
            try:
                result = self.analyze_single_feature(feature_name)
                if result:
                    print(f"  ✅ 分析完成")
                    success_count += 1
                else:
                    print(f"  ❌ 分析失败")
            except Exception as e:
                print(f"  ❌ 分析出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成汇总报告
        self.generate_summary_report()
        
        print(f"\n🎉 分析完成！")
        print(f"  📁 结果保存在: {self.results_dir.absolute()}")
        print(f"  📊 成功分析特征: {success_count}/13 个")
        print(f"  📊 生成图表: {len(self.analysis_report['features_analyzed'])} 个")
        print(f"  📋 分析报告: feature_analysis_report.json")
        print(f"  📄 汇总报告: analysis_summary.txt")
        print(f"  📋 所有列名: all_columns.txt")
        
        # 给出改进建议
        if success_count < 13:
            print(f"\n💡 改进建议:")
            print(f"   1. 查看 all_columns.txt 文件中的所有列名")
            print(f"   2. 手动找到对应的特征列")
            print(f"   3. 更新 feature_mappings 中的候选列名")
            print(f"   4. 重新运行分析")

def main():
    """主函数"""
    # 创建分析器
    analyzer = FeatureDistributionAnalyzer()
    
    # 运行分析
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
