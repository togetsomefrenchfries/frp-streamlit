#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据列名检测工具 - 两层标题结构分析
Data Column Inspector - Two-Level Header Analysis

专门分析Excel文件的两层标题结构：
- 第3行：大类标题 (如 "solution information", "fiber properties")
- 第4行：具体细分列名 (如 "pH", "ingredient", "diameter")
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter, defaultdict

class TwoLevelHeaderInspector:
    """两层标题结构检测器"""
    
    def __init__(self, data_file_path=None):
        self.data_file_path = data_file_path
        self.data = None
        self.category_row = 2  # 第3行(索引2) - 大类
        self.detail_row = 3    # 第4行(索引3) - 细分
        
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
                    # 读取Excel文件，保留所有行作为数据
                    self.data = pd.read_excel(path, sheet_name=0, header=None)
                    print(f"✅ 成功加载数据文件: {path}")
                    print(f"   数据形状: {self.data.shape}")
                    return True
                    
                except Exception as e:
                    print(f"❌ 读取文件失败 {path}: {e}")
                    continue
        
        print("❌ 未找到可用的数据文件")
        return False
    
    def analyze_two_level_headers(self):
        """分析两层标题结构"""
        print("\n📋 两层标题结构分析")
        print("=" * 80)
        
        if len(self.data) <= self.detail_row:
            print("❌ 数据行数不足，无法分析两层标题")
            return
        
        # 获取第3行(大类)和第4行(细分)
        category_row = self.data.iloc[self.category_row]
        detail_row = self.data.iloc[self.detail_row]
        
        print(f"🔍 第3行 (大类标题):")
        print(f"   非空值: {category_row.count()}/{len(category_row)}")
        
        print(f"🔍 第4行 (细分列名):")
        print(f"   非空值: {detail_row.count()}/{len(detail_row)}")
        
        # 分析两层结构
        self._analyze_category_structure(category_row, detail_row)
        
    def _analyze_category_structure(self, category_row, detail_row):
        """分析大类和细分的对应关系"""
        print(f"\n📊 大类与细分列名对应关系分析:")
        print("=" * 60)
        
        # 构建大类到细分的映射
        category_mapping = defaultdict(list)
        current_category = None
        
        for i in range(len(category_row)):
            # 获取大类名称
            category_val = category_row.iloc[i]
            detail_val = detail_row.iloc[i]
            
            # 如果大类有值，更新当前大类
            if pd.notna(category_val) and str(category_val).strip():
                current_category = str(category_val).strip()
            
            # 如果细分有值，添加到当前大类下
            if pd.notna(detail_val) and str(detail_val).strip():
                detail_name = str(detail_val).strip()
                if current_category:
                    category_mapping[current_category].append((i, detail_name))
                else:
                    category_mapping["未分类"].append((i, detail_name))
        
        # 显示分类结果
        print(f"发现 {len(category_mapping)} 个大类:")
        
        for category, details in category_mapping.items():
            print(f"\n📁 {category} ({len(details)} 个细分列):")
            for pos, detail_name in details:
                # 显示该列的示例数据
                sample_data = self.data.iloc[self.detail_row+1:self.detail_row+4, pos].values
                sample_str = [str(val)[:15] if pd.notna(val) else 'NaN' for val in sample_data]
                print(f"   [{pos:3d}] {detail_name}")
                print(f"        示例: {sample_str}")
        
        # 保存分类结构
        self._save_category_structure(category_mapping)
        
        return category_mapping
    
    def search_target_features_in_structure(self):
        """在两层结构中搜索目标特征"""
        print(f"\n🎯 在两层结构中搜索13个目标特征")
        print("=" * 80)
        
        # 目标特征及其可能出现的大类和细分关键词
        target_features = {
            'pH_of_condition_enviroment': {
                'category_keywords': ['solution', 'environment', 'condition', 'chemical'],
                'detail_keywords': ['ph', 'acid', 'alkaline', 'acidity']
            },
            'Chloride_ion': {
                'category_keywords': ['solution', 'environment', 'chemical', 'ingredient'],
                'detail_keywords': ['chloride', 'cl', 'salt', 'nacl', 'ingredient', 'composition']
            },
            'concrete': {
                'category_keywords': ['environment', 'condition', 'structural'],
                'detail_keywords': ['concrete', 'crack', 'cover', 'cement', 'mortar']
            },
            'diameter': {
                'category_keywords': ['geometry', 'dimension', 'physical', 'fiber', 'specimen'],
                'detail_keywords': ['diameter', 'dia', 'size', 'cross-section', 'area']
            },
            'load_value': {
                'category_keywords': ['mechanical', 'loading', 'stress', 'test'],
                'detail_keywords': ['load', 'stress', 'strain', 'force', 'loading', 'value']
            },
            'fiber_content': {
                'category_keywords': ['fiber', 'composition', 'material', 'properties'],
                'detail_keywords': ['content', 'weight', 'volume', 'fraction', 'percentage', 'fiber']
            },
            'Glass_or_Basalt': {
                'category_keywords': ['fiber', 'material', 'type', 'composition'],
                'detail_keywords': ['glass', 'basalt', 'type', 'material', 'fiber']
            },
            'Vinyl_ester_or_Epoxy': {
                'category_keywords': ['matrix', 'resin', 'material', 'polymer'],
                'detail_keywords': ['vinyl', 'epoxy', 'resin', 'matrix', 'polymer']
            },
            'condition_time': {
                'category_keywords': ['time', 'duration', 'exposure', 'aging'],
                'detail_keywords': ['time', 'duration', 'period', 'days', 'hours', 'exposure']
            },
            'Temperature': {
                'category_keywords': ['environment', 'condition', 'thermal', 'temperature'],
                'detail_keywords': ['temperature', 'temp', 'thermal', 'heat', 'degree']
            },
            'Tensile_strength_retention': {
                'category_keywords': ['mechanical', 'strength', 'retention', 'degradation'],
                'detail_keywords': ['retention', 'strength', 'tensile', 'remaining', 'degradation']
            },
            'surface_treatment': {
                'category_keywords': ['surface', 'treatment', 'modification', 'fiber'],
                'detail_keywords': ['surface', 'treatment', 'coating', 'sizing', 'modification']
            },
            'glass_transition_temperature': {
                'category_keywords': ['thermal', 'temperature', 'properties', 'material'],
                'detail_keywords': ['tg', 'glass', 'transition', 'temperature']
            }
        }
        
        # 获取两层标题
        category_row = self.data.iloc[self.category_row]
        detail_row = self.data.iloc[self.detail_row]
        
        feature_matches = {}
        
        for feature_name, search_criteria in target_features.items():
            print(f"\n🔍 搜索特征: {feature_name}")
            
            candidates = []
            category_keywords = search_criteria['category_keywords']
            detail_keywords = search_criteria['detail_keywords']
            
            for i in range(len(detail_row)):
                category_val = str(category_row.iloc[i]).lower() if pd.notna(category_row.iloc[i]) else ""
                detail_val = str(detail_row.iloc[i]).lower() if pd.notna(detail_row.iloc[i]) else ""
                
                # 检查大类匹配
                category_match = any(keyword in category_val for keyword in category_keywords)
                # 检查细分匹配
                detail_match = any(keyword in detail_val for keyword in detail_keywords)
                
                # 如果大类或细分有匹配，就加入候选
                if category_match or detail_match:
                    match_score = 0
                    if category_match: match_score += 1
                    if detail_match: match_score += 2  # 细分匹配权重更高
                    
                    candidates.append({
                        'position': i,
                        'category': str(category_row.iloc[i]) if pd.notna(category_row.iloc[i]) else "N/A",
                        'detail': str(detail_row.iloc[i]) if pd.notna(detail_row.iloc[i]) else "N/A",
                        'score': match_score
                    })
            
            # 按匹配分数排序
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            if candidates:
                print(f"   找到 {len(candidates)} 个候选列:")
                for idx, candidate in enumerate(candidates[:5]):  # 只显示前5个
                    print(f"   {idx+1}. [{candidate['position']:3d}] {candidate['category']} → {candidate['detail']} (分数:{candidate['score']})")
                    
                    # 显示示例数据
                    sample_data = self.data.iloc[self.detail_row+1:self.detail_row+4, candidate['position']].values
                    sample_str = [str(val)[:12] if pd.notna(val) else 'NaN' for val in sample_data]
                    print(f"        示例: {sample_str}")
                
                feature_matches[feature_name] = candidates
            else:
                print(f"   ❌ 未找到候选列")
                feature_matches[feature_name] = []
        
        # 保存搜索结果
        self._save_feature_search_results(feature_matches)
        
        return feature_matches
    
    def generate_preprocessor_mapping(self, feature_matches):
        """生成preprocessor.py的映射代码"""
        print(f"\n💻 生成preprocessor.py的映射代码")
        print("=" * 80)
        
        # 获取列名
        detail_row = self.data.iloc[self.detail_row]
        column_names = [str(val) if pd.notna(val) else f'Unnamed_{i}' for i, val in enumerate(detail_row)]
        
        mapping_code = """
# 基于实际数据结构的特征映射 (两层标题结构)
self.feature_mappings = {"""
        
        for feature_name, candidates in feature_matches.items():
            if candidates:
                # 选择最佳候选(分数最高的)
                best_candidates = [column_names[c['position']] for c in candidates[:3]]  # 取前3个
                
                mapping_code += f"""
    '{feature_name}': {{
        'candidates': {best_candidates},
        'type': '{"numerical" if feature_name not in ["Glass_or_Basalt", "Vinyl_ester_or_Epoxy", "surface_treatment", "concrete"] else "categorical"}',
        'positions': {[c['position'] for c in candidates[:3]]},
        'scores': {[c['score'] for c in candidates[:3]]}
    }},"""
            else:
                mapping_code += f"""
    '{feature_name}': {{
        'candidates': [],
        'type': '{"numerical" if feature_name not in ["Glass_or_Basalt", "Vinyl_ester_or_Epoxy", "surface_treatment", "concrete"] else "categorical"}',
        'positions': [],
        'scores': []
    }},"""
        
        mapping_code += """
}"""
        
        # 保存代码
        code_file = Path("analysis_results") / "preprocessor_mapping_code.py"
        code_file.parent.mkdir(exist_ok=True)
        
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write("# 基于两层标题结构分析生成的特征映射代码\n")
            f.write("# 可以直接复制到 preprocessor.py 中\n\n")
            f.write(mapping_code)
        
        print(f"💾 映射代码已保存到: {code_file}")
        print("📋 你可以复制这些代码到 preprocessor.py 中替换现有的 feature_mappings")
        
    def _save_category_structure(self, category_mapping):
        """保存分类结构到文件"""
        output_file = Path("analysis_results") / "two_level_header_structure.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("两层标题结构分析报告\n")
            f.write("第3行: 大类标题\n")
            f.write("第4行: 细分列名\n")
            f.write("=" * 50 + "\n\n")
            
            for category, details in category_mapping.items():
                f.write(f"📁 {category} ({len(details)} 个细分列):\n")
                for pos, detail_name in details:
                    f.write(f"   [{pos:3d}] {detail_name}\n")
                f.write("\n")
        
        print(f"\n💾 两层结构分析已保存到: {output_file}")
    
    def _save_feature_search_results(self, feature_matches):
        """保存特征搜索结果"""
        output_file = Path("analysis_results") / "feature_search_results.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("13个目标特征搜索结果\n")
            f.write("基于两层标题结构\n")
            f.write("=" * 50 + "\n\n")
            
            for feature_name, candidates in feature_matches.items():
                f.write(f"🎯 {feature_name}:\n")
                if candidates:
                    for idx, candidate in enumerate(candidates):
                        f.write(f"   {idx+1}. [{candidate['position']:3d}] {candidate['category']} → {candidate['detail']} (分数:{candidate['score']})\n")
                else:
                    f.write("   ❌ 未找到候选列\n")
                f.write("\n")
        
        print(f"💾 特征搜索结果已保存到: {output_file}")
    
    def quick_preview_data_format(self):
        """快速预览数据格式 - 显示前几行原始数据"""
        print("\n👀 快速数据格式预览")
        print("=" * 80)
        
        if self.data is None:
            print("❌ 请先加载数据")
            return
        
        # 显示前8行的原始数据，重点关注第3行和第4行
        print("🔍 前8行原始数据 (重点: 第3行大类, 第4行细分):")
        print()
        
        for row_idx in range(min(8, len(self.data))):
            print(f"第{row_idx+1}行:", end=" ")
            
            # 特别标记第3行和第4行
            if row_idx == 2:
                print("📁 [大类标题] ", end="")
            elif row_idx == 3:
                print("📋 [细分列名] ", end="")
            else:
                print("   [数据行]   ", end="")
            
            # 显示前20列的内容
            row_data = []
            for col_idx in range(min(20, len(self.data.columns))):
                cell_value = self.data.iloc[row_idx, col_idx]
                if pd.isna(cell_value):
                    cell_str = "NaN"
                else:
                    cell_str = str(cell_value)[:15]  # 限制长度
                row_data.append(f"'{cell_str}'")
            
            print("[" + ", ".join(row_data) + "]")
            
            # 如果是第3行或第4行，额外分析
            if row_idx in [2, 3]:
                non_null_count = self.data.iloc[row_idx].count()
                print(f"        → 非空值: {non_null_count}/{len(self.data.columns)} 列")
        
        print(f"\n📊 数据基本信息:")
        print(f"   总行数: {len(self.data)}")
        print(f"   总列数: {len(self.data.columns)}")
        
        # 分析第3行和第4行的特点
        if len(self.data) > 3:
            category_row = self.data.iloc[2]  # 第3行
            detail_row = self.data.iloc[3]    # 第4行
            
            print(f"\n🔍 第3行 (大类) 分析:")
            category_values = [str(val) for val in category_row.dropna() if str(val).strip()]
            print(f"   有效大类: {len(category_values)} 个")
            if category_values:
                print(f"   示例大类: {category_values[:5]}")
            
            print(f"\n🔍 第4行 (细分) 分析:")
            detail_values = [str(val) for val in detail_row.dropna() if str(val).strip()]
            print(f"   有效细分: {len(detail_values)} 个")
            if detail_values:
                print(f"   示例细分: {detail_values[:10]}")
    
    def show_column_pairs_preview(self):
        """显示第3行-第4行的列对应关系预览"""
        print("\n🔗 第3行-第4行对应关系预览 (前30列)")
        print("=" * 80)
        
        if len(self.data) <= 3:
            print("❌ 数据行数不足")
            return
        
        category_row = self.data.iloc[2]  # 第3行
        detail_row = self.data.iloc[3]    # 第4行
        
        print("列号  │  第3行(大类)           │  第4行(细分)           │  示例数据")
        print("─" * 75)
        
        for i in range(min(30, len(self.data.columns))):
            category_val = category_row.iloc[i]
            detail_val = detail_row.iloc[i]
            
            # 格式化显示
            category_str = str(category_val)[:20] if pd.notna(category_val) else "─"
            detail_str = str(detail_val)[:20] if pd.notna(detail_val) else "─"
            
            # 获取示例数据
            if len(self.data) > 4:
                sample_val = self.data.iloc[4, i]
                sample_str = str(sample_val)[:15] if pd.notna(sample_val) else "NaN"
            else:
                sample_str = "─"
            
            print(f"{i:3d}   │  {category_str:<20} │  {detail_str:<20} │  {sample_str}")
        
        if len(self.data.columns) > 30:
            print(f"... 还有 {len(self.data.columns) - 30} 列")

    def run_full_analysis(self):
        """运行完整的两层结构分析"""
        print("🚀 开始两层标题结构完整分析")
        print("=" * 60)
        
        # 1. 加载数据
        if not self.load_data():
            return
        
        # 2. 快速预览数据格式
        self.quick_preview_data_format()
        
        # 3. 显示列对应关系
        self.show_column_pairs_preview()
        
        # 4. 分析两层标题结构  
        self.analyze_two_level_headers()
        
        # 5. 在结构中搜索目标特征
        feature_matches = self.search_target_features_in_structure()
        
        # 6. 生成preprocessor映射代码
        self.generate_preprocessor_mapping(feature_matches)
        
        print(f"\n🎉 两层结构分析完成！")
        print(f"📁 结果保存在 analysis_results/ 目录")
        print(f"💡 下一步: 查看生成的映射代码，更新 preprocessor.py")

def main():
    """主函数"""
    print("📋 两层标题结构分析工具")
    print("第3行: 大类标题 (如 'solution information')")
    print("第4行: 细分列名 (如 'pH', 'ingredient')")
    print()
    
    inspector = TwoLevelHeaderInspector()
    
    try:
        inspector.run_full_analysis()
        
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
