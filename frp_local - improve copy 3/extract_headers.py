#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取Excel文件的标题行
Extract Headers from Excel File

专门用于读取database 4.xlsx文件的第3行和第4行内容
并输出详细的列信息到txt文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

class HeaderExtractor:
    """Excel标题行提取器"""
    
    def __init__(self):
        self.data = None
        self.file_path = None
        
    def find_excel_file(self):
        """查找Excel文件"""
        print("🔍 搜索Excel文件...")
        
        # 可能的文件路径
        possible_paths = [
            "E:/大学/intern/2025-summer-concret/database 4.xlsx",
            "E:\\大学\\intern\\2025-summer-concret\\database 4.xlsx",
            "../database 4.xlsx",
            "../../database 4.xlsx",
            "../../../database 4.xlsx",
            "data/database 4.xlsx",
            "../data/database 4.xlsx",
            "../../data/database 4.xlsx",
            "database 4.xlsx",
            "./database 4.xlsx"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                self.file_path = path
                print(f"✅ 找到Excel文件: {path}")
                return True
        
        print("❌ 未找到database 4.xlsx文件")
        print("请确保文件位于以下任一位置:")
        for path in possible_paths[:5]:
            print(f"   - {path}")
        return False
    
    def load_excel_data(self):
        """加载Excel数据"""
        print(f"📖 正在读取Excel文件: {self.file_path}")
        
        try:
            # 读取Excel文件，不设置header，保留所有原始行
            self.data = pd.read_excel(self.file_path, sheet_name=0, header=None)
            print(f"✅ Excel读取成功")
            print(f"   数据形状: {self.data.shape}")
            print(f"   总行数: {len(self.data)}")
            print(f"   总列数: {len(self.data.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Excel读取失败: {e}")
            return False
    
    def extract_header_info(self):
        """提取标题行信息"""
        if self.data is None:
            print("❌ 请先加载数据")
            return None
        
        print("🔍 提取第3行和第4行标题信息...")
        
        # 检查是否有足够的行数
        if len(self.data) < 4:
            print("❌ 数据行数不足4行")
            return None
        
        # 获取第3行和第4行 (索引2和3)
        row3 = self.data.iloc[2]  # 第3行 - 大类标题
        row4 = self.data.iloc[3]  # 第4行 - 细分列名
        
        # 创建详细信息
        header_info = {
            'total_columns': len(self.data.columns),
            'row3_data': row3,
            'row4_data': row4,
            'file_path': self.file_path,
            'data_shape': self.data.shape
        }
        
        print(f"✅ 标题信息提取完成")
        print(f"   第3行非空值: {row3.count()}/{len(row3)}")
        print(f"   第4行非空值: {row4.count()}/{len(row4)}")
        
        return header_info
    
    def save_header_info(self, header_info):
        """保存标题信息到txt文件"""
        if header_info is None:
            print("❌ 没有标题信息可保存")
            return False
        
        # 创建输出目录
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # 创建输出文件
        output_file = output_dir / "database4_headers_row3_row4.txt"
        
        print(f"💾 正在保存标题信息到: {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Database 4.xlsx 第3行和第4行标题信息\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"文件路径: {header_info['file_path']}\n")
                f.write(f"数据形状: {header_info['data_shape']}\n")
                f.write(f"总列数: {header_info['total_columns']}\n\n")
                
                # 第3行信息
                f.write("📁 第3行 (大类标题):\n")
                f.write("-" * 40 + "\n")
                row3 = header_info['row3_data']
                f.write(f"非空值数量: {row3.count()}/{len(row3)}\n\n")
                
                for i, value in enumerate(row3):
                    if pd.isna(value):
                        value_str = "NaN"
                    else:
                        value_str = str(value)
                    f.write(f"列{i:3d}: {value_str}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
                
                # 第4行信息
                f.write("📋 第4行 (细分列名):\n")
                f.write("-" * 40 + "\n")
                row4 = header_info['row4_data']
                f.write(f"非空值数量: {row4.count()}/{len(row4)}\n\n")
                
                for i, value in enumerate(row4):
                    if pd.isna(value):
                        value_str = "NaN"
                    else:
                        value_str = str(value)
                    f.write(f"列{i:3d}: {value_str}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
                
                # 第3行和第4行对应关系
                f.write("🔗 第3行-第4行对应关系:\n")
                f.write("-" * 40 + "\n")
                f.write("列号  | 第3行(大类)          | 第4行(细分)\n")
                f.write("-" * 60 + "\n")
                
                for i in range(len(row3)):
                    row3_val = row3.iloc[i]
                    row4_val = row4.iloc[i]
                    
                    row3_str = str(row3_val)[:18] if pd.notna(row3_val) else "─"
                    row4_str = str(row4_val)[:18] if pd.notna(row4_val) else "─"
                    
                    f.write(f"{i:3d}   | {row3_str:<18} | {row4_str}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("提取完成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
            print(f"✅ 标题信息已保存到: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            return False
    
    def create_column_summary(self, header_info):
        """创建列汇总信息"""
        if header_info is None:
            return False
        
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        summary_file = output_dir / "database4_column_summary.txt"
        
        print(f"📊 创建列汇总信息: {summary_file}")
        
        try:
            row3 = header_info['row3_data']
            row4 = header_info['row4_data']
            
            # 统计信息
            row3_non_null = row3.count()
            row4_non_null = row4.count()
            total_cols = len(row3)
            
            # 找出有值的列
            row3_with_values = [(i, str(val)) for i, val in enumerate(row3) if pd.notna(val)]
            row4_with_values = [(i, str(val)) for i, val in enumerate(row4) if pd.notna(val)]
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("Database 4.xlsx 列汇总信息\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("📊 统计概览:\n")
                f.write(f"   总列数: {total_cols}\n")
                f.write(f"   第3行有值列数: {row3_non_null} ({row3_non_null/total_cols*100:.1f}%)\n")
                f.write(f"   第4行有值列数: {row4_non_null} ({row4_non_null/total_cols*100:.1f}%)\n\n")
                
                f.write("📁 第3行有值的列 (大类标题):\n")
                f.write("-" * 40 + "\n")
                for pos, value in row3_with_values:
                    f.write(f"   列{pos:3d}: {value}\n")
                
                f.write(f"\n📋 第4行有值的列 (细分列名):\n")
                f.write("-" * 40 + "\n")
                for pos, value in row4_with_values:
                    f.write(f"   列{pos:3d}: {value}\n")
                
                # 分析可能的特征列
                f.write(f"\n🎯 可能的关键特征列:\n")
                f.write("-" * 40 + "\n")
                
                keywords = ['ph', 'temperature', 'time', 'retention', 'diameter', 
                           'fiber', 'glass', 'strength', 'concrete', 'ingredient']
                
                for keyword in keywords:
                    matches = []
                    for pos, value in row4_with_values:
                        if keyword.lower() in value.lower():
                            matches.append((pos, value))
                    
                    if matches:
                        f.write(f"\n   '{keyword}' 相关列:\n")
                        for pos, value in matches:
                            f.write(f"      列{pos:3d}: {value}\n")
            
            print(f"✅ 列汇总信息已保存")
            return True
            
        except Exception as e:
            print(f"❌ 创建汇总信息失败: {e}")
            return False
    
    def run_extraction(self):
        """运行完整的提取流程"""
        print("🚀 开始提取Excel标题信息")
        print("=" * 50)
        
        # 步骤1: 查找文件
        if not self.find_excel_file():
            return False
        
        # 步骤2: 加载数据
        if not self.load_excel_data():
            return False
        
        # 步骤3: 提取标题信息
        header_info = self.extract_header_info()
        if header_info is None:
            return False
        
        # 步骤4: 保存详细信息
        if not self.save_header_info(header_info):
            return False
        
        # 步骤5: 创建汇总信息
        if not self.create_column_summary(header_info):
            return False
        
        print("\n🎉 标题提取完成！")
        print("📁 结果保存在 analysis_results/ 目录:")
        print("   - database4_headers_row3_row4.txt (详细信息)")
        print("   - database4_column_summary.txt (汇总信息)")
        
        return True

def main():
    """主函数"""
    extractor = HeaderExtractor()
    
    try:
        extractor.run_extraction()
    except KeyboardInterrupt:
        print("\n⚠️  用户取消操作")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
