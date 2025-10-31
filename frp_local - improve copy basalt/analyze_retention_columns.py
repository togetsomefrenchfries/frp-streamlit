#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def analyze_potential_retention_cols():
    """分析可能的retention列"""
    try:
        # 读取数据库文件
        df = pd.read_excel('../../database 4.xlsx')
        
        # 重点分析的列
        potential_retention_cols = [93, 104, 108, 136, 137, 138, 139, 140]
        
        print(f"分析可能的retention列:")
        print(f"=" * 60)
        
        for col_idx in potential_retention_cols:
            if col_idx < len(df.columns):
                col_data = df.iloc[:, col_idx]
                col_name = df.columns[col_idx]
                
                print(f"\n列 {col_idx}: {col_name}")
                print(f"-" * 40)
                
                # 转换为数值
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                non_null_data = numeric_data.dropna()
                
                if len(non_null_data) > 0:
                    print(f"  非空数据量: {len(non_null_data)}/{len(df)} ({len(non_null_data)/len(df)*100:.1f}%)")
                    print(f"  数值范围: {non_null_data.min():.6f} - {non_null_data.max():.6f}")
                    print(f"  均值: {non_null_data.mean():.6f}")
                    print(f"  中位数: {non_null_data.median():.6f}")
                    print(f"  标准差: {non_null_data.std():.6f}")
                    
                    # 显示一些实际的数值例子
                    sample_values = non_null_data.head(10).tolist()
                    print(f"  前10个值: {sample_values}")
                    
                    # 检查与Target_parameter='tensile'的关系
                    target_col = df.iloc[:, 29]  # AE列 = 30 (索引29)
                    tensile_mask = target_col.astype(str).str.lower().str.contains('tensile', na=False)
                    
                    if tensile_mask.any():
                        tensile_retention = numeric_data[tensile_mask].dropna()
                        if len(tensile_retention) > 0:
                            print(f"  tensile数据中的retention: {len(tensile_retention)}个")
                            print(f"    范围: {tensile_retention.min():.6f} - {tensile_retention.max():.6f}")
                            print(f"    均值: {tensile_retention.mean():.6f}")
                            print(f"    样例: {tensile_retention.head(5).tolist()}")
                else:
                    print(f"  没有有效的数值数据")
                    
        # 特别检查一下我们当前用的第100列(索引99)是什么
        print(f"\n" + "="*60)
        print(f"当前代码使用的第100列 (索引99) 分析:")
        col_100_data = df.iloc[:, 99]
        print(f"  列名: {df.columns[99]}")
        print(f"  数据类型: {col_100_data.dtype}")
        print(f"  unique值分布:")
        value_counts = col_100_data.value_counts().head(20)
        for value, count in value_counts.items():
            print(f"    '{value}': {count}个")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_potential_retention_cols()