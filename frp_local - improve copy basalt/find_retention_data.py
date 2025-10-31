#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def find_retention_data():
    """寻找实际的retention数据"""
    try:
        # 读取数据库文件
        df = pd.read_excel('../../database 4.xlsx')
        
        print(f"寻找真实的retention数据...")
        print(f"数据库总行数: {len(df)}, 总列数: {len(df.columns)}")
        
        # 检查第100列的实际数据
        if len(df.columns) > 99:
            col_100 = df.iloc[:, 99]
            print(f"\n第100列 (索引99) 的实际数据:")
            print(f"  列名: {df.columns[99]}")
            print(f"  数据类型: {col_100.dtype}")
            print(f"  非空数据: {col_100.notna().sum()}/{len(df)}")
            print(f"  前10个值: {list(col_100.head(10))}")
            print(f"  unique值: {col_100.value_counts().head(10)}")
            
        # 搜索可能包含retention数据的列
        print(f"\n搜索可能包含retention数据的列:")
        potential_cols = []
        
        # 搜索列名中包含相关关键词的列
        keywords = ['retention', 'strength', 'stress', 'modulus', 'strain', 'tensile', '%']
        for i, col_name in enumerate(df.columns):
            col_name_str = str(col_name).lower()
            for keyword in keywords:
                if keyword in col_name_str and 'unnamed' not in col_name_str:
                    potential_cols.append((i, col_name, keyword))
                    break
                    
        if potential_cols:
            print(f"  找到{len(potential_cols)}个可能相关的列:")
            for idx, col_name, keyword in potential_cols:
                col_data = df.iloc[:, idx]
                non_null = col_data.notna().sum()
                print(f"    索引{idx}: {col_name} (匹配'{keyword}') - 非空: {non_null}")
                
        # 寻找数值型列，特别是在合理范围内的retention值(通常0-1之间或0-100%)
        print(f"\n搜索可能的retention数值列 (0-2范围内的数值):")
        retention_candidates = []
        
        for i, col_name in enumerate(df.columns):
            col_data = df.iloc[:, i]
            # 检查是否为数值型且在合理范围内
            if col_data.dtype in ['float64', 'int64']:
                # 去除缺失值后检查范围
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_data) > 0:
                    min_val = numeric_data.min()
                    max_val = numeric_data.max()
                    mean_val = numeric_data.mean()
                    
                    # retention通常在0-2之间 (比例) 或 0-200% 之间
                    if (0 <= min_val <= 2 and 0 <= max_val <= 2) or (0 <= min_val <= 200 and 0 <= max_val <= 200):
                        non_null_count = numeric_data.notna().sum()
                        retention_candidates.append((i, col_name, min_val, max_val, mean_val, len(numeric_data)))
                        
        if retention_candidates:
            print(f"  找到{len(retention_candidates)}个可能的retention数值列:")
            for idx, col_name, min_val, max_val, mean_val, count in retention_candidates:
                print(f"    索引{idx}: {col_name}")
                print(f"      范围: {min_val:.3f} - {max_val:.3f}, 均值: {mean_val:.3f}, 数据量: {count}")
                
        # 特别检查一些特定列范围
        print(f"\n检查特定列范围 (30-50, 90-110):")
        for col_range in [(30, 50), (90, 110)]:
            start, end = col_range
            print(f"  检查列 {start}-{end}:")
            for i in range(start, min(end, len(df.columns))):
                col_data = df.iloc[:, i]
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_data) > 0:
                    min_val = numeric_data.min()
                    max_val = numeric_data.max()
                    if 0 <= min_val <= 2 and 0 <= max_val <= 2:
                        print(f"    索引{i}: {df.columns[i]} - 范围: {min_val:.3f}-{max_val:.3f}, 数据量: {len(numeric_data)}")
                
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    find_retention_data()