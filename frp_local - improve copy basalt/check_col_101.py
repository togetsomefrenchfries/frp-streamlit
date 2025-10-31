#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def check_col_101():
    """检查第101列的内容"""
    try:
        # 读取数据库文件
        df = pd.read_excel('../../database 4.xlsx')
        
        print(f"检查第101列:")
        print(f"第101列 (索引100): {df.columns[100]}")
        
        print(f"\n第101列样例数据:")
        col_101 = df.iloc[:, 100]
        print(f"  类型: {col_101.dtype}")
        print(f"  前15个值: {col_101.head(15).tolist()}")
        print(f"  非空数据: {col_101.notna().sum()}/{len(df)}")
        
        # 转换为数值并查看
        numeric_101 = pd.to_numeric(col_101, errors='coerce')
        valid_101 = numeric_101.dropna()
        if len(valid_101) > 0:
            print(f"  数值统计:")
            print(f"    有效数值: {len(valid_101)}")
            print(f"    数值范围: {valid_101.min():.6f} - {valid_101.max():.6f}")
            print(f"    均值: {valid_101.mean():.6f}")
            print(f"    中位数: {valid_101.median():.6f}")
            print(f"    标准差: {valid_101.std():.6f}")
            print(f"    前10个数值: {valid_101.head(10).tolist()}")
        else:
            print(f"  没有有效的数值数据")
            
        # 检查unique值
        print(f"\n第101列unique值分布:")
        value_counts = col_101.value_counts().head(20)
        for value, count in value_counts.items():
            print(f"    '{value}': {count}个")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_col_101()