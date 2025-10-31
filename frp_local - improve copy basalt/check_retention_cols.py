#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def check_retention_columns():
    """检查retention列的内容"""
    try:
        # 读取数据库文件
        df = pd.read_excel('../../database 4.xlsx')
        
        print(f"检查第103和104列:")
        print(f"第103列 (索引102): {df.columns[102]}")
        print(f"第104列 (索引103): {df.columns[103]}")
        
        print(f"\n第103列样例数据:")
        col_103 = df.iloc[:, 102]
        print(f"  类型: {col_103.dtype}")
        print(f"  前10个值: {col_103.head(10).tolist()}")
        print(f"  非空数据: {col_103.notna().sum()}/{len(df)}")
        
        print(f"\n第104列样例数据:")
        col_104 = df.iloc[:, 103]
        print(f"  类型: {col_104.dtype}")
        print(f"  前10个值: {col_104.head(10).tolist()}")
        print(f"  非空数据: {col_104.notna().sum()}/{len(df)}")
        
        # 转换为数值并查看
        numeric_104 = pd.to_numeric(col_104, errors='coerce')
        valid_104 = numeric_104.dropna()
        if len(valid_104) > 0:
            print(f"  数值范围: {valid_104.min():.3f} - {valid_104.max():.3f}")
            print(f"  均值: {valid_104.mean():.3f}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_retention_columns()