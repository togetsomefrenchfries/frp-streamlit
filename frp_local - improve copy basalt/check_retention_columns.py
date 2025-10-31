#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def check_columns():
    """检查数据库文件的列结构"""
    try:
        # 读取数据库文件
        df = pd.read_excel('../../database 4.xlsx')
        
        print(f"数据库基本信息:")
        print(f"  总行数: {len(df)}")
        print(f"  总列数: {len(df.columns)}")
        
        print(f"\n关键列检查:")
        # 检查第30列 (Target_parameter)
        if len(df.columns) > 29:
            print(f"  第30列 (索引29): {df.columns[29]}")
            target_values = df.iloc[:, 29].value_counts().head(5)
            print(f"    示例值: {dict(target_values)}")
        else:
            print(f"  第30列: 不存在")
            
        # 检查第100列 (retention1)
        if len(df.columns) > 99:
            print(f"  第100列 (索引99): {df.columns[99]}")
            retention_values = df.iloc[:, 99].describe()
            print(f"    数值统计: {retention_values}")
        else:
            print(f"  第100列: 不存在")
            
        # 搜索包含retention的列
        print(f"\n搜索包含'retention'的列:")
        retention_cols = []
        for i, col_name in enumerate(df.columns):
            if 'retention' in str(col_name).lower():
                retention_cols.append((i, col_name))
                
        if retention_cols:
            for idx, col_name in retention_cols:
                print(f"  索引{idx}: {col_name}")
                # 显示该列的一些统计信息
                col_data = df.iloc[:, idx]
                non_null_count = col_data.notna().sum()
                print(f"    非空数据: {non_null_count}/{len(df)} ({non_null_count/len(df)*100:.1f}%)")
                if col_data.dtype in ['int64', 'float64']:
                    print(f"    数值范围: {col_data.min():.3f} - {col_data.max():.3f}")
                    print(f"    均值: {col_data.mean():.3f}")
        else:
            print("  没有找到包含'retention'的列名")
            
        # 列出所有列名 (前50列)
        print(f"\n前50列列名:")
        for i, col in enumerate(df.columns[:50]):
            print(f"  {i:2d}: {col}")
            
        if len(df.columns) > 50:
            print(f"\n后50列列名:")
            start_idx = max(50, len(df.columns) - 50)
            for i, col in enumerate(df.columns[start_idx:], start_idx):
                print(f"  {i:2d}: {col}")
                
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_columns()