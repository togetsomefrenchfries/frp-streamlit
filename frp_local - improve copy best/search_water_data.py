#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索数据中的water相关内容
"""

from data_loader import DataLoader
import pandas as pd

def search_water_data():
    """搜索water相关数据"""
    
    loader = DataLoader()
    df = loader.load_data()
    
    print('🔍 搜索 water 相关数据')
    print('=' * 60)
    
    # 搜索关键词
    search_terms = ['distilled water', 'artificial sea water', 'seawater', 'sea water', 'tap water']
    
    for term in search_terms:
        print(f'\n🔍 搜索: "{term}"')
        print('-' * 40)
        
        total_count = 0
        found_columns = []
        
        for col in df.columns:
            # 转换为字符串并搜索
            col_data = df[col].astype(str).str.lower()
            matches = col_data.str.contains(term.lower(), na=False, regex=False)
            count = matches.sum()
            
            if count > 0:
                total_count += count
                found_columns.append((col, count))
                print(f'  📍 {col} (列{df.columns.get_loc(col)}): {count} 个匹配')
                
                # 显示几个示例
                examples = df[matches][col].head(5).tolist()
                for i, example in enumerate(examples, 1):
                    print(f'     {i}. "{example}"')
        
        print(f'  ✅ 总计: {total_count} 个匹配，分布在 {len(found_columns)} 个列中')
    
    # 特别检查ingredient相关列
    print(f'\n📊 Ingredient相关列详细分析')
    print('=' * 40)
    
    ingredient_cols = ['ingredient', 'ingredient.1', 'cycle_ingredient']
    for col in ingredient_cols:
        if col in df.columns:
            print(f'\n列: {col}')
            non_null_data = df[col].dropna()
            print(f'  非空数据: {len(non_null_data)} 条')
            
            # 显示唯一值示例
            unique_values = non_null_data.unique()[:10]
            print(f'  唯一值示例 (前10个):')
            for i, val in enumerate(unique_values, 1):
                print(f'    {i}. "{val}"')

if __name__ == "__main__":
    search_water_data()