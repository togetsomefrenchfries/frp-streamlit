#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查BH列中的文本型pH数据
"""

from data_loader import DataLoader
import pandas as pd

def analyze_bh_column():
    """分析BH列的详细情况"""
    
    loader = DataLoader()
    df = loader.load_data()
    
    print('🔍 BH列(pH)详细分析')
    print('=' * 50)
    
    ph_col = df['pH']
    print(f'BH列总数据量: {len(ph_col)}')
    print(f'非空数据: {ph_col.count()}')
    
    # 检查数据类型
    numeric_ph = pd.to_numeric(ph_col, errors='coerce')
    numeric_count = numeric_ph.count()
    text_count = ph_col.count() - numeric_count
    
    print(f'数值型pH: {numeric_count}')
    print(f'文本型pH: {text_count}')
    
    if text_count > 0:
        print('\n📍 文本型pH数据:')
        text_ph_mask = ph_col.notna() & pd.to_numeric(ph_col, errors='coerce').isna()
        text_ph_data = ph_col[text_ph_mask]
        
        print(f'文本型数据数量: {len(text_ph_data)}')
        for i, (idx, val) in enumerate(text_ph_data.items()):
            if i < 20:  # 显示前20个
                print(f'  {i+1}. 行{idx}: "{val}"')
            elif i == 20:
                print(f'  ... 还有{len(text_ph_data)-20}个')
                break
        
        print('\n📊 文本型pH数据统计:')
        unique_text_values = text_ph_data.value_counts()
        for val, count in unique_text_values.head(10).items():
            print(f'  "{val}": {count} 次')
    
    else:
        print('\n✅ BH列中所有pH数据都是数值型！')
        print('\n📊 数值pH范围:')
        valid_numeric = numeric_ph.dropna()
        print(f'  最小值: {valid_numeric.min()}')
        print(f'  最大值: {valid_numeric.max()}')
        print(f'  平均值: {valid_numeric.mean():.2f}')
        print(f'  中位数: {valid_numeric.median()}')

if __name__ == "__main__":
    analyze_bh_column()