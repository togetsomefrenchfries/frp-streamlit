#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的pH检测逻辑
"""

from data_loader import DataLoader
import pandas as pd

def test_fixed_ph_detection():
    """测试修复后的pH检测逻辑"""
    
    loader = DataLoader()
    df = loader.load_data()
    
    print('🧪 测试修复后的pH检测逻辑')
    print('=' * 50)
    
    # 统计不同列中的水类型数据
    water_columns = ['cycle_pH', 'pHafter', 'concrete', 'note_of_concrete', 'ingredient']
    water_types = ['distilled water', 'sea water', 'seawater', 'tap water', 'artificial sea water']
    
    print('📊 水类型数据分布统计:')
    for col in water_columns:
        if col in df.columns:
            print(f'\n🔍 {col} 列:')
            col_data = df[col].astype(str).str.lower()
            
            for water_type in water_types:
                matches = col_data.str.contains(water_type, na=False, regex=False)
                count = matches.sum()
                if count > 0:
                    print(f'  📍 "{water_type}": {count} 个')
    
    # 测试新的检测逻辑
    print('\n🔧 新检测逻辑测试 (前10行数据):')
    print('-' * 40)
    
    for i in range(min(10, len(df))):
        idx = df.index[i]
        
        # 模拟新的检测逻辑
        solution_text = ''
        source_col = 'none'
        
        water_description_columns = [
            'solution_condition', 'ingredient', 'cycle_pH', 
            'pHafter', 'concrete', 'note_of_concrete'
        ]
        
        for col in water_description_columns:
            if col in df.columns and not solution_text:
                col_value = df.loc[idx, col]
                if col_value is not None and not pd.isna(col_value):
                    test_text = str(col_value).lower()
                    if test_text and test_text != 'nan':
                        solution_text = test_text
                        source_col = col
                        break
        
        water_types_check = ['tap water', 'sea water', 'seawater', 'distilled water', 
                           'deionized water', 'di water', 'pure water', 'artificial sea water']
        
        detected_water = None
        for water_type in water_types_check:
            if water_type in solution_text:
                detected_water = water_type
                break
        
        if detected_water:
            has_chloride = 'sea' in solution_text
            print(f'  行{i+1}: 检测到"{detected_water}" (来源:{source_col}) → pH=7.0, 氯离子={has_chloride}')

if __name__ == "__main__":
    test_fixed_ph_detection()