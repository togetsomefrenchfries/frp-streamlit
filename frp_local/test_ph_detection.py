#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试pH检测逻辑
"""

from data_loader import DataLoader
import pandas as pd

def test_ph_detection():
    """测试pH检测逻辑"""
    
    # 加载数据
    loader = DataLoader()
    df = loader.load_data()
    
    print('🔍 pH检测测试')
    print('=' * 50)
    
    # 检查相关列的存在性
    relevant_columns = ['pH', 'solution_condition', 'ingredient']
    print('相关列存在性检查:')
    for col in relevant_columns:
        exists = col in df.columns
        print(f'  {col}: {"✅ 存在" if exists else "❌ 不存在"}')
    
    print()
    
    # 分析pH列的内容
    if 'pH' in df.columns:
        ph_col = df['pH']
        print('BH列(pH)数据类型分析:')
        print(f'  总数据量: {len(ph_col)}')
        print(f'  非空数据: {ph_col.count()}')
        numeric_count = pd.to_numeric(ph_col, errors='coerce').count()
        print(f'  数值型数据: {numeric_count}')
        print(f'  文本型数据: {ph_col.count() - numeric_count}')
        
        print()
        print('pH列唯一值示例 (前20个):')
        unique_values = ph_col.dropna().unique()[:20]
        for i, val in enumerate(unique_values, 1):
            print(f'  {i:2d}. "{val}" ({type(val).__name__})')
    
    print()
    
    # 测试预处理逻辑
    print('🧪 pH检测逻辑测试')
    print('-' * 30)
    
    test_cases = [
        ('artificial seawater', '人工海水'),
        ('distilled water', '蒸馏水'),
        ('tap water', '自来水'),
        ('seawater', '海水'),
        ('7.5', '数值pH'),
        ('alkaline solution', '碱性溶液'),
        ('Unknown', '未知溶液')
    ]
    
    water_types = ['tap water', 'sea water', 'seawater', 'distilled water', 
                   'deionized water', 'di water', 'pure water']
    
    for test_input, description in test_cases:
        test_lower = test_input.lower()
        
        # 模拟检测逻辑
        is_numeric = False
        try:
            float(test_input)
            is_numeric = True
        except:
            pass
        
        if is_numeric:
            result_ph = float(test_input)
            detection_type = '数值检测'
        elif any(water_type in test_lower for water_type in water_types):
            result_ph = 7.0
            detection_type = '水类型检测'
            # 海水特殊处理
            if 'sea' in test_lower:
                detection_type += ' + 氯离子检测'
        else:
            result_ph = 7.0  # 默认值
            detection_type = '默认值'
        
        print(f'  {description}: "{test_input}" → pH={result_ph} ({detection_type})')

if __name__ == "__main__":
    test_ph_detection()