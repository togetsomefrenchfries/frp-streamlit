#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正后的pH检测逻辑
"""

from data_loader import DataLoader
import pandas as pd

def test_ph_extraction_logic():
    """测试pH提取逻辑"""
    
    loader = DataLoader()
    df = loader.load_data()
    
    print('🔍 pH提取逻辑完整测试')
    print('=' * 60)
    
    # 统计pH数据来源
    print('📊 pH数据来源分析:')
    
    # 检查BH列(pH)的使用情况
    ph_col = df['pH']
    numeric_ph_count = pd.to_numeric(ph_col, errors='coerce').count()
    total_ph_count = ph_col.count()
    
    print(f'  BH列(pH): {numeric_ph_count}/{total_ph_count} 数值型数据')
    print(f'  缺失数据: {len(df) - total_ph_count} 条需要其他方式推断')
    
    # 模拟完整的pH检测逻辑
    print(f'\n🧪 pH检测优先级测试:')
    print('-' * 40)
    
    test_cases = [
        # (BH列值, ingredient值, 期望pH, 期望氯离子, 描述)
        (12.5, None, 12.5, 0, 'BH列有数值'),
        (None, 'NaCl solution', 7.0, 1, 'BH列空，ingredient有氯化钠'),
        (None, 'artificial sea water', 8.0, 1, 'BH列空，人工海水'),
        (None, 'seawater', 8.0, 1, 'BH列空，海水'),
        (None, 'distilled water', 7.0, 0, 'BH列空，蒸馏水'),
        (None, 'tap water', 7.0, 0, 'BH列空，自来水'),
        (8.2, 'seawater', 8.2, 1, 'BH列优先，但海水仍检测氯离子'),
    ]
    
    for i, (bh_ph, ingredient_text, expected_ph, expected_cl, desc) in enumerate(test_cases, 1):
        print(f'  测试{i}: {desc}')
        
        # 模拟检测逻辑
        final_ph = 7.0  # 默认值
        chloride_ion = 0
        ph_source = 'default'
        
        # 步骤1: 检查BH列
        if bh_ph is not None:
            final_ph = bh_ph
            ph_source = 'BH列(pH)'
        
        # 步骤2: 氯离子检测 (无论pH来源如何)
        if ingredient_text:
            ingredient_lower = ingredient_text.lower()
            chloride_keywords = ['cl', 'chloride', 'nacl', 'cacl2', 'mgcl2', 'salt', 'seawater', 'sea water', 'artificial sea water']
            if any(keyword in ingredient_lower for keyword in chloride_keywords):
                chloride_ion = 1
            
            # 如果BH列为空，基于水类型推断pH
            if bh_ph is None:
                water_types = ['tap water', 'sea water', 'seawater', 'distilled water', 
                              'artificial sea water']
                for water_type in water_types:
                    if water_type in ingredient_lower:
                        if 'sea' in ingredient_lower:
                            final_ph = 8.0
                            chloride_ion = 1
                        else:
                            final_ph = 7.0
                        ph_source = f'水类型推断({water_type})'
                        break
        
        status = '✅' if (final_ph == expected_ph and chloride_ion == expected_cl) else '❌'
        print(f'    {status} pH={final_ph} (来源:{ph_source}), 氯离子={chloride_ion}')
    
    # 统计实际数据中的情况
    print(f'\n📊 实际数据统计:')
    print('-' * 30)
    
    # 有BH列数据的情况
    has_bh_ph = ph_col.notna().sum()
    missing_bh_ph = len(df) - has_bh_ph
    
    print(f'  有BH列pH数据: {has_bh_ph} 条 ({has_bh_ph/len(df)*100:.1f}%)')
    print(f'  缺少BH列pH: {missing_bh_ph} 条 ({missing_bh_ph/len(df)*100:.1f}%)')
    
    # 在缺少BH列的数据中，有多少可以通过水类型推断
    missing_ph_mask = ph_col.isna()
    if missing_ph_mask.sum() > 0:
        water_columns = ['cycle_pH', 'pHafter', 'concrete', 'note_of_concrete', 'ingredient']
        water_types = ['distilled water', 'sea water', 'seawater', 'tap water', 'artificial sea water']
        
        can_infer_count = 0
        for col in water_columns:
            if col in df.columns:
                col_data = df[missing_ph_mask][col].astype(str).str.lower()
                for water_type in water_types:
                    can_infer_count += col_data.str.contains(water_type, na=False).sum()
        
        print(f'  可通过水类型推断: {can_infer_count} 条')
        print(f'  仍需默认值: {missing_bh_ph - can_infer_count} 条')

if __name__ == "__main__":
    test_ph_extraction_logic()