#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于实际数据列名的特征映射建议

根据观察到的实际列名，提供更准确的特征映射建议
"""

def get_feature_mapping_suggestions():
    """基于实际列名的特征映射建议"""
    
    suggestions = {
        'pH_of_condition_enviroment': {
            'primary_candidates': [
                'solution, moisture and temperature could be used in following columns'
            ],
            'description': '环境条件pH值 - 可能在solution相关列中',
            'analysis_note': '这个列可能包含pH、湿度、温度等多种环境条件信息'
        },
        
        'Chloride_ion': {
            'primary_candidates': [
                'solution, moisture and temperature could be used in following columns'
            ],
            'description': '氯离子浓度 - 可能在solution相关列中',
            'analysis_note': '氯离子信息可能与溶液条件一起记录'
        },
        
        'concrete': {
            'primary_candidates': [
                'wet-dry: WD, freeze-thaw:FT, hot temperature: HT, combination:com, other: O'
            ],
            'description': '混凝土环境 - 可能在环境条件描述中',
            'analysis_note': '混凝土环境可能通过不同的测试条件来体现'
        },
        
        'diameter': {
            'primary_candidates': [
                'transverse',
                'longitudinal'
            ],
            'description': '纤维直径 - 横向和纵向尺寸',
            'analysis_note': '可能记录为横向和纵向的尺寸信息'
        },
        
        'load_value': {
            'primary_candidates': [
                'normally stress, MPa',
                'normally strain', 
                'normally modulus, MPa'
            ],
            'description': '载荷值 - 应力、应变、模量',
            'analysis_note': '载荷信息通过应力、应变或模量来表示'
        },
        
        'fiber_content': {
            'primary_candidates': [
                '%,weight',
                '%,volume',
                '%,volume.1'
            ],
            'description': '纤维含量 - 重量百分比或体积百分比',
            'analysis_note': '纤维含量以重量或体积百分比形式记录'
        },
        
        'Glass_or_Basalt': {
            'primary_candidates': [
                'Notreported means Notreported, SMD means inexistence, \',\' means range, \';\' means different stages'
            ],
            'description': '纤维类型 - 可能在说明列中',
            'analysis_note': '纤维类型信息可能需要从复合信息列中提取'
        },
        
        'Vinyl_ester_or_Epoxy': {
            'primary_candidates': [
                'Notreported means Notreported, SMD means inexistence, \',\' means range, \';\' means different stages'
            ],
            'description': '树脂类型 - 可能在说明列中',
            'analysis_note': '树脂类型信息可能需要从复合信息列中提取'
        },
        
        'condition_time': {
            'primary_candidates': [
                'before condition, before test',
                'before condition, after test',
                'dry and recover'
            ],
            'description': '条件时间 - 测试前后的时间信息',
            'analysis_note': '时间信息记录在测试条件相关列中'
        },
        
        'Temperature': {
            'primary_candidates': [
                'solution, moisture and temperature could be used in following columns',
                'wet-dry: WD, freeze-thaw:FT, hot temperature: HT, combination:com, other: O'
            ],
            'description': '温度 - 环境温度或测试温度',
            'analysis_note': '温度信息可能在环境条件或测试条件中'
        },
        
        'Tensile_strength_retention': {
            'primary_candidates': [
                'before condition, before test',
                'before condition, after test',
                'Shear: short beam test (interlaminar shear strength), Thermogravimetreic Analysis'
            ],
            'description': '拉伸强度保持率 - 测试前后对比',
            'analysis_note': '强度保持率可能通过测试前后数据计算得出'
        },
        
        'surface_treatment': {
            'primary_candidates': [
                'ASTM D570/ASTM D5229',
                'Shear: short beam test (interlaminar shear strength), Thermogravimetreic Analysis'
            ],
            'description': '表面处理 - 可能在测试标准或方法中体现',
            'analysis_note': '表面处理信息可能通过测试标准来间接表示'
        },
        
        'glass_transition_temperature': {
            'primary_candidates': [
                'Tg1',
                'Tg2'
            ],
            'description': '玻璃化转变温度 - 直接匹配',
            'analysis_note': '这是最明确的匹配，Tg代表glass transition temperature'
        }
    }
    
    return suggestions

def print_mapping_suggestions():
    """打印特征映射建议"""
    suggestions = get_feature_mapping_suggestions()
    
    print("🎯 基于实际数据列名的特征映射建议")
    print("=" * 60)
    
    for i, (feature_name, info) in enumerate(suggestions.items(), 1):
        print(f"\n{i:2d}. {feature_name}")
        print(f"    描述: {info['description']}")
        print(f"    建议列名:")
        for j, candidate in enumerate(info['primary_candidates'], 1):
            print(f"      {j}. {candidate}")
        print(f"    分析说明: {info['analysis_note']}")

def get_updated_feature_mappings():
    """生成更新后的特征映射代码"""
    suggestions = get_feature_mapping_suggestions()
    
    print("\n📝 更新后的特征映射代码:")
    print("=" * 40)
    print("self.feature_mappings = {")
    
    for feature_name, info in suggestions.items():
        print(f"    '{feature_name}': {{")
        print(f"        'candidates': {info['primary_candidates']},")
        
        # 根据特征类型设置
        if feature_name in ['Glass_or_Basalt', 'Vinyl_ester_or_Epoxy', 'surface_treatment', 'concrete']:
            data_type = 'categorical'
        else:
            data_type = 'numerical'
        
        print(f"        'type': '{data_type}',")
        print(f"        'description': '{info['description']}',")
        print(f"        'special_values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported']")
        print(f"    }},")
    
    print("}")

def main():
    """主函数"""
    print_mapping_suggestions()
    print("\n" + "="*60)
    get_updated_feature_mappings()

if __name__ == "__main__":
    main()
