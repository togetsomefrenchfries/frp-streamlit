#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比分析 app.py 和 dataset code（excel to SQL）.py 的列处理差异
"""

def analyze_column_mappings():
    """分析两个文件中的列映射差异"""
    
    print("🔍 对比分析 app.py 和 dataset code 的列处理差异")
    print("=" * 80)
    
    # dataset code（excel to SQL）.py中的列映射（前132列）
    dataset_code_columns = [
        'feature_name',                    # 位置1  -> Python索引0
        'Title',                          # 位置2  -> Python索引1
        'Author',                         # 位置3  -> Python索引2
        'SCI',                           # 位置4  -> Python索引3
        'Journal_or_Conference_name',     # 位置5  -> Python索引4
        'Year',                          # 位置6  -> Python索引5
        'No_field',                      # 位置7  -> Python索引6
        'no_field_secondary',            # 位置8  -> Python索引7
        'Fiber_type',                    # 位置9  -> Python索引8  ✓
        'Fiber_type_detail',             # 位置10 -> Python索引9
        'Matrix_type',                   # 位置11 -> Python索引10 ✓
        'Matrix_type_detail',            # 位置12 -> Python索引11
        'glass_transition_temperature',   # 位置13 -> Python索引12 ✓
        'glass_transition_temperature_run_2', # 位置14 -> Python索引13 ✓
        'cure_ratio',                    # 位置15 -> Python索引14
        'Fiber_content_weight',          # 位置16 -> Python索引15 ✓
        'Fiber_content_volume',          # 位置17 -> Python索引16 ✓
        'Void_content',                  # 位置18 -> Python索引17
        'diameter',                      # 位置19 -> Python索引18 ✓
        'average_area',                  # 位置20 -> Python索引19
        'nominal_area',                  # 位置21 -> Python索引20 ✓
        'rib',                          # 位置22 -> Python索引21
        'surface_treatment',             # 位置23 -> Python索引22 ✓
        'Water_absorption_at_saturation', # 位置24 -> Python索引23
        'Water_absorption_test_standard', # 位置25 -> Python索引24
        'Water_absorption_note',         # 位置26 -> Python索引25
        'Brand_name',                    # 位置27 -> Python索引26
        'Manufacturer',                  # 位置28 -> Python索引27
        'Important_notes',               # 位置29 -> Python索引28
        'Notes_of_rebar',               # 位置30 -> Python索引29
        'Target_parameter',              # 位置31 -> Python索引30
        'note_of_target_parameter',      # 位置32 -> Python索引31
        'num_1',                        # 位置33 -> Python索引32
        'note_of_number',               # 位置34 -> Python索引33
        'Value1_1',                     # 位置35 -> Python索引34
        'COV1_1',                       # 位置36 -> Python索引35
        'note_of_Value1',               # 位置37 -> Python索引36
        'Value2_1',                     # 位置38 -> Python索引37
        'COV2_1',                       # 位置39 -> Python索引38
        'Value2note_1',                 # 位置40 -> Python索引39
        'Value3_1',                     # 位置41 -> Python索引40
        'COV3_1',                       # 位置42 -> Python索引41
        'Value3note_1',                 # 位置43 -> Python索引42
        'SEM_T_BCBT',                   # 位置44 -> Python索引43
        'SEM_L_BCBT',                   # 位置45 -> Python索引44
        'OTHER_main',                   # 位置46 -> Python索引45
        'OTHER1_1',                     # 位置47 -> Python索引46
        'FTIR_1',                       # 位置48 -> Python索引47
        'note_1',                       # 位置49 -> Python索引48
        'temperature',                   # 位置50 -> Python索引49 ✓
        'note_of_temperature',           # 位置51 -> Python索引50
        'time_field',                    # 位置52 -> Python索引51 ✓
        'note_of_time',                  # 位置53 -> Python索引52
        'concrete',                      # 位置54 -> Python索引53 ✓
        'pH_of_concrete',                # 位置55 -> Python索引54 ✓
        'strength_of_concrete',          # 位置56 -> Python索引55
        'crack',                         # 位置57 -> Python索引56 ✓
        'cover',                         # 位置58 -> Python索引57 ✓
        'note_of_concrete',              # 位置59 -> Python索引58
        'pH_1',                          # 位置60 -> Python索引59 ✓
        'pHafter',                       # 位置61 -> Python索引60 ✓
        'ingredient_1',                  # 位置62 -> Python索引61 ✓
        'pH_2',                          # 位置63 -> Python索引62
        'RH_1',                          # 位置64 -> Python索引63
        'ingredient_2',                  # 位置65 -> Python索引64 ✓
        'note_2',                        # 位置66 -> Python索引65
        'Location',                      # 位置67 -> Python索引66
        'Effektive_Klimaklassifikation', # 位置68 -> Python索引67
        'field_average_humidity',        # 位置69 -> Python索引68
        'field_average_temperature',     # 位置70 -> Python索引69 ✓
        'number_field',                  # 位置71 -> Python索引70
        'type_field',                    # 位置72 -> Python索引71
        'SolutionorMoisture',            # 位置73 -> Python索引72
        'cycle_pH',                      # 位置74 -> Python索引73
        'cycle_pH_after',                # 位置75 -> Python索引74
        'cycle_ingredient',              # 位置76 -> Python索引75
        'temp',                          # 位置77 -> Python索引76
        'temp2',                         # 位置78 -> Python索引77 ✓
        'RH_2',                          # 位置79 -> Python索引78 ✓
        'RH2',                           # 位置80 -> Python索引79
        'OTHER1_2',                      # 位置81 -> Python索引80
        'OTHER2_main',                   # 位置82 -> Python索引81
        'time_in_cycle',                 # 位置83 -> Python索引82
        'note_3',                        # 位置84 -> Python索引83
        'UV',                            # 位置85 -> Python索引84 ✓
        'note_4',                        # 位置86 -> Python索引85
        'stress_or_strain',              # 位置87 -> Python索引86
        'type_of_load',                  # 位置88 -> Python索引87
        'value_load',                    # 位置89 -> Python索引88
        'ultimate_tensile_strength',     # 位置90 -> Python索引89
        'tensile_modulus',               # 位置91 -> Python索引90 ✓
        'note_5',                        # 位置92 -> Python索引91
        'after_condition',               # 位置93 -> Python索引92
        'note_6',                        # 位置94 -> Python索引93
        'num_2',                         # 位置95 -> Python索引94
        'Value1_2',                      # 位置96 -> Python索引95
        'COV1_2',                        # 位置97 -> Python索引96
        'Value1note',                    # 位置98 -> Python索引97
        'retention1',                    # 位置99 -> Python索引98
        'Value2_2',                      # 位置100 -> Python索引99
        'COV2_2',                        # 位置101 -> Python索引100 ✓
        'Value2note_2',                  # 位置102 -> Python索引101
        'retention2',                    # 位置103 -> Python索引102
        'Value3_2',                      # 位置104 -> Python索引103
        'COV3_2',                        # 位置105 -> Python索引104 ✓
        'Value3note_2',                  # 位置106 -> Python索引105
        'retention3',                    # 位置107 -> Python索引106
        'num_3',                         # 位置108 -> Python索引107
        'water_absorption_ratio',        # 位置109 -> Python索引108 ✓
        'COV_1',                         # 位置110 -> Python索引109
        'note_7',                        # 位置111 -> Python索引110
        'num_4',                         # 位置112 -> Python索引111
        'glass_transition_temperature_2', # 位置113 -> Python索引112
        'run2',                          # 位置114 -> Python索引113
        'COV_2',                         # 位置115 -> Python索引114 ✓
        'cure_ratio_2',                  # 位置116 -> Python索引115
        'note_8',                        # 位置117 -> Python索引116
        'num_5',                         # 位置118 -> Python索引117
        'OTHERS',                        # 位置119 -> Python索引118
        'OTHERS_note',                   # 位置120 -> Python索引119
        'SEM_T_BCAT',                    # 位置121 -> Python索引120
        'SEM_L_BCAT',                    # 位置122 -> Python索引121
        'SEM_T_ACBT',                    # 位置123 -> Python索引122
        'SEM_L_ACBT',                    # 位置124 -> Python索引123
        'SEM_T_ACAT',                    # 位置125 -> Python索引124
        'SEM_L_ACAT',                    # 位置126 -> Python索引125
        'other_lower',                   # 位置127 -> Python索引126
        'other2_final',                  # 位置128 -> Python索引127
        'note_9',                        # 位置129 -> Python索引128
        'FTIR_2',                        # 位置130 -> Python索引129
        'note_10',                       # 位置131 -> Python索引130
        'important_note'                 # 位置132 -> Python索引131
    ]
    
    # analyze_13features_distribution.py中的特征映射
    features_13_mappings = {
        'pH_of_condition_enviroment': {
            'column_indices': [54, 59, 60],  # 对应dataset_code的pH_of_concrete, pH_1, pHafter
            'dataset_columns': ['pH_of_concrete', 'pH_1', 'pHafter']
        },
        'Chloride_ion': {
            'column_indices': [61, 64, 77],  # 对应dataset_code的ingredient_1, ingredient_2, temp2
            'dataset_columns': ['ingredient_1', 'ingredient_2', 'temp2']
        },
        'concrete': {
            'column_indices': [53, 56, 57],  # 对应dataset_code的concrete, crack, cover
            'dataset_columns': ['concrete', 'crack', 'cover']
        },
        'diameter': {
            'column_indices': [18, 20],      # 对应dataset_code的diameter, nominal_area
            'dataset_columns': ['diameter', 'nominal_area']
        },
        'load_value': {
            'column_indices': [90],          # 对应dataset_code的tensile_modulus
            'dataset_columns': ['tensile_modulus']
        },
        'fiber_content': {
            'column_indices': [15, 16],      # 对应dataset_code的Fiber_content_weight, Fiber_content_volume
            'dataset_columns': ['Fiber_content_weight', 'Fiber_content_volume']
        },
        'Glass_or_Basalt': {
            'column_indices': [8],           # 对应dataset_code的Fiber_type
            'dataset_columns': ['Fiber_type']
        },
        'Vinyl_ester_or_Epoxy': {
            'column_indices': [10],          # 对应dataset_code的Matrix_type
            'dataset_columns': ['Matrix_type']
        },
        'condition_time': {
            'column_indices': [51, 84],      # 对应dataset_code的time_field, UV
            'dataset_columns': ['time_field', 'UV']
        },
        'Temperature': {
            'column_indices': [49, 69, 78],  # 对应dataset_code的temperature, field_average_temperature, RH_2
            'dataset_columns': ['temperature', 'field_average_temperature', 'RH_2']
        },
        'Tensile_strength_retention': {
            'column_indices': [100, 104, 108], # 对应dataset_code的COV2_2, COV3_2, water_absorption_ratio
            'dataset_columns': ['COV2_2', 'COV3_2', 'water_absorption_ratio']
        },
        'surface_treatment': {
            'column_indices': [22],          # 对应dataset_code的surface_treatment
            'dataset_columns': ['surface_treatment']
        },
        'glass_transition_temperature': {
            'column_indices': [12, 114, 13], # 对应dataset_code的glass_transition_temperature, COV_2, glass_transition_temperature_run_2
            'dataset_columns': ['glass_transition_temperature', 'COV_2', 'glass_transition_temperature_run_2']
        }
    }
    
    print("\n📊 13个特征与dataset_code列名的对应关系:")
    print("-" * 80)
    
    for i, (feature_name, mapping) in enumerate(features_13_mappings.items(), 1):
        print(f"\n[{i:2d}] {feature_name}")
        print(f"     Python索引: {mapping['column_indices']}")
        print(f"     对应列名: {mapping['dataset_columns']}")
        
        # 验证索引和列名的对应关系
        print(f"     验证对应:")
        for j, (idx, col_name) in enumerate(zip(mapping['column_indices'], mapping['dataset_columns'])):
            if idx < len(dataset_code_columns) and dataset_code_columns[idx] == col_name:
                print(f"       ✅ 索引{idx} = {col_name}")
            else:
                if idx < len(dataset_code_columns):
                    actual_col = dataset_code_columns[idx]
                    print(f"       ❌ 索引{idx} = {actual_col} (预期: {col_name})")
                else:
                    print(f"       ❌ 索引{idx} 超出范围 (预期: {col_name})")

def analyze_bu_column_shift():
    """分析BU列之后的位置偏差问题"""
    print(f"\n\n🔍 分析BU列之后的位置偏差问题")
    print("=" * 60)
    
    # BU列是第73列，Python索引72
    bu_column_index = 72
    print(f"BU列位置: Excel第73列 = Python索引{bu_column_index}")
    
    # 检查analyze_13features_distribution.py中大于72的索引
    large_indices = [77, 78, 84, 90, 100, 104, 108, 114]
    
    print(f"\n📋 大于BU列位置的索引分析:")
    print(f"{'索引':<6} | {'Excel列':<8} | {'预期功能':<20} | {'可能的偏差'}")
    print("-" * 60)
    
    index_analysis = [
        (77, 'temp2', '温度相关'),
        (78, 'RH_2', '湿度相关'),
        (84, 'UV', '紫外线条件'),
        (90, 'tensile_modulus', '拉伸模量'),
        (100, 'COV2_2', '变异系数'),
        (104, 'COV3_2', '变异系数'),
        (108, 'water_absorption_ratio', '吸水率'),
        (114, 'COV_2', '变异系数')
    ]
    
    for idx, expected_col, function in index_analysis:
        excel_col_num = idx + 1
        deviation = "可能偏差" if idx > bu_column_index else "正常"
        print(f"{idx:<6} | 第{excel_col_num}列{'':<3} | {function:<20} | {deviation}")

def suggest_corrections():
    """建议修正方案"""
    print(f"\n\n🛠️ 建议修正方案")
    print("=" * 50)
    
    suggestions = [
        "1. 验证实际Excel文件的列结构",
        "2. 检查BU列(第73列)之后是否有列插入或删除",
        "3. 对比database 1.xlsx和database 4.xlsx的列结构差异",
        "4. 使用列名匹配而不是固定索引位置",
        "5. 增加列验证和映射检查机制"
    ]
    
    for suggestion in suggestions:
        print(suggestion)
    
    print(f"\n💡 关键发现:")
    key_findings = [
        "• app.py使用列名进行特征工程，更加灵活",
        "• dataset_code使用固定132列结构",
        "• analyze_13features_distribution.py使用固定索引，可能因文件版本差异失效",
        "• BU列之后的索引可能因为Excel文件结构变化而不准确"
    ]
    
    for finding in key_findings:
        print(finding)

if __name__ == "__main__":
    analyze_column_mappings()
    analyze_bu_column_shift()
    suggest_corrections()