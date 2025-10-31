#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细对比 app.py 和 dataset code 中每个特征的提取和处理方法
"""

def analyze_feature_extraction_methods():
    """详细分析每个特征在两个系统中的处理方法"""
    
    print("🔍 详细对比 app.py 和 dataset code 的特征提取方法")
    print("=" * 80)
    
    # 13个特征的详细对比分析
    feature_comparisons = [
        {
            'id': 1,
            'feature_name': 'pH_of_condition_enviroment',
            'description': '环境条件pH值',
            'app_py_method': {
                'source_columns': ['Condition_environment', 'pH_of_concrete', 'solution_condition', 'pH_1', 'pHafter', 'ingredient_1'],
                'processing_logic': [
                    '1. 判断环境类型：混凝土环境 vs 溶液环境',
                    '2. 混凝土环境：使用pH_of_concrete，默认值13.0',
                    '3. 溶液环境：优先使用数值pH，否则根据溶液类型赋值',
                    '4. 海水等特殊溶液赋值7.0',
                    '5. 如果有pHafter，取平均值',
                    '6. 智能文本解析提取pH值'
                ],
                'default_value': '7.0 (溶液) 或 13.0 (混凝土)',
                'complexity': '高 - 复杂逻辑判断'
            },
            'dataset_code_method': {
                'source_columns': ['pH_of_concrete', 'pH_1', 'pHafter'],
                'processing_logic': [
                    '1. 直接从固定列位置读取数值',
                    '2. 按优先级：索引54 → 索引59 → 索引60',
                    '3. 使用第一个非空数值',
                    '4. 无复杂判断逻辑'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 简单数值提取'
            },
            'key_differences': [
                '• app.py有智能环境类型判断',
                '• app.py支持文本解析和默认值',
                '• dataset_code仅做简单数值提取',
                '• app.py处理更全面但复杂'
            ]
        },
        {
            'id': 2,
            'feature_name': 'Chloride_ion',
            'description': '氯离子浓度指示器',
            'app_py_method': {
                'source_columns': ['ingredient_1', 'ingredient_2', 'solution_condition'],
                'processing_logic': [
                    '1. 默认值为0（无氯离子）',
                    '2. 检查溶液成分文本中的氯离子关键词',
                    '3. 关键词：cl, chloride, nacl, cacl2, mgcl2, salt',
                    '4. 海水环境自动设置为1',
                    '5. 文本匹配算法'
                ],
                'default_value': '0',
                'complexity': '中 - 文本关键词匹配'
            },
            'dataset_code_method': {
                'source_columns': ['ingredient_1', 'ingredient_2', 'temp2'],
                'processing_logic': [
                    '1. 直接从固定列位置读取',
                    '2. 按优先级：索引61 → 索引64 → 索引77',
                    '3. 保持原始分类值',
                    '4. 无文本解析'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 简单分类提取'
            },
            'key_differences': [
                '• app.py进行智能文本解析',
                '• app.py输出二元指示器(0/1)',
                '• dataset_code保持原始分类值',
                '• app.py有专门的化学知识库'
            ]
        },
        {
            'id': 3,
            'feature_name': 'concrete',
            'description': '混凝土环境指示器',
            'app_py_method': {
                'source_columns': ['concrete', 'crack', 'cover', 'Condition_environment'],
                'processing_logic': [
                    '1. 默认值为0（非混凝土环境）',
                    '2. 检查混凝土相关列是否有值',
                    '3. 检查环境描述中的混凝土关键词',
                    '4. 关键词：concrete, cover, crack, cement, mortar',
                    '5. 任一条件满足即设为1'
                ],
                'default_value': '0',
                'complexity': '中 - 多列条件判断'
            },
            'dataset_code_method': {
                'source_columns': ['concrete', 'crack', 'cover'],
                'processing_logic': [
                    '1. 直接从固定列位置读取',
                    '2. 按优先级：索引53 → 索引56 → 索引57',
                    '3. 保持原始分类值',
                    '4. 无逻辑判断'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 简单分类提取'
            },
            'key_differences': [
                '• app.py综合多个信息源',
                '• app.py输出二元指示器',
                '• app.py有环境类型推断',
                '• dataset_code仅单列提取'
            ]
        },
        {
            'id': 4,
            'feature_name': 'diameter',
            'description': '纤维直径(mm)',
            'app_py_method': {
                'source_columns': ['diameter', 'nominal_area'],
                'processing_logic': [
                    '1. 优先使用直接测量的直径值',
                    '2. 如果无直径值，从nominal_area计算',
                    '3. 计算公式：diameter = 2 * sqrt(area / π)',
                    '4. 数值验证（必须>0）'
                ],
                'default_value': 'NaN',
                'complexity': '中 - 几何计算'
            },
            'dataset_code_method': {
                'source_columns': ['diameter', 'nominal_area'],
                'processing_logic': [
                    '1. 直接从固定列位置读取',
                    '2. 按优先级：索引18 → 索引20',
                    '3. 使用第一个非空数值',
                    '4. 无计算转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 简单数值提取'
            },
            'key_differences': [
                '• app.py支持面积到直径的转换',
                '• app.py有数学计算能力',
                '• dataset_code仅直接提取',
                '• app.py更智能的数据填补'
            ]
        },
        {
            'id': 5,
            'feature_name': 'load_value',
            'description': '载荷比例值',
            'app_py_method': {
                'source_columns': ['type_of_load', 'stress_or_strain', 'value_load', 'ultimate_tensile_strength', 'tensile_modulus'],
                'processing_logic': [
                    '1. 检查是否为preloading（设为0）',
                    '2. 应力情况：load_value = stress / ultimate_tensile_strength',
                    '3. 应变情况：load_value = strain * 0.001 * modulus / UTS',
                    '4. 复杂的工程计算',
                    '5. 单位换算和归一化'
                ],
                'default_value': '0',
                'complexity': '高 - 复杂工程计算'
            },
            'dataset_code_method': {
                'source_columns': ['tensile_modulus'],
                'processing_logic': [
                    '1. 直接从索引90读取数值',
                    '2. 无任何计算或转换',
                    '3. 保持原始值'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接提取'
            },
            'key_differences': [
                '• app.py进行复杂的工程力学计算',
                '• app.py考虑载荷类型和单位',
                '• app.py输出归一化比例值',
                '• dataset_code仅提取模量值'
            ]
        },
        {
            'id': 6,
            'feature_name': 'fiber_content',
            'description': '纤维含量(%)',
            'app_py_method': {
                'source_columns': ['Fiber_content_weight', 'Fiber_content_volume', 'Fiber_type', 'Matrix_type'],
                'processing_logic': [
                    '1. 优先使用重量百分比',
                    '2. 如果只有体积百分比，进行密度转换',
                    '3. 查找材料密度表：Glass(2.55), Carbon(1.84), Basalt(2.67)',
                    '4. 基质密度：Vinyl ester(1.09), Epoxy(1.1), Polyester(1.38)',
                    '5. 公式：Wf = (100 * Vf * ρf) / (Vf * ρf + (100-Vf) * ρm)',
                    '6. 智能材料识别和密度匹配'
                ],
                'default_value': 'NaN',
                'complexity': '高 - 材料科学计算'
            },
            'dataset_code_method': {
                'source_columns': ['Fiber_content_weight', 'Fiber_content_volume'],
                'processing_logic': [
                    '1. 按优先级：索引15 → 索引16',
                    '2. 直接提取数值',
                    '3. 无单位转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接数值提取'
            },
            'key_differences': [
                '• app.py支持体积到重量的转换',
                '• app.py有材料密度数据库',
                '• app.py进行复杂的材料科学计算',
                '• dataset_code无单位转换能力'
            ]
        },
        {
            'id': 7,
            'feature_name': 'Glass_or_Basalt',
            'description': '纤维类型编码',
            'app_py_method': {
                'source_columns': ['Fiber_type'],
                'processing_logic': [
                    '1. 文本匹配纤维类型',
                    '2. Glass → 1',
                    '3. Basalt → 0',
                    '4. 其他类型保持NaN',
                    '5. 大小写不敏感匹配'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 简单分类编码'
            },
            'dataset_code_method': {
                'source_columns': ['Fiber_type'],
                'processing_logic': [
                    '1. 直接从索引8提取',
                    '2. 保持原始文本值',
                    '3. 无编码转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接文本提取'
            },
            'key_differences': [
                '• app.py进行数值编码',
                '• app.py输出二元值(0/1)',
                '• dataset_code保持文本值',
                '• app.py便于机器学习使用'
            ]
        },
        {
            'id': 8,
            'feature_name': 'Vinyl_ester_or_Epoxy',
            'description': '基质类型编码',
            'app_py_method': {
                'source_columns': ['Matrix_type'],
                'processing_logic': [
                    '1. 文本匹配基质类型',
                    '2. Vinyl ester → 1',
                    '3. Epoxy → 0',
                    '4. 其他类型保持NaN'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 简单分类编码'
            },
            'dataset_code_method': {
                'source_columns': ['Matrix_type'],
                'processing_logic': [
                    '1. 直接从索引10提取',
                    '2. 保持原始文本值',
                    '3. 无编码转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接文本提取'
            },
            'key_differences': [
                '• app.py进行数值编码',
                '• app.py输出二元值(0/1)',
                '• dataset_code保持文本值',
                '• 编码方式相同但输出格式不同'
            ]
        },
        {
            'id': 9,
            'feature_name': 'condition_time',
            'description': '条件时间(天/小时)',
            'app_py_method': {
                'source_columns': ['time_field'],
                'processing_logic': [
                    '1. 直接从time_field复制数值',
                    '2. 验证数值有效性',
                    '3. 字符串数值尝试转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接复制'
            },
            'dataset_code_method': {
                'source_columns': ['time_field', 'UV'],
                'processing_logic': [
                    '1. 按优先级：索引51 → 索引84',
                    '2. 直接提取数值',
                    '3. 无验证'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接提取'
            },
            'key_differences': [
                '• app.py有数值验证',
                '• dataset_code有备选列',
                '• 处理逻辑基本相同',
                '• app.py支持字符串转数值'
            ]
        },
        {
            'id': 10,
            'feature_name': 'Temperature',
            'description': '温度(°C)',
            'app_py_method': {
                'source_columns': ['temperature'],
                'processing_logic': [
                    '1. 直接从temperature复制数值',
                    '2. 验证数值有效性',
                    '3. 字符串数值尝试转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接复制'
            },
            'dataset_code_method': {
                'source_columns': ['temperature', 'field_average_temperature', 'RH_2'],
                'processing_logic': [
                    '1. 按优先级：索引49 → 索引69 → 索引78',
                    '2. 直接提取数值',
                    '3. 注意：索引78是RH_2，可能不是温度'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接提取'
            },
            'key_differences': [
                '• dataset_code有多个备选列',
                '• app.py只用主温度列',
                '• dataset_code的索引78(RH_2)可能错误',
                '• app.py更精确的列选择'
            ]
        },
        {
            'id': 11,
            'feature_name': 'Tensile_strength_retention',
            'description': '拉伸强度保持率',
            'app_py_method': {
                'source_columns': ['retention1'],
                'processing_logic': [
                    '1. 直接从retention1复制数值',
                    '2. 验证数值有效性(0-1或0-100%)',
                    '3. 可能的百分比转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接复制+验证'
            },
            'dataset_code_method': {
                'source_columns': ['COV2_2', 'COV3_2', 'water_absorption_ratio'],
                'processing_logic': [
                    '1. 按优先级：索引100 → 索引104 → 索引108',
                    '2. 直接提取数值',
                    '3. 注意：这些列是变异系数和吸水率，不是强度保持率'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接提取（但列选择可能错误）'
            },
            'key_differences': [
                '• app.py使用正确的retention1列',
                '• dataset_code使用的列不是强度保持率',
                '• app.py更准确的数据源',
                '• dataset_code的映射可能有误'
            ]
        },
        {
            'id': 12,
            'feature_name': 'surface_treatment',
            'description': '表面处理',
            'app_py_method': {
                'source_columns': ['surface_treatment'],
                'processing_logic': [
                    '1. 文本匹配表面处理类型',
                    '2. "sand coated" → 0',
                    '3. "Smooth" → 1',
                    '4. 其他保持原值'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 文本编码'
            },
            'dataset_code_method': {
                'source_columns': ['surface_treatment'],
                'processing_logic': [
                    '1. 直接从索引22提取',
                    '2. 保持原始文本值',
                    '3. 无编码转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接文本提取'
            },
            'key_differences': [
                '• app.py进行文本编码',
                '• app.py输出数值(0/1)',
                '• dataset_code保持文本值',
                '• 数据源相同但处理不同'
            ]
        },
        {
            'id': 13,
            'feature_name': 'glass_transition_temperature',
            'description': '玻璃化转变温度(°C)',
            'app_py_method': {
                'source_columns': ['glass_transition_temperature'],
                'processing_logic': [
                    '1. 直接从glass_transition_temperature复制',
                    '2. 验证数值有效性',
                    '3. 字符串数值尝试转换'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接复制'
            },
            'dataset_code_method': {
                'source_columns': ['glass_transition_temperature', 'COV_2', 'glass_transition_temperature_run_2'],
                'processing_logic': [
                    '1. 按优先级：索引12 → 索引114 → 索引13',
                    '2. 直接提取数值',
                    '3. 注意：索引114是COV_2，不是温度'
                ],
                'default_value': 'NaN',
                'complexity': '低 - 直接提取'
            },
            'key_differences': [
                '• app.py使用单一正确列',
                '• dataset_code包含错误的备选列(COV_2)',
                '• app.py更准确的数据源',
                '• dataset_code的备选策略有问题'
            ]
        }
    ]
    
    return feature_comparisons

def print_detailed_comparison():
    """打印详细对比结果"""
    comparisons = analyze_feature_extraction_methods()
    
    for comp in comparisons:
        print(f"\n{'='*80}")
        print(f"特征 {comp['id']:2d}: {comp['feature_name']}")
        print(f"描述: {comp['description']}")
        print(f"{'='*80}")
        
        # app.py方法
        print(f"\n🔧 app.py 处理方法:")
        print(f"   数据源列: {comp['app_py_method']['source_columns']}")
        print(f"   复杂度: {comp['app_py_method']['complexity']}")
        print(f"   默认值: {comp['app_py_method']['default_value']}")
        print(f"   处理逻辑:")
        for logic in comp['app_py_method']['processing_logic']:
            print(f"     {logic}")
        
        # dataset_code方法
        print(f"\n📋 dataset_code 处理方法:")
        print(f"   数据源列: {comp['dataset_code_method']['source_columns']}")
        print(f"   复杂度: {comp['dataset_code_method']['complexity']}")
        print(f"   默认值: {comp['dataset_code_method']['default_value']}")
        print(f"   处理逻辑:")
        for logic in comp['dataset_code_method']['processing_logic']:
            print(f"     {logic}")
        
        # 关键差异
        print(f"\n🔍 关键差异:")
        for diff in comp['key_differences']:
            print(f"   {diff}")

def summarize_overall_differences():
    """总结整体差异"""
    print(f"\n\n{'='*80}")
    print("📊 整体差异总结")
    print(f"{'='*80}")
    
    categories = [
        {
            'title': '处理复杂度',
            'app_py': '高 - 包含复杂的工程计算、文本解析、逻辑判断',
            'dataset_code': '低 - 主要是简单的数值/文本提取'
        },
        {
            'title': '数据源策略',
            'app_py': '列名驱动 - 基于语义化列名进行智能处理',
            'dataset_code': '位置驱动 - 基于固定索引位置提取'
        },
        {
            'title': '容错能力',
            'app_py': '强 - 多重备选方案、默认值、数值验证',
            'dataset_code': '弱 - 依赖固定结构，易因文件变化失效'
        },
        {
            'title': '工程知识',
            'app_py': '丰富 - 包含材料密度、力学计算、化学知识',
            'dataset_code': '有限 - 基础的数据提取'
        },
        {
            'title': '输出格式',
            'app_py': '机器学习友好 - 数值编码、归一化处理',
            'dataset_code': '原始数据保持 - 保持Excel原始格式'
        },
        {
            'title': '维护性',
            'app_py': '高 - 适应性强，代码逻辑清晰',
            'dataset_code': '低 - 硬编码索引，易受文件结构影响'
        }
    ]
    
    for cat in categories:
        print(f"\n🔹 {cat['title']}:")
        print(f"   app.py: {cat['app_py']}")
        print(f"   dataset_code: {cat['dataset_code']}")
    
    print(f"\n💡 关键发现:")
    findings = [
        "• app.py采用智能特征工程方法，包含丰富的领域知识",
        "• dataset_code采用简单映射策略，更接近原始数据",
        "• app.py更适合机器学习应用，有完整的数据预处理",
        "• dataset_code更适合数据存储和基础分析",
        "• 两者在BU列后的列映射存在差异，需要对齐",
        "• app.py的方法更robust但计算开销更大"
    ]
    
    for finding in findings:
        print(f"   {finding}")

if __name__ == "__main__":
    print_detailed_comparison()
    summarize_overall_differences()