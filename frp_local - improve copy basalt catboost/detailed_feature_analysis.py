#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13个特征的详细数据提取和转化规则分析
"""

def analyze_each_feature_extraction():
    """详细分析每个特征的数据提取和转化规则"""
    
    print("🔍 13个特征的数据提取和转化规则详细分析")
    print("=" * 80)
    
    # 13个特征的详细配置
    features_detail = [
        {
            'id': 1,
            'name': 'pH_of_condition_enviroment',
            'description': '环境条件pH值',
            'candidate_columns': {
                'python_indices': [54, 59, 60],
                'excel_columns': ['BC', 'BH', 'BI'],
                'excel_numbers': ['第55列', '第60列', '第61列']
            },
            'data_type': 'numerical',
            'expected_range': '1-14',
            'extraction_rule': '选择第一个有效列 → 转换为数值型 → 过滤异常值',
            'transformation_steps': [
                '1. 按顺序检查候选列：BC列(索引54) → BH列(索引59) → BI列(索引60)',
                '2. 选择第一个存在且列数未超出范围的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 使用pd.to_numeric()转换为数值型，errors="coerce"处理非数值',
                '5. 识别特殊值：SMD、NotReported、N/A等',
                '6. 统计有效数值、缺失值、特殊值的数量',
                '7. 计算统计量：均值、标准差、中位数、分位数等'
            ],
            'final_output': '单个数值型特征列，包含pH值(1-14)或NaN'
        },
        {
            'id': 2,
            'name': 'Chloride_ion',
            'description': '氯离子浓度',
            'candidate_columns': {
                'python_indices': [61, 64, 77],
                'excel_columns': ['BJ', 'BM', 'BZ'],
                'excel_numbers': ['第62列', '第65列', '第78列']
            },
            'data_type': 'categorical',
            'expected_range': '分类值(如高/中/低、具体浓度等)',
            'extraction_rule': '选择第一个有效列 → 保持原始分类 → 统计类别分布',
            'transformation_steps': [
                '1. 按顺序检查候选列：BJ列(索引61) → BM列(索引64) → BZ列(索引77)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 保持原始字符串/分类格式，不进行数值转换',
                '5. 识别特殊值：SMD、NotReported、N/A等',
                '6. 使用value_counts()统计各类别的频次',
                '7. 计算类别分布、最频繁类别等'
            ],
            'final_output': '单个分类型特征列，包含氯离子浓度类别或特殊值'
        },
        {
            'id': 3,
            'name': 'concrete',
            'description': '混凝土环境',
            'candidate_columns': {
                'python_indices': [53, 56, 57],
                'excel_columns': ['BB', 'BE', 'BF'],
                'excel_numbers': ['第54列', '第57列', '第58列']
            },
            'data_type': 'categorical',
            'expected_range': '0/1 或 Yes/No',
            'extraction_rule': '选择第一个有效列 → 保持分类格式 → 二元分类',
            'transformation_steps': [
                '1. 按顺序检查候选列：BB列(索引53) → BE列(索引56) → BF列(索引57)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 保持原始格式(可能是0/1、Yes/No、True/False等)',
                '5. 识别特殊值和缺失值',
                '6. 统计各类别的分布',
                '7. 生成二元分类的频次统计'
            ],
            'final_output': '单个二元分类特征列，表示是否为混凝土环境'
        },
        {
            'id': 4,
            'name': 'diameter',
            'description': '纤维直径',
            'candidate_columns': {
                'python_indices': [18, 20],
                'excel_columns': ['S', 'U'],
                'excel_numbers': ['第19列', '第21列']
            },
            'data_type': 'numerical',
            'expected_range': '毫米(mm)，通常0.1-10mm',
            'extraction_rule': '选择第一个有效列 → 转换为数值型 → 单位标准化',
            'transformation_steps': [
                '1. 按顺序检查候选列：S列(索引18) → U列(索引20)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 使用pd.to_numeric()转换为数值型',
                '5. 识别和处理特殊值、异常值',
                '6. 检查数值范围的合理性(直径应为正数)',
                '7. 计算统计摘要：最小值、最大值、均值等'
            ],
            'final_output': '单个数值型特征列，包含纤维直径(mm)或NaN'
        },
        {
            'id': 5,
            'name': 'load_value',
            'description': '载荷值',
            'candidate_columns': {
                'python_indices': [90],
                'excel_columns': ['CM'],
                'excel_numbers': ['第91列']
            },
            'data_type': 'numerical',
            'expected_range': '力值单位(N、kN等)',
            'extraction_rule': '单一列提取 → 数值转换 → 载荷分析',
            'transformation_steps': [
                '1. 直接使用CM列(索引90) - 只有一个候选列',
                '2. 提取该列的所有数据：data.iloc[:, 90]',
                '3. 使用pd.to_numeric()转换为数值型',
                '4. 识别特殊值和异常值',
                '5. 检查载荷值的合理性(应为正数)',
                '6. 计算载荷分布统计量',
                '7. 分析载荷范围和分布特征'
            ],
            'final_output': '单个数值型特征列，包含载荷值或NaN'
        },
        {
            'id': 6,
            'name': 'fiber_content',
            'description': '纤维含量',
            'candidate_columns': {
                'python_indices': [15, 16],
                'excel_columns': ['P', 'Q'],
                'excel_numbers': ['第16列', '第17列']
            },
            'data_type': 'numerical',
            'expected_range': '百分比(%)，通常0-100%',
            'extraction_rule': '选择第一个有效列 → 数值转换 → 百分比处理',
            'transformation_steps': [
                '1. 按顺序检查候选列：P列(索引15) → Q列(索引16)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 使用pd.to_numeric()转换为数值型',
                '5. 检查百分比范围(应在0-100之间，或0-1之间)',
                '6. 识别和处理异常值',
                '7. 计算纤维含量的分布统计'
            ],
            'final_output': '单个数值型特征列，包含纤维含量百分比或NaN'
        },
        {
            'id': 7,
            'name': 'Glass_or_Basalt',
            'description': '纤维类型',
            'candidate_columns': {
                'python_indices': [8],
                'excel_columns': ['I'],
                'excel_numbers': ['第9列']
            },
            'data_type': 'categorical',
            'expected_range': 'Glass、Basalt等纤维类型',
            'extraction_rule': '单一列提取 → 分类标准化 → 纤维类型分类',
            'transformation_steps': [
                '1. 直接使用I列(索引8) - 只有一个候选列',
                '2. 提取该列的所有数据：data.iloc[:, 8]',
                '3. 保持字符串格式，进行分类分析',
                '4. 标准化纤维类型名称(Glass、Basalt、Carbon等)',
                '5. 识别特殊值和未知类型',
                '6. 统计各纤维类型的频次',
                '7. 计算纤维类型分布比例'
            ],
            'final_output': '单个分类型特征列，包含纤维类型名称或特殊值'
        },
        {
            'id': 8,
            'name': 'Vinyl_ester_or_Epoxy',
            'description': '树脂类型',
            'candidate_columns': {
                'python_indices': [10],
                'excel_columns': ['K'],
                'excel_numbers': ['第11列']
            },
            'data_type': 'categorical',
            'expected_range': 'Vinyl_ester、Epoxy等树脂类型',
            'extraction_rule': '单一列提取 → 分类标准化 → 树脂类型分类',
            'transformation_steps': [
                '1. 直接使用K列(索引10) - 只有一个候选列',
                '2. 提取该列的所有数据：data.iloc[:, 10]',
                '3. 保持字符串格式，进行分类分析',
                '4. 标准化树脂类型名称(Vinyl_ester、Epoxy、Polyester等)',
                '5. 识别特殊值和未知类型',
                '6. 统计各树脂类型的频次',
                '7. 计算树脂类型分布比例'
            ],
            'final_output': '单个分类型特征列，包含树脂类型名称或特殊值'
        },
        {
            'id': 9,
            'name': 'condition_time',
            'description': '条件时间',
            'candidate_columns': {
                'python_indices': [51, 84],
                'excel_columns': ['AZ', 'CM'],
                'excel_numbers': ['第52列', '第85列']
            },
            'data_type': 'numerical',
            'expected_range': '天数(days)或小时(hours)',
            'extraction_rule': '选择第一个有效列 → 数值转换 → 时间单位处理',
            'transformation_steps': [
                '1. 按顺序检查候选列：AZ列(索引51) → CM列(索引84)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 使用pd.to_numeric()转换为数值型',
                '5. 检查时间值的合理性(应为正数)',
                '6. 识别时间单位(days、hours、weeks等)',
                '7. 计算时间分布统计量'
            ],
            'final_output': '单个数值型特征列，包含条件时间(days/hours)或NaN'
        },
        {
            'id': 10,
            'name': 'Temperature',
            'description': '温度',
            'candidate_columns': {
                'python_indices': [49, 69, 78],
                'excel_columns': ['AX', 'BR', 'CA'],
                'excel_numbers': ['第50列', '第70列', '第79列']
            },
            'data_type': 'numerical',
            'expected_range': '摄氏度(°C)，通常-50到200°C',
            'extraction_rule': '选择第一个有效列 → 数值转换 → 温度范围验证',
            'transformation_steps': [
                '1. 按顺序检查候选列：AX列(索引49) → BR列(索引69) → CA列(索引78)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 使用pd.to_numeric()转换为数值型',
                '5. 检查温度范围的合理性(-273°C以上)',
                '6. 识别温度单位(°C、°F、K等)',
                '7. 计算温度分布统计量'
            ],
            'final_output': '单个数值型特征列，包含温度值(°C)或NaN'
        },
        {
            'id': 11,
            'name': 'Tensile_strength_retention',
            'description': '拉伸强度保持率',
            'candidate_columns': {
                'python_indices': [100, 104, 108],
                'excel_columns': ['CW', 'DA', 'DE'],
                'excel_numbers': ['第101列', '第105列', '第109列']
            },
            'data_type': 'numerical',
            'expected_range': '0-1或0-100%',
            'extraction_rule': '选择第一个有效列 → 数值转换 → 比例标准化',
            'transformation_steps': [
                '1. 按顺序检查候选列：CW列(索引100) → DA列(索引104) → DE列(索引108)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 使用pd.to_numeric()转换为数值型',
                '5. 检查保持率范围(0-1或0-100)',
                '6. 如果是百分比格式，转换为0-1范围',
                '7. 计算强度保持率的分布统计'
            ],
            'final_output': '单个数值型特征列，包含强度保持率(0-1)或NaN'
        },
        {
            'id': 12,
            'name': 'surface_treatment',
            'description': '表面处理',
            'candidate_columns': {
                'python_indices': [22],
                'excel_columns': ['W'],
                'excel_numbers': ['第23列']
            },
            'data_type': 'categorical',
            'expected_range': 'Yes/No、True/False、1/0等',
            'extraction_rule': '单一列提取 → 二元分类 → 布尔转换',
            'transformation_steps': [
                '1. 直接使用W列(索引22) - 只有一个候选列',
                '2. 提取该列的所有数据：data.iloc[:, 22]',
                '3. 保持字符串格式，进行分类分析',
                '4. 标准化表面处理状态(Yes/No、有/无等)',
                '5. 识别特殊值和未知状态',
                '6. 统计处理/未处理的比例',
                '7. 计算二元分类分布'
            ],
            'final_output': '单个分类型特征列，包含表面处理状态或特殊值'
        },
        {
            'id': 13,
            'name': 'glass_transition_temperature',
            'description': '玻璃化转变温度',
            'candidate_columns': {
                'python_indices': [12, 114, 13],
                'excel_columns': ['M', 'DK', 'N'],
                'excel_numbers': ['第13列', '第115列', '第14列']
            },
            'data_type': 'numerical',
            'expected_range': '摄氏度(°C)，通常50-200°C',
            'extraction_rule': '选择第一个有效列 → 数值转换 → 温度范围验证',
            'transformation_steps': [
                '1. 按顺序检查候选列：M列(索引12) → DK列(索引114) → N列(索引13)',
                '2. 选择第一个存在的列',
                '3. 提取该列的所有数据：data.iloc[:, selected_index]',
                '4. 使用pd.to_numeric()转换为数值型',
                '5. 检查玻璃化转变温度的合理性(通常50-200°C)',
                '6. 识别异常值和特殊值',
                '7. 计算玻璃化转变温度的分布统计'
            ],
            'final_output': '单个数值型特征列，包含玻璃化转变温度(°C)或NaN'
        }
    ]
    
    # 打印每个特征的详细信息
    for feature in features_detail:
        print(f"\n{'='*80}")
        print(f"特征 {feature['id']:2d}: {feature['name']}")
        print(f"{'='*80}")
        print(f"📋 描述: {feature['description']}")
        print(f"📊 数据类型: {feature['data_type']}")
        print(f"📏 期望范围: {feature['expected_range']}")
        
        print(f"\n🎯 候选数据列:")
        candidate = feature['candidate_columns']
        for i, (py_idx, excel_col, excel_num) in enumerate(zip(
            candidate['python_indices'], 
            candidate['excel_columns'], 
            candidate['excel_numbers']
        )):
            priority = "第一优先" if i == 0 else f"第{i+1}备选"
            print(f"  {priority}: Python索引{py_idx} = Excel {excel_col}列 = {excel_num}")
        
        print(f"\n⚙️ 提取规则: {feature['extraction_rule']}")
        
        print(f"\n🔄 详细转换步骤:")
        for step in feature['transformation_steps']:
            print(f"  {step}")
        
        print(f"\n✅ 最终输出: {feature['final_output']}")

def show_extraction_summary():
    """显示提取规则汇总"""
    print(f"\n\n{'='*80}")
    print("📋 13个特征提取规则汇总")
    print(f"{'='*80}")
    
    summary_data = [
        ("数值型特征", 9, "pH、直径、载荷、纤维含量、条件时间、温度、强度保持率、玻璃化转变温度", 
         "pd.to_numeric() → 统计分析 → 数值范围验证"),
        ("分类型特征", 4, "氯离子、混凝土环境、纤维类型、树脂类型、表面处理", 
         "value_counts() → 分类统计 → 类别分布分析"),
        ("单候选列", 4, "载荷值、纤维类型、树脂类型、表面处理", 
         "直接提取单列，无备选方案"),
        ("多候选列", 9, "其余9个特征", 
         "按优先级顺序选择第一个有效列")
    ]
    
    for category, count, features, process in summary_data:
        print(f"\n🔹 {category} ({count}个):")
        print(f"   特征: {features}")
        print(f"   处理: {process}")

def show_common_transformations():
    """显示通用转换规则"""
    print(f"\n\n{'='*80}")
    print("🔄 通用转换规则")
    print(f"{'='*80}")
    
    transformations = [
        {
            'title': '特殊值处理',
            'description': '所有特征都会识别以下特殊值',
            'values': ['SMD', 'NotReported', 'N/A', 'Unknown', 'Not reported', 'not reported'],
            'action': '统计数量，在分析中排除或单独标记'
        },
        {
            'title': '数值型转换',
            'description': '9个数值型特征的统一处理',
            'values': ['pd.to_numeric(errors="coerce")', '转换失败的值变为NaN', '计算统计量(均值、标准差等)'],
            'action': '生成直方图、箱线图等可视化'
        },
        {
            'title': '分类型处理',
            'description': '4个分类型特征的统一处理',
            'values': ['保持原始字符串格式', 'value_counts()统计频次', '计算类别分布比例'],
            'action': '生成条形图、饼图等可视化'
        },
        {
            'title': '列选择策略',
            'description': '多候选列的选择逻辑',
            'values': ['按索引顺序检查', '选择第一个未超出数据范围的列', '其他候选列被忽略'],
            'action': '返回单一列索引用于数据提取'
        }
    ]
    
    for trans in transformations:
        print(f"\n🔧 {trans['title']}:")
        print(f"   说明: {trans['description']}")
        print(f"   详情: {', '.join(trans['values'])}")
        print(f"   结果: {trans['action']}")

if __name__ == "__main__":
    analyze_each_feature_extraction()
    show_extraction_summary()
    show_common_transformations()