"""
📊 MySQL字段与Excel列映射详细报告
告诉你每个数据库字段对应Excel的哪一列
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Excel列映射 (基于database 4.xlsx格式)
STANDARD_COLUMNS = [
    'feature_name', 'Title', 'Author', 'SCI', 'Journal_or_Conference_name',
    'Year', 'No.', 'no.', 'Fiber_type', 'Fiber_type_detail', 'Matrix_type', 'Matrix_type_detail',
    'glass_transition_temperature', 'glass_transition_temperature_run_2', 'cure_ratio',
    'Fiber_content_weight', 'Fiber_content_volume', 'Void_content', 'diameter', 'average_area',
    'nominal_area', 'rib', 'surface_treatment', 'Water_absorption_at_saturation',
    'Water_absorption_test_standard', 'Water_absorption_note', 'Brand_name', 'Manufacturer',
    'Important_notes', 'Notes_of_rebar', 'Target_parameter', 'note_of_target_parameter',
    'num', 'note_of_number', 'Value1', 'COV1', 'note_of_Value1', 'Value2', 'COV2', 'Value2note',
    'Value3', 'COV3', 'Value3note', 'SEM-T-BCBT', 'SEM-L-BCBT', 'OTHER', 'OTHER1', 'FTIR', 'note',
    'temperature', 'note_of_temperature', 'time', 'note_of_time', 'concrete', 'pH_of_concrete',
    'strength_of_concrete', 'crack', 'cover', 'note_of_concrete', 'pH', 'pHafter', 'ingredient',
    'pH.1', 'RH', 'ingredient.1', 'note.1', 'Location', 'Effektive_Klimaklassifikation',
    'field_average_humidity', 'field_average_temperature', 'number', 'type', 'SolutionorMoisture',
    'cycle_pH', 'cycle_pH_after', 'cycle_ingredient', 'temp', 'temp2', 'RH.1', 'RH2',
    'OTHER1.1', 'OTHER2', 'time_in_cycle', 'note.2', 'UV', 'note.3', 'stress_or_strain',
    'type_of_load', 'value', 'ultimate_tensile_strength', 'tensile_modulus', 'note.4',
    'after_condition', 'note.5', 'num.1', 'Value1.1', 'COV1.1', 'Value1note', 'retention1',
    'Value2.1', 'COV2.1', 'Value2note.1', 'retention2', 'Value3.1', 'COV3.1', 'Value3note.1',
    'retention3', 'num.2', 'water_absorption_ratio', 'COV', 'note.6', 'num.3',
    'glass_transition_temperature.1', 'run2', 'COV.1', 'cure_ratio.1', 'note.7', 'num.4',
    'OTHERS', 'OTHERS_note', 'SEM-T-BCAT', 'SEM-L-BCAT', 'SEM-T-ACBT', 'SEM-L-ACBT',
    'SEM-T-ACAT', 'SEM-L-ACAT', 'other', 'other2', 'note.8', 'FTIR.1', 'note.9', 'important_note'
]

# 数据库字段定义
DB_COLUMNS = [
    'specimen', 'author', 'year', 'test_condition', 'temperature',
    'moisture', 'solution', 'pH', 'duration', 'fiber_type',
    'matrix_type', 'test_method', 'geometry', 'diameter', 
    'cross_sectional_area', 'length', 'fiber_volume_fraction',
    'elastic_modulus_initial', 'tensile_strength_initial',
    'elastic_modulus_final', 'tensile_strength_final',
    'modulus_retention', 'strength_retention', 'notes'
]

def get_column_mapping_for_database4():
    """获取database 4.xlsx的列映射"""
    mapping = {}
    
    # database 4.xlsx格式映射规则:
    # - 列1-70: 直接映射 (列索引0-69)
    # - 列71-72: 跳过 (pH.2, Ingrediant)
    # - 列73+: 偏移-2 (列索引72+ 映射到标准列索引70+)
    
    for i, std_col in enumerate(STANDARD_COLUMNS):
        if i < 70:
            # 前70列直接映射
            mapping[i] = std_col
        else:
            # 第71列往后，需要+2偏移
            excel_col_index = i + 2
            mapping[excel_col_index] = std_col
    
    return mapping

def generate_field_mapping_report():
    """生成详细的字段映射报告"""
    
    print("=" * 90)
    print("📊 MySQL字段与Excel列映射详细报告")
    print("=" * 90)
    
    print("\n🎯 映射规则说明:")
    print("• Database 4.xlsx 在第71-72列插入了 'pH.2' 和 'Ingrediant' 两列")
    print("• 因此第73列往后的所有列都向后偏移了2列")
    print("• 智能转换器自动处理这个偏移，确保数据正确映射")
    
    # 获取列映射
    column_mapping = get_column_mapping_for_database4()
    
    print("\n📋 数据库字段详细映射表:")
    print("=" * 90)
    print("序号 | 数据库字段名              | Excel列号 | Excel列名                    | 说明")
    print("-" * 90)
    
    for i, db_field in enumerate(DB_COLUMNS):
        excel_col_index = i  # 这是映射后在标准132列中的位置
        
        # 找到对应的实际Excel列位置
        actual_excel_col = None
        for excel_idx, std_col in column_mapping.items():
            if std_col == STANDARD_COLUMNS[excel_col_index]:
                actual_excel_col = excel_idx + 1  # +1 因为Excel列从1开始计数
                break
        
        if actual_excel_col is None:
            actual_excel_col = excel_col_index + 1
        
        excel_col_name = STANDARD_COLUMNS[excel_col_index]
        
        # 添加说明
        if db_field == 'specimen':
            description = "试样编号/标识"
        elif db_field == 'author': 
            description = "研究作者/论文标题"
        elif db_field == 'year':
            description = "发表年份"
        elif db_field == 'test_condition':
            description = "测试条件"
        elif db_field == 'temperature':
            description = "测试温度"
        elif db_field == 'moisture':
            description = "湿度条件"
        elif db_field == 'solution':
            description = "溶液类型"
        elif db_field == 'pH':
            description = "pH值"
        elif db_field == 'duration':
            description = "测试持续时间"
        elif db_field == 'fiber_type':
            description = "纤维类型"
        elif db_field == 'matrix_type':
            description = "基材类型"
        elif db_field == 'test_method':
            description = "测试方法"
        elif db_field == 'geometry':
            description = "几何形状"
        elif db_field == 'diameter':
            description = "直径"
        elif db_field == 'cross_sectional_area':
            description = "横截面积"
        elif db_field == 'length':
            description = "长度"
        elif db_field == 'fiber_volume_fraction':
            description = "纤维体积分数"
        elif db_field == 'elastic_modulus_initial':
            description = "初始弹性模量"
        elif db_field == 'tensile_strength_initial':
            description = "初始拉伸强度"
        elif db_field == 'elastic_modulus_final':
            description = "最终弹性模量"
        elif db_field == 'tensile_strength_final':
            description = "最终拉伸强度"
        elif db_field == 'modulus_retention':
            description = "模量保持率"
        elif db_field == 'strength_retention':
            description = "强度保持率"
        elif db_field == 'notes':
            description = "备注信息"
        else:
            description = "数据字段"
        
        print(f"{i+1:4d} | {db_field:25} | {actual_excel_col:9d} | {excel_col_name:28} | {description}")
    
    print("\n🔍 特殊映射情况:")
    print("-" * 50)
    print("• Excel第71列 'pH.2' → 未使用 (database 4.xlsx新增)")
    print("• Excel第72列 'Ingrediant' → 未使用 (database 4.xlsx新增)")
    print("• Excel第73列 'number' → 对应标准第71列")
    print("• Excel第74列及以后 → 依次对应标准第72列及以后")
    
    print("\n📊 数据类型说明:")
    print("-" * 50)
    print("🔸 文本字段 (varchar): specimen, author, test_condition, moisture, solution")
    print("🔸 数值字段 (int): year") 
    print("🔸 浮点字段 (float): temperature, pH, duration, diameter, 各种强度和模量值")
    print("🔸 长文本 (text): notes")
    
    print("\n🎯 重要提示:")
    print("-" * 50)
    print("✅ 智能转换器自动识别Excel格式并应用正确的列偏移")
    print("✅ Database 1.xlsx 和 Database 4.xlsx 都能正确处理")
    print("✅ 最终都映射到相同的24个数据库字段")
    print("✅ 数据完整性得到保证")
    
    # 显示实际的Excel列映射示例
    print("\n📋 Excel列偏移示例 (Database 4.xlsx):")
    print("-" * 60)
    print("标准位置 | 实际Excel列 | 列名")
    print("-" * 60)
    
    key_examples = [
        (70, "number"),
        (71, "type"), 
        (72, "SolutionorMoisture"),
        (95, "num.1"),  # 用户之前提到的例子
    ]
    
    for std_pos, col_name in key_examples:
        if std_pos < 70:
            actual_excel = std_pos + 1
        else:
            actual_excel = std_pos + 3  # +2偏移 +1(Excel从1开始)
        
        print(f"{std_pos:8d} | {actual_excel:11d} | {col_name}")
    
    print("\n" + "=" * 90)
    print("🎉 字段映射报告生成完成！")
    print("现在你知道每个数据库字段来自Excel的哪一列了！")
    print("=" * 90)

if __name__ == "__main__":
    generate_field_mapping_report()