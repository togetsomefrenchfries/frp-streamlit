import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '666666',  # ⚠️ 请修改为您的MySQL密码
    'database': 'haigui_database',
    'charset': 'utf8mb4'
}

# Excel文件的完整路径
EXCEL_FILE_PATH = 'D:/haigui/database 1.xlsx'

# 基于Excel实际位置的列名映射（132个字段）
MYSQL_COLUMNS = [
    'feature_name',                    # 位置1
    'Title',                          # 位置2
    'Author',                         # 位置3
    'SCI',                           # 位置4
    'Journal_or_Conference_name',     # 位置5
    'Year',                          # 位置6
    'No_field',                      # 位置7
    'no_field_secondary',            # 位置8
    'Fiber_type',                    # 位置9
    'Fiber_type_detail',             # 位置10
    'Matrix_type',                   # 位置11
    'Matrix_type_detail',            # 位置12
    'glass_transition_temperature',   # 位置13
    'glass_transition_temperature_run_2', # 位置14
    'cure_ratio',                    # 位置15
    'Fiber_content_weight',          # 位置16
    'Fiber_content_volume',          # 位置17
    'Void_content',                  # 位置18
    'diameter',                      # 位置19
    'average_area',                  # 位置20
    'nominal_area',                  # 位置21
    'rib',                          # 位置22
    'surface_treatment',             # 位置23
    'Water_absorption_at_saturation', # 位置24
    'Water_absorption_test_standard', # 位置25
    'Water_absorption_note',         # 位置26
    'Brand_name',                    # 位置27
    'Manufacturer',                  # 位置28
    'Important_notes',               # 位置29
    'Notes_of_rebar',               # 位置30
    'Target_parameter',              # 位置31
    'note_of_target_parameter',      # 位置32
    'num_1',                        # 位置33
    'note_of_number',               # 位置34
    'Value1_1',                     # 位置35
    'COV1_1',                       # 位置36
    'note_of_Value1',               # 位置37
    'Value2_1',                     # 位置38
    'COV2_1',                       # 位置39
    'Value2note_1',                 # 位置40
    'Value3_1',                     # 位置41
    'COV3_1',                       # 位置42
    'Value3note_1',                 # 位置43
    'SEM_T_BCBT',                   # 位置44
    'SEM_L_BCBT',                   # 位置45
    'OTHER_main',                   # 位置46
    'OTHER1_1',                     # 位置47
    'FTIR_1',                       # 位置48
    'note_1',                       # 位置49
    'temperature',                   # 位置50
    'note_of_temperature',           # 位置51
    'time_field',                    # 位置52
    'note_of_time',                  # 位置53
    'concrete',                      # 位置54
    'pH_of_concrete',                # 位置55
    'strength_of_concrete',          # 位置56
    'crack',                         # 位置57
    'cover',                         # 位置58
    'note_of_concrete',              # 位置59
    'pH_1',                          # 位置60
    'pHafter',                       # 位置61
    'ingredient_1',                  # 位置62
    'pH_2',                          # 位置63
    'RH_1',                          # 位置64
    'ingredient_2',                  # 位置65
    'note_2',                        # 位置66
    'Location',                      # 位置67
    'Effektive_Klimaklassifikation', # 位置68
    'field_average_humidity',        # 位置69
    'field_average_temperature',     # 位置70
    'number_field',                  # 位置71
    'type_field',                    # 位置72
    'SolutionorMoisture',            # 位置73
    'cycle_pH',                      # 位置74
    'cycle_pH_after',                # 位置75
    'cycle_ingredient',              # 位置76
    'temp',                          # 位置77
    'temp2',                         # 位置78
    'RH_2',                          # 位置79
    'RH2',                           # 位置80
    'OTHER1_2',                      # 位置81
    'OTHER2_main',                   # 位置82
    'time_in_cycle',                 # 位置83
    'note_3',                        # 位置84
    'UV',                            # 位置85
    'note_4',                        # 位置86
    'stress_or_strain',              # 位置87
    'type_of_load',                  # 位置88
    'value_load',                    # 位置89
    'ultimate_tensile_strength',     # 位置90
    'tensile_modulus',               # 位置91
    'note_5',                        # 位置92
    'after_condition',               # 位置93
    'note_6',                        # 位置94
    'num_2',                         # 位置95
    'Value1_2',                      # 位置96
    'COV1_2',                        # 位置97
    'Value1note',                    # 位置98
    'retention1',                    # 位置99
    'Value2_2',                      # 位置100
    'COV2_2',                        # 位置101
    'Value2note_2',                  # 位置102
    'retention2',                    # 位置103
    'Value3_2',                      # 位置104
    'COV3_2',                        # 位置105
    'Value3note_2',                  # 位置106
    'retention3',                    # 位置107
    'num_3',                         # 位置108
    'water_absorption_ratio',        # 位置109
    'COV_1',                         # 位置110
    'note_7',                        # 位置111
    'num_4',                         # 位置112
    'glass_transition_temperature_2', # 位置113
    'run2',                          # 位置114
    'COV_2',                         # 位置115
    'cure_ratio_2',                  # 位置116
    'note_8',                        # 位置117
    'num_5',                         # 位置118
    'OTHERS',                        # 位置119
    'OTHERS_note',                   # 位置120
    'SEM_T_BCAT',                    # 位置121
    'SEM_L_BCAT',                    # 位置122
    'SEM_T_ACBT',                    # 位置123
    'SEM_L_ACBT',                    # 位置124
    'SEM_T_ACAT',                    # 位置125
    'SEM_L_ACAT',                    # 位置126
    'other_lower',                   # 位置127
    'other2_final',                  # 位置128
    'note_9',                        # 位置129
    'FTIR_2',                        # 位置130
    'note_10',                       # 位置131
    'important_note'                 # 位置132
]

def test_connection():
    """测试数据库连接"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            logger.info("✅ 数据库连接测试成功")
            connection.close()
            return True
    except Error as e:
        logger.error(f"❌ 数据库连接失败: {e}")
        return False

def check_files():
    """检查文件是否存在"""
    logger.info(f"📍 检查文件: {EXCEL_FILE_PATH}")
    
    if os.path.exists(EXCEL_FILE_PATH):
        logger.info("✅ 找到Excel文件")
        file_size = os.path.getsize(EXCEL_FILE_PATH) / (1024*1024)  # MB
        logger.info(f"📊 文件大小: {file_size:.2f} MB")
        return True
    else:
        logger.error("❌ Excel文件不存在")
        logger.error(f"请确保文件位于: {EXCEL_FILE_PATH}")
        return False

def read_excel_data():
    """读取Excel数据"""
    try:
        logger.info("📖 正在读取Excel文件...")
        df = pd.read_excel(EXCEL_FILE_PATH, header=3, engine='openpyxl')
        logger.info(f"✅ Excel读取成功，数据形状: {df.shape}")
        
        # 只取前132列
        df = df.iloc[:, :132]
        logger.info(f"使用前132列，调整后形状: {df.shape}")
        
        return df
    except Exception as e:
        logger.error(f"❌ Excel读取失败: {e}")
        return None

def clean_data(df):
    """清理数据"""
    logger.info("🧹 正在清理数据...")
    
    df_clean = df.copy()
    
    # 处理特殊值
    special_values = ['SMD', 'Notreported', 'N/A', '', ' ', 'nan', 'NULL', 'None']
    df_clean = df_clean.replace(special_values, None)
    df_clean = df_clean.replace({np.nan: None})
    
    # 处理数值字段
    numeric_positions = [5, 18, 34, 35]  # Year, diameter, Value1_1, COV1_1
    for pos in numeric_positions:
        if pos < len(df_clean.columns):
            df_clean.iloc[:, pos] = pd.to_numeric(df_clean.iloc[:, pos], errors='coerce')
    
    # 处理百分比字段
    retention_positions = [98, 102, 106]  # retention1, retention2, retention3
    for pos in retention_positions:
        if pos < len(df_clean.columns):
            df_clean.iloc[:, pos] = df_clean.iloc[:, pos].astype(str).str.replace('%', '').str.replace('nan', '')
    
    # 限制文本长度
    for col_idx in range(len(df_clean.columns)):
        if df_clean.iloc[:, col_idx].dtype == 'object':
            df_clean.iloc[:, col_idx] = df_clean.iloc[:, col_idx].astype(str).str[:2000]
            df_clean.iloc[:, col_idx] = df_clean.iloc[:, col_idx].replace('None', None)
    
    logger.info("✅ 数据清理完成")
    return df_clean

def insert_data(df):
    """插入数据到MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        # —— 第一步：清空表 —— 
        cursor.execute("TRUNCATE TABLE research_data")
        connection.commit()
        logger.info("🧹 已清空表 research_data，准备插入新数据")
        # 使用所有132列
        columns_str = ', '.join([f"`{col}`" for col in MYSQL_COLUMNS])
        placeholders = ', '.join(['%s'] * len(MYSQL_COLUMNS))
        query = f"INSERT INTO research_data ({columns_str}) VALUES ({placeholders})"
        
        logger.info(f"📝 准备插入 {len(MYSQL_COLUMNS)} 个字段到 {len(df)} 行数据")
        
        # 准备数据
        data_rows = []
        for _, row in df.iterrows():
            row_data = []
            for i in range(len(MYSQL_COLUMNS)):
                if i < len(row):
                    value = row.iloc[i]
                    
                    if pd.isna(value) or value is None:
                        row_data.append(None)
                    elif isinstance(value, (int, float)) and not np.isnan(value):
                        row_data.append(value)
                    else:
                        str_value = str(value).strip()
                        if str_value in ['nan', 'None', 'NULL', '', 'SMD', 'Notreported']:
                            row_data.append(None)
                        else:
                            row_data.append(str_value)
                else:
                    row_data.append(None)
            
            data_rows.append(tuple(row_data))
        
        # 批量插入
        batch_size = 500
        total_rows = len(data_rows)
        inserted = 0
        
        logger.info(f"🚀 开始插入 {total_rows} 行数据...")
        
        for i in range(0, total_rows, batch_size):
            batch = data_rows[i:i + batch_size]
            try:
                cursor.executemany(query, batch)
                connection.commit()
                inserted += len(batch)
                
                progress = (inserted / total_rows) * 100
                logger.info(f"📊 进度: {inserted}/{total_rows} ({progress:.1f}%)")
                
            except Error as batch_error:
                logger.warning(f"批次插入失败，尝试单行插入: {batch_error}")
                connection.rollback()
                
                for j, row in enumerate(batch):
                    try:
                        cursor.execute(query, row)
                        connection.commit()
                        inserted += 1
                    except Error as single_error:
                        logger.error(f"单行插入失败 (行 {i+j+1}): {single_error}")
        
        logger.info(f"✅ 数据插入完成！共插入 {inserted} 行")
        
        cursor.close()
        connection.close()
        return inserted > 0
        
    except Error as e:
        logger.error(f"❌ 数据插入失败: {e}")
        return False

def verify_data():
    """验证插入的数据"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM research_data")
        count = cursor.fetchone()[0]
        logger.info(f"📊 数据库中总记录数: {count}")
        
        # 显示样本数据
        cursor.execute("""
            SELECT Title, Author, Year, Fiber_type 
            FROM research_data 
            WHERE Title IS NOT NULL 
            LIMIT 3
        """)
        
        samples = cursor.fetchall()
        logger.info("📋 样本数据:")
        for i, (title, author, year, fiber) in enumerate(samples, 1):
            title_short = title[:50] + "..." if title and len(title) > 50 else title
            logger.info(f"  {i}. {title_short}")
            logger.info(f"     作者: {author} | 年份: {year} | 纤维: {fiber}")
        
        cursor.close()
        connection.close()
        return True
        
    except Error as e:
        logger.error(f"❌ 数据验证失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🎯 开始Excel到MySQL数据迁移")
    logger.info("=" * 70)
    
    # 步骤1: 检查文件
    logger.info("步骤1: 检查Excel文件")
    if not check_files():
        return
    
    # 步骤2: 测试连接
    logger.info("步骤2: 测试数据库连接")
    if not test_connection():
        logger.error("❌ 请检查MySQL密码和服务状态")
        return
    
    # 步骤3: 读取Excel
    logger.info("步骤3: 读取Excel数据")
    df = read_excel_data()
    if df is None:
        return
    
    # 步骤4: 清理数据
    logger.info("步骤4: 清理数据")
    df_clean = clean_data(df)
    
    # 步骤5: 插入数据
    logger.info("步骤5: 插入数据到MySQL")
    if not insert_data(df_clean):
        logger.error("❌ 数据插入失败")
        return
    
    # 步骤6: 验证数据
    logger.info("步骤6: 验证数据")
    if verify_data():
        logger.info("=" * 70)
        logger.info("🎉 数据迁移成功完成！")
        logger.info("💡 您现在可以在MySQL Workbench中查看和查询数据")

if __name__ == "__main__":
    main()