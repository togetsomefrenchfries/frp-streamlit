import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
import logging
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置 - 支持本地/Railway切换
def get_db_config():
    """获取数据库配置 - 优先使用Railway，备用本地"""
    # 检查Railway环境变量
    railway_url = os.getenv("DATABASE_URL")
    if railway_url:
        # 解析Railway URL: mysql://user:pass@host:port/database
        import re
        match = re.match(r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', railway_url)
        if match:
            user, password, host, port, database = match.groups()
            logger.info("🌐 使用Railway数据库配置")
            return {
                'host': host,
                'port': int(port),
                'user': user,
                'password': password,
                'database': database,
                'charset': 'utf8mb4'
            }
    
    # 备用本地配置
    logger.info("🏠 使用本地数据库配置")
    return {
        'host': os.getenv("DB_HOST", "localhost"),
        'port': int(os.getenv("DB_PORT", "3306")),
        'user': os.getenv("DB_USER", "root"),
        'password': os.getenv("DB_PASSWORD", "666666"),
        'database': os.getenv("DB_NAME", "haigui_database"),
        'charset': 'utf8mb4'
    }

# Excel文件路径 - database 4.xlsx 在当前目录
EXCEL_FILE_PATH = './database 4.xlsx'

# 基于Database 4.xlsx实际位置的列名映射（134个字段 - 包含新增的2个字段）
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
    'pH_2_additional',               # 位置71 ⭐ 新增字段1 (database 4中的pH.2)
    'Ingrediant_additional',         # 位置72 ⭐ 新增字段2 (database 4中的Ingrediant)  
    'number_field',                  # 位置73 (原71)
    'type_field',                    # 位置74 (原72)
    'SolutionorMoisture',            # 位置75 (原73)
    'cycle_pH',                      # 位置76 (原74)
    'cycle_pH_after',                # 位置77 (原75)
    'cycle_ingredient',              # 位置78 (原76)
    'temp',                          # 位置79 (原77)
    'temp2',                         # 位置80 (原78)
    'RH_2',                          # 位置81 (原79)
    'RH2',                           # 位置82 (原80)
    'OTHER1_2',                      # 位置83 (原81)
    'OTHER2_main',                   # 位置84 (原82)
    'time_in_cycle',                 # 位置85 (原83)
    'note_3',                        # 位置86 (原84)
    'UV',                            # 位置87 (原85)
    'note_4',                        # 位置88 (原86)
    'stress_or_strain',              # 位置89 (原87)
    'type_of_load',                  # 位置90 (原88)
    'value_load',                    # 位置91 (原89)
    'ultimate_tensile_strength',     # 位置92 (原90)
    'tensile_modulus',               # 位置93 (原91)
    'note_5',                        # 位置94 (原92)
    'after_condition',               # 位置95 (原93)
    'note_6',                        # 位置96 (原94)
    'num_2',                         # 位置97 (原95)
    'Value1_2',                      # 位置98 (原96)
    'COV1_2',                        # 位置99 (原97)
    'Value1note',                    # 位置100 (原98)
    'retention1',                    # 位置101 (原99)
    'Value2_2',                      # 位置102 (原100)
    'COV2_2',                        # 位置103 (原101)
    'Value2note_2',                  # 位置104 (原102)
    'retention2',                    # 位置105 (原103)
    'Value3_2',                      # 位置106 (原104)
    'COV3_2',                        # 位置107 (原105)
    'Value3note_2',                  # 位置108 (原106)
    'retention3',                    # 位置109 (原107)
    'num_3',                         # 位置110 (原108)
    'water_absorption_ratio',        # 位置111 (原109)
    'COV_1',                         # 位置112 (原110)
    'note_7',                        # 位置113 (原111)
    'num_4',                         # 位置114 (原112)
    'glass_transition_temperature_2', # 位置115 (原113)
    'run2',                          # 位置116 (原114)
    'COV_2',                         # 位置117 (原115)
    'cure_ratio_2',                  # 位置118 (原116)
    'note_8',                        # 位置119 (原117)
    'num_5',                         # 位置120 (原118)
    'OTHERS',                        # 位置121 (原119)
    'OTHERS_note',                   # 位置122 (原120)
    'SEM_T_BCAT',                    # 位置123 (原121)
    'SEM_L_BCAT',                    # 位置124 (原122)
    'SEM_T_ACBT',                    # 位置125 (原123)
    'SEM_L_ACBT',                    # 位置126 (原124)
    'SEM_T_ACAT',                    # 位置127 (原125)
    'SEM_L_ACAT',                    # 位置128 (原126)
    'other_lower',                   # 位置129 (原127)
    'other2_final',                  # 位置130 (原128)
    'note_9',                        # 位置131 (原129)
    'FTIR_2',                        # 位置132 (原130)
    'note_10',                       # 位置133 (原131)
    'important_note'                 # 位置134 (原132)
]

def test_connection():
    """测试数据库连接"""
    try:
        db_config = get_db_config()
        connection = mysql.connector.connect(**db_config)
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
        
        # 取前134列（适配database 4.xlsx的新结构）
        df = df.iloc[:, :134]
        logger.info(f"使用前134列（包含2个新字段），调整后形状: {df.shape}")
        
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

def create_table(cursor):
    """创建数据表"""
    # 创建简化表结构，只包含主要字段
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        feature_name VARCHAR(255),
        Title TEXT,
        Author VARCHAR(500),
        SCI VARCHAR(100),
        Journal_or_Conference_name VARCHAR(500),
        Year INT,
        No_field VARCHAR(100),
        no_field_secondary VARCHAR(100),
        Fiber_type VARCHAR(200),
        Fiber_type_detail VARCHAR(500),
        Matrix_type VARCHAR(200),
        Matrix_type_detail VARCHAR(500),
        glass_transition_temperature DECIMAL(10,3),
        glass_transition_temperature_run_2 DECIMAL(10,3),
        cure_ratio DECIMAL(10,3),
        Fiber_content_weight DECIMAL(10,3),
        Fiber_content_volume DECIMAL(10,3),
        Void_content DECIMAL(10,3),
        diameter DECIMAL(10,3),
        average_area DECIMAL(10,3),
        nominal_area DECIMAL(10,3),
        rib VARCHAR(100),
        surface_treatment VARCHAR(200),
        Water_absorption_at_saturation DECIMAL(10,3),
        Water_absorption_test_standard VARCHAR(200),
        Water_absorption_note TEXT,
        Brand_name VARCHAR(200),
        Manufacturer VARCHAR(200),
        Important_notes TEXT,
        Notes_of_rebar TEXT,
        Target_parameter VARCHAR(100),
        note_of_target_parameter TEXT,
        num_1 DECIMAL(10,3),
        note_of_number TEXT,
        Value1_1 DECIMAL(10,3),
        COV1_1 DECIMAL(10,3),
        note_of_Value1 TEXT,
        Value2_1 DECIMAL(10,3),
        COV2_1 DECIMAL(10,3),
        Value2note_1 TEXT,
        Value3_1 DECIMAL(10,3),
        COV3_1 DECIMAL(10,3),
        Value3note_1 TEXT,
        SEM_T_BCBT VARCHAR(100),
        SEM_L_BCBT VARCHAR(100),
        OTHER_main VARCHAR(200),
        OTHER1_1 VARCHAR(200),
        FTIR_1 VARCHAR(200),
        note_1 TEXT,
        temperature DECIMAL(10,3),
        note_of_temperature TEXT,
        time_field DECIMAL(10,3),
        note_of_time TEXT,
        concrete VARCHAR(200),
        pH_of_concrete DECIMAL(10,3),
        strength_of_concrete DECIMAL(10,3),
        crack VARCHAR(100),
        cover DECIMAL(10,3),
        note_of_concrete TEXT,
        pH_1 DECIMAL(10,3),
        pHafter DECIMAL(10,3),
        ingredient_1 VARCHAR(200),
        pH_2 DECIMAL(10,3),
        RH_1 DECIMAL(10,3),
        ingredient_2 VARCHAR(200),
        note_2 TEXT,
        Location VARCHAR(200),
        Effektive_Klimaklassifikation VARCHAR(200),
        field_average_humidity DECIMAL(10,3),
        field_average_temperature DECIMAL(10,3),
        pH_2_additional DECIMAL(10,3),
        Ingrediant_additional VARCHAR(200),
        number_field VARCHAR(100),
        type_field VARCHAR(100),
        SolutionorMoisture VARCHAR(200),
        cycle_pH DECIMAL(10,3),
        cycle_pH_after DECIMAL(10,3),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    cursor.execute(create_table_sql)
    logger.info("✅ 数据表结构创建/验证完成")

def insert_data(df):
    """插入数据到MySQL"""
    try:
        db_config = get_db_config()
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # 创建表（如果不存在）
        create_table(cursor)
        
        # —— 第一步：清空表 —— 
        cursor.execute("TRUNCATE TABLE data")
        connection.commit()
        logger.info("🧹 已清空表 data，准备插入新数据")
        
        # 使用简化的字段列表（前70个主要字段）
        main_columns = MYSQL_COLUMNS[:70]  # 只使用前70个主要字段
        columns_str = ', '.join([f"`{col}`" for col in main_columns])
        placeholders = ', '.join(['%s'] * len(main_columns))
        query = f"INSERT INTO data ({columns_str}) VALUES ({placeholders})"
        
        logger.info(f"📝 准备插入 {len(main_columns)} 个字段到 {len(df)} 行数据")
        
        # 准备数据
        data_rows = []
        for _, row in df.iterrows():
            row_data = []
            for i in range(len(main_columns)):
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
        db_config = get_db_config()
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM data")
        count = cursor.fetchone()[0]
        logger.info(f"📊 数据库中总记录数: {count}")
        
        # 显示样本数据
        cursor.execute("""
            SELECT Title, Author, Year, Fiber_type 
            FROM data 
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

def show_env_setup_guide():
    """显示环境设置指南"""
    logger.info("💡 数据库配置指南:")
    logger.info("=" * 50)
    logger.info("📁 请在项目根目录创建 .env 文件，内容如下:")
    logger.info("")
    logger.info("# 本地数据库配置")
    logger.info("DB_HOST=localhost")
    logger.info("DB_PORT=3306") 
    logger.info("DB_USER=root")
    logger.info("DB_PASSWORD=你的密码")
    logger.info("DB_NAME=haigui_database")
    logger.info("")
    logger.info("# Railway数据库配置（可选，优先使用）")
    logger.info("DATABASE_URL=mysql://user:pass@host:port/database")
    logger.info("")
    logger.info("=" * 50)

def main():
    """主函数"""
    logger.info("🎯 开始Excel到MySQL数据迁移 (Database 4.xlsx 兼容版)")
    logger.info("=" * 70)
    
    # 显示配置信息
    db_config = get_db_config()
    current_host = db_config.get('host', 'unknown')
    current_db = db_config.get('database', 'unknown')
    
    # 步骤0: 显示当前配置
    logger.info("步骤0: 当前数据库配置")
    logger.info(f"  🎯 目标数据库: {current_host}:{db_config.get('port', 3306)}")
    logger.info(f"  📊 数据库名: {current_db}")
    logger.info(f"  👤 用户: {db_config.get('user', 'unknown')}")
    
    # 步骤1: 检查文件
    logger.info("步骤1: 检查Excel文件")
    if not check_files():
        logger.info("📋 Excel文件检查失败，可能的原因:")
        logger.info("  - 确保 database 4.xlsx 在上级目录")
        logger.info("  - 检查文件路径是否正确")
        return
    
    # 步骤2: 测试连接
    logger.info("步骤2: 测试数据库连接")
    if not test_connection():
        logger.error("❌ 数据库连接失败，请检查:")
        logger.error("  - MySQL服务是否启动")
        logger.error("  - 用户名密码是否正确")
        logger.error("  - 网络连接是否正常（Railway）")
        show_env_setup_guide()
        return
    
    # 步骤3: 读取Excel
    logger.info("步骤3: 读取Excel数据 (Database 4.xlsx)")
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
        logger.info("🎉 Database 4.xlsx 数据迁移成功完成！")
        logger.info(f"📊 目标数据库: {current_host} -> {current_db}")
        logger.info("💡 数据已包含新增的2个字段 (pH.2, Ingrediant)")
        logger.info("🔍 您现在可以在MySQL Workbench或应用中查看数据")

if __name__ == "__main__":
    main()