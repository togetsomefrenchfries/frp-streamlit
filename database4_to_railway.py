"""
Database 4.xlsx 到 Railway 数据库迁移脚本
专门用于将 database 4.xlsx 文件中的数据迁移到新的Railway数据库

使用方法：
1. 确保 database 4.xlsx 文件在当前目录
2. 准备好新的Railway数据库连接信息
3. 运行脚本: python database4_to_railway.py
4. 按提示输入Railway数据库信息
5. 等待数据迁移完成

特点：
- 自动处理Excel文件的复杂结构
- 支持132个字段的完整映射
- 分批导入，避免内存溢出
- 详细的进度显示和错误处理
- 自动生成配置文件供应用使用
"""

import pandas as pd
from sqlalchemy import create_engine, text
import mysql.connector
from mysql.connector import Error
import numpy as np
import logging
import os
import urllib.parse
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Database4ToRailwayMigrator:
    def __init__(self):
        # Excel文件路径 - 可修改为您的文件位置
        self.excel_file = 'database 4.xlsx'  # 默认当前目录
        # self.excel_file = r'C:\Users\YourName\Documents\database 4.xlsx'  # 绝对路径示例
        # self.excel_file = r'../数据文件/database 4.xlsx'  # 相对路径示例
        self.railway_config = None
        self.railway_engine = None
        
        # Database 4.xlsx 的132个字段映射（基于您现有的代码）
        self.mysql_columns = [
            'feature_name', 'Title', 'Author', 'SCI', 'Journal_or_Conference_name',
            'Year', 'No_field', 'no_field_secondary', 'Fiber_type', 'Fiber_type_detail',
            'Matrix_type', 'Matrix_type_detail', 'glass_transition_temperature', 
            'glass_transition_temperature_run_2', 'cure_ratio', 'Fiber_content_weight',
            'Fiber_content_volume', 'Void_content', 'diameter', 'average_area',
            'nominal_area', 'rib', 'surface_treatment', 'Water_absorption_at_saturation',
            'Water_absorption_test_standard', 'Water_absorption_note', 'Brand_name',
            'Manufacturer', 'Important_notes', 'Notes_of_rebar', 'Target_parameter',
            'note_of_target_parameter', 'num_1', 'note_of_number', 'Value1_1',
            'COV1_1', 'note_of_Value1', 'Value2_1', 'COV2_1', 'Value2note_1',
            'Value3_1', 'COV3_1', 'Value3note_1', 'SEM_T_BCBT', 'SEM_L_BCBT',
            'OTHER_main', 'OTHER1_1', 'FTIR_1', 'note_1', 'temperature',
            'note_of_temperature', 'time_field', 'note_of_time', 'concrete',
            'pH_of_concrete', 'strength_of_concrete', 'crack', 'cover',
            'note_of_concrete', 'pH_1', 'pHafter', 'ingredient_1', 'pH_2',
            'RH_1', 'ingredient_2', 'note_2', 'Location', 'Effektive_Klimaklassifikation',
            'field_average_humidity', 'field_average_temperature', 'number_field',
            'type_field', 'SolutionorMoisture', 'cycle_pH', 'cycle_pH_after',
            'cycle_ingredient', 'temp', 'temp2', 'RH_2', 'RH2', 'OTHER1_2',
            'OTHER2_main', 'time_in_cycle', 'note_3', 'UV', 'note_4',
            'stress_or_strain', 'type_of_load', 'value_load', 'ultimate_tensile_strength',
            'tensile_modulus', 'note_5', 'after_condition', 'note_6', 'num_2',
            'Value1_2', 'COV1_2', 'Value1note', 'retention1', 'Value2_2',
            'COV2_2', 'Value2note_2', 'retention2', 'Value3_2', 'COV3_2',
            'Value3note_2', 'retention3', 'num_3', 'water_absorption_ratio',
            'COV_1', 'note_7', 'num_4', 'glass_transition_temperature_2',
            'run2', 'COV_2', 'cure_ratio_2', 'note_8', 'num_5', 'OTHERS',
            'OTHERS_note', 'SEM_T_BCAT', 'SEM_L_BCAT', 'SEM_T_ACBT',
            'SEM_L_ACBT', 'SEM_T_ACAT', 'SEM_L_ACAT', 'other_lower',
            'other2_final', 'note_9', 'FTIR_2', 'note_10', 'important_note'
        ]
    
    def check_excel_file(self):
        """检查Excel文件是否存在，如果不存在则提示用户输入路径"""
        logger.info(f"📍 检查Excel文件: {self.excel_file}")
        
        if os.path.exists(self.excel_file):
            file_size = os.path.getsize(self.excel_file) / (1024*1024)  # MB
            logger.info(f"✅ 文件存在，大小: {file_size:.2f} MB")
            return True
        else:
            logger.error(f"❌ 找不到文件: {self.excel_file}")
            print("\n📁 请指定Excel文件的位置:")
            print("示例:")
            print("  - 当前目录: database 4.xlsx")
            print("  - 绝对路径: C:\\Users\\YourName\\Documents\\database 4.xlsx")
            print("  - 相对路径: ../数据文件/database 4.xlsx")
            
            new_path = input("\n请输入Excel文件完整路径: ").strip().strip('"')
            
            if os.path.exists(new_path):
                self.excel_file = new_path
                file_size = os.path.getsize(self.excel_file) / (1024*1024)
                logger.info(f"✅ 找到文件: {self.excel_file}")
                logger.info(f"文件大小: {file_size:.2f} MB")
                return True
            else:
                logger.error(f"❌ 指定的文件不存在: {new_path}")
                return False
    
    def get_railway_config(self):
        """获取Railway数据库配置"""
        print("🚂 配置Railway数据库连接")
        print("=" * 50)
        print("请输入您的Railway数据库连接信息:")
        print("(可以在Railway项目的Variables页面找到这些信息)")
        print()
        
        host = input("数据库主机 (例: containers-us-west-xxx.railway.app): ").strip()
        port = input("数据库端口 (默认: 3306): ").strip() or "3306"
        user = input("数据库用户名 (通常是 root): ").strip() or "root"
        password = input("数据库密码: ").strip()
        database = input("数据库名称 (通常是 railway): ").strip() or "railway"
        
        self.railway_config = {
            'host': host,
            'port': int(port),
            'user': user,
            'password': password,
            'database': database
        }
        
        return self.railway_config
    
    def test_railway_connection(self):
        """测试Railway数据库连接"""
        logger.info("🔌 测试Railway数据库连接...")
        
        try:
            # 使用mysql.connector测试连接
            connection = mysql.connector.connect(
                host=self.railway_config['host'],
                port=self.railway_config['port'],
                user=self.railway_config['user'],
                password=self.railway_config['password'],
                database=self.railway_config['database'],
                charset='utf8mb4'
            )
            
            if connection.is_connected():
                logger.info("✅ Railway数据库连接成功")
                connection.close()
                
                # 创建SQLAlchemy引擎
                encoded_password = urllib.parse.quote_plus(self.railway_config['password'])
                connection_string = (
                    f"mysql+pymysql://{self.railway_config['user']}:{encoded_password}@"
                    f"{self.railway_config['host']}:{self.railway_config['port']}/{self.railway_config['database']}"
                )
                
                self.railway_engine = create_engine(connection_string)
                return True
            
        except Error as e:
            logger.error(f"❌ Railway数据库连接失败: {e}")
            return False
    
    def create_research_data_table(self):
        """在Railway数据库中创建research_data表"""
        logger.info("🔨 创建research_data表...")
        
        # 生成创建表的SQL（基于132个字段）
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS research_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            {', '.join([f'`{col}` TEXT' for col in self.mysql_columns])},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            
            INDEX idx_title (Title(100)),
            INDEX idx_author (Author(100)),
            INDEX idx_year (Year(10)),
            INDEX idx_fiber_type (Fiber_type(50)),
            INDEX idx_matrix_type (Matrix_type(50))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            with self.railway_engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            logger.info("✅ research_data表创建成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 创建表失败: {e}")
            return False
    
    def read_excel_data(self):
        """读取database 4.xlsx数据"""
        logger.info("📖 读取Excel文件数据...")
        
        try:
            # 从第4行开始读取（跳过标题行）
            df = pd.read_excel(self.excel_file, header=3, engine='openpyxl')
            logger.info(f"✅ Excel读取成功，原始形状: {df.shape}")
            
            # 只取前132列，匹配字段映射
            df = df.iloc[:, :132]
            logger.info(f"调整后形状: {df.shape}")
            
            # 设置列名
            df.columns = self.mysql_columns
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Excel读取失败: {e}")
            return None
    
    def clean_data(self, df):
        """清理数据"""
        logger.info("🧹 清理数据...")
        
        df_clean = df.copy()
        
        # 处理特殊值
        special_values = ['SMD', 'Notreported', 'N/A', '', ' ', 'nan', 'NULL', 'None']
        df_clean = df_clean.replace(special_values, None)
        df_clean = df_clean.replace({np.nan: None})
        
        # 处理数值字段（Year等）
        numeric_columns = ['Year', 'diameter', 'Value1_1', 'COV1_1']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 处理百分比字段
        retention_columns = ['retention1', 'retention2', 'retention3']
        for col in retention_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace('%', '').str.replace('nan', '')
        
        # 限制文本长度，避免数据库错误
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].astype(str).str[:2000]
            df_clean[col] = df_clean[col].replace('None', None)
        
        logger.info("✅ 数据清理完成")
        logger.info(f"清理后数据形状: {df_clean.shape}")
        
        return df_clean
    
    def migrate_data_to_railway(self, df):
        """将数据迁移到Railway数据库"""
        logger.info(f"🚀 开始迁移数据到Railway ({len(df)} 条记录)")
        
        try:
            # 清空现有数据
            with self.railway_engine.connect() as conn:
                conn.execute(text("TRUNCATE TABLE research_data"))
                conn.commit()
            
            logger.info("🧹 已清空表，开始插入新数据")
            
            # 分批插入数据
            batch_size = 100  # Railway可能对批次大小有限制
            total_batches = (len(df) + batch_size - 1) // batch_size
            inserted_count = 0
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size].copy()
                batch_num = i // batch_size + 1
                
                logger.info(f"📊 插入第 {batch_num}/{total_batches} 批 ({len(batch_df)} 条记录)...")
                
                try:
                    batch_df.to_sql(
                        'research_data', 
                        self.railway_engine, 
                        if_exists='append', 
                        index=False, 
                        method='multi'
                    )
                    
                    inserted_count += len(batch_df)
                    progress = (inserted_count / len(df)) * 100
                    logger.info(f"✅ 第 {batch_num} 批插入成功，总进度: {progress:.1f}%")
                    
                except Exception as batch_error:
                    logger.warning(f"⚠️ 批次插入失败，尝试逐行插入: {batch_error}")
                    
                    # 逐行插入
                    for _, row in batch_df.iterrows():
                        try:
                            row_df = pd.DataFrame([row])
                            row_df.to_sql(
                                'research_data', 
                                self.railway_engine, 
                                if_exists='append', 
                                index=False
                            )
                            inserted_count += 1
                        except Exception as row_error:
                            logger.error(f"❌ 单行插入失败: {row_error}")
            
            logger.info(f"🎉 数据迁移完成！成功插入 {inserted_count} 条记录")
            return inserted_count > 0
            
        except Exception as e:
            logger.error(f"❌ 数据迁移失败: {e}")
            return False
    
    def verify_migration(self):
        """验证数据迁移结果"""
        logger.info("🔍 验证迁移结果...")
        
        try:
            with self.railway_engine.connect() as conn:
                # 检查总记录数
                result = conn.execute(text("SELECT COUNT(*) as count FROM research_data"))
                count = result.fetchone()[0]
                logger.info(f"📊 Railway数据库中共有 {count} 条记录")
                
                # 获取样本数据
                sample_result = conn.execute(text("""
                    SELECT Title, Author, Year, Fiber_type, Matrix_type 
                    FROM research_data 
                    WHERE Title IS NOT NULL 
                    LIMIT 3
                """))
                
                samples = sample_result.fetchall()
                logger.info("📋 样本数据:")
                for i, (title, author, year, fiber, matrix) in enumerate(samples, 1):
                    title_short = title[:50] + "..." if title and len(title) > 50 else title
                    logger.info(f"  {i}. {title_short}")
                    logger.info(f"     作者: {author} | 年份: {year}")
                    logger.info(f"     纤维: {fiber} | 基体: {matrix}")
                
                # 统计信息
                stats_result = conn.execute(text("""
                    SELECT 
                        COUNT(DISTINCT Fiber_type) as fiber_types,
                        COUNT(DISTINCT Matrix_type) as matrix_types,
                        COUNT(DISTINCT Author) as authors,
                        MIN(Year) as min_year,
                        MAX(Year) as max_year
                    FROM research_data 
                    WHERE Fiber_type IS NOT NULL OR Matrix_type IS NOT NULL
                """))
                
                stats = stats_result.fetchone()
                if stats:
                    logger.info(f"📈 数据统计:")
                    logger.info(f"  - 纤维类型: {stats[0]} 种")
                    logger.info(f"  - 基体类型: {stats[1]} 种")
                    logger.info(f"  - 作者数量: {stats[2]} 人")
                    logger.info(f"  - 年份范围: {stats[3]}-{stats[4]}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ 验证失败: {e}")
            return False
    
    def generate_config_files(self):
        """生成配置文件"""
        logger.info("📝 生成应用配置文件...")
        
        # 生成.env文件
        env_content = f"""# FRP预测平台数据库配置
# Database 4.xlsx 迁移到Railway完成
# 迁移时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DB_HOST={self.railway_config['host']}
DB_PORT={self.railway_config['port']}
DB_USER={self.railway_config['user']}
DB_PASSWORD={self.railway_config['password']}
DB_NAME={self.railway_config['database']}

# 应用配置
SECRET_KEY=frp-railway-{datetime.now().strftime('%Y%m%d')}-secret
"""
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        # 生成Streamlit Cloud Secrets配置
        secrets_content = f'''DB_HOST = "{self.railway_config['host']}"
DB_PORT = "{self.railway_config['port']}"
DB_NAME = "{self.railway_config['database']}"
DB_USER = "{self.railway_config['user']}"
DB_PASSWORD = "{self.railway_config['password']}"
SECRET_KEY = "frp-railway-{datetime.now().strftime('%Y%m%d')}-secret"'''
        
        with open('railway_streamlit_secrets.toml', 'w', encoding='utf-8') as f:
            f.write(secrets_content)
        
        logger.info("✅ .env 文件已生成")
        logger.info("✅ railway_streamlit_secrets.toml 文件已生成")
        logger.info("📋 请将 railway_streamlit_secrets.toml 的内容复制到Streamlit Cloud的Secrets配置中")
    
    def run_migration(self):
        """运行完整的迁移流程"""
        print("🚂 Database 4.xlsx 到 Railway 数据迁移工具")
        print("=" * 70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 步骤1: 检查Excel文件
        if not self.check_excel_file():
            return False
        
        # 步骤2: 获取Railway配置
        self.get_railway_config()
        
        # 步骤3: 测试Railway连接
        if not self.test_railway_connection():
            logger.error("❌ 请检查Railway数据库连接信息")
            return False
        
        # 步骤4: 创建数据表
        if not self.create_research_data_table():
            return False
        
        # 步骤5: 读取Excel数据
        df = self.read_excel_data()
        if df is None:
            return False
        
        # 步骤6: 清理数据
        df_clean = self.clean_data(df)
        
        # 步骤7: 迁移数据
        if not self.migrate_data_to_railway(df_clean):
            return False
        
        # 步骤8: 验证结果
        if not self.verify_migration():
            logger.warning("⚠️ 验证失败，请手动检查数据")
        
        # 步骤9: 生成配置文件
        self.generate_config_files()
        
        print("\n" + "=" * 70)
        print("🎉 Database 4.xlsx 迁移到Railway成功完成！")
        print()
        print("📁 生成的文件:")
        print("  ├── .env (本地开发配置)")
        print("  └── railway_streamlit_secrets.toml (Streamlit Cloud配置)")
        print()
        print("🔄 下一步:")
        print("1. 使用 .env 文件进行本地测试")
        print("2. 将 railway_streamlit_secrets.toml 内容复制到Streamlit Cloud")
        print("3. 重新部署您的应用")
        print()
        
        return True

def main():
    """主函数"""
    migrator = Database4ToRailwayMigrator()
    
    try:
        success = migrator.run_migration()
        if success:
            print("✨ 迁移成功！您的Database 4.xlsx数据现在已在Railway云数据库中")
        else:
            print("❌ 迁移失败，请检查错误信息并重试")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户取消了迁移过程")
    except Exception as e:
        logger.error(f"❌ 发生未预期的错误: {e}")

if __name__ == "__main__":
    main()