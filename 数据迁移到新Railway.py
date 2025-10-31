"""
FRP预测平台 - 数据迁移脚本（用于新部署）
将现有FRP数据迁移到新的Railway数据库

使用方法：
1. 在Railway创建新的MySQL数据库
2. 获取数据库连接信息
3. 运行脚本时输入连接信息
4. 选择数据源（CSV文件或原数据库）
5. 自动完成数据迁移

支持的数据源：
- 从CSV文件导入（推荐）
- 从原始数据库导入（需要连接信息）
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os
import urllib.parse
from datetime import datetime

class FRPDataMigrator:
    def __init__(self):
        self.target_db = None
        self.data_df = None
        
    def get_target_database_config(self):
        """获取目标Railway数据库配置"""
        print("🔐 配置目标Railway数据库")
        print("=" * 50)
        
        host = input("Railway数据库主机 (例: containers-us-west-xxx.railway.app): ")
        port = input("数据库端口 (默认: 3306): ") or "3306"
        user = input("数据库用户名 (通常是 root): ") or "root"
        password = input("数据库密码: ")
        database = input("数据库名称 (通常是 railway): ") or "railway"
        
        return {
            'host': host,
            'port': int(port),
            'user': user,
            'password': password,
            'database': database
        }
    
    def create_database_connection(self, config):
        """创建数据库连接"""
        try:
            # URL编码密码以处理特殊字符
            encoded_password = urllib.parse.quote_plus(config['password'])
            
            connection_string = (
                f"mysql+pymysql://{config['user']}:{encoded_password}@"
                f"{config['host']}:{config['port']}/{config['database']}"
            )
            
            engine = create_engine(connection_string)
            
            # 测试连接
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print(f"✅ 成功连接到数据库: {config['host']}")
            return engine
            
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return None
    
    def create_frp_table(self, engine):
        """在目标数据库中创建FRP数据表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS research_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Fiber_Type VARCHAR(100),
            Matrix_Type VARCHAR(100),
            Environmental_Condition VARCHAR(200),
            Temperature_C DECIMAL(8,2),
            Humidity_percent DECIMAL(5,2),
            pH_Value DECIMAL(4,2),
            Solution_Type VARCHAR(200),
            Concentration_mol_L DECIMAL(10,6),
            Duration_days INT,
            Duration_hours DECIMAL(10,2),
            Tensile_Strength_Retention_percent DECIMAL(5,2),
            Mass_Change_percent DECIMAL(8,4),
            Diameter_Change_percent DECIMAL(8,4),
            Appearance_Change TEXT,
            Test_Method VARCHAR(200),
            Specimen_Preparation VARCHAR(500),
            Reference VARCHAR(1000),
            DOI VARCHAR(200),
            Additional_Notes TEXT,
            Data_Source VARCHAR(200),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            
            INDEX idx_fiber_type (Fiber_Type),
            INDEX idx_temperature (Temperature_C),
            INDEX idx_duration (Duration_days),
            INDEX idx_retention (Tensile_Strength_Retention_percent)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            with engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            print("✅ 数据表创建成功")
            return True
        except Exception as e:
            print(f"❌ 数据表创建失败: {e}")
            return False
    
    def load_data_from_csv(self):
        """从CSV文件加载数据"""
        print("\n📁 从CSV文件加载数据")
        print("=" * 30)
        
        # 查找可能的CSV文件
        csv_files = []
        for file in os.listdir('.'):
            if file.endswith('.csv') and any(keyword in file.lower() for keyword in ['research', 'data', 'frp', 'export']):
                csv_files.append(file)
        
        if csv_files:
            print("找到以下CSV文件:")
            for i, file in enumerate(csv_files):
                print(f"{i+1}. {file}")
            
            try:
                choice = int(input(f"\n请选择文件 (1-{len(csv_files)}): ")) - 1
                selected_file = csv_files[choice]
            except (ValueError, IndexError):
                selected_file = csv_files[0]
                print(f"使用默认文件: {selected_file}")
        else:
            selected_file = input("请输入CSV文件路径: ")
        
        try:
            df = pd.read_csv(selected_file)
            print(f"✅ 成功加载 {len(df)} 条记录")
            print(f"数据列: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ CSV文件加载失败: {e}")
            return None
    
    def load_data_from_database(self):
        """从原数据库加载数据"""
        print("\n🗄️ 从原数据库加载数据")
        print("=" * 30)
        
        source_config = {
            'host': input("原数据库主机 (例: hopper.proxy.rlwy.net): "),
            'port': int(input("原数据库端口 (默认: 3306): ") or "3306"),
            'user': input("原数据库用户名: "),
            'password': input("原数据库密码: "),
            'database': input("原数据库名称: ")
        }
        
        source_engine = self.create_database_connection(source_config)
        if not source_engine:
            return None
        
        try:
            # 获取数据
            df = pd.read_sql("SELECT * FROM research_data", source_engine)
            print(f"✅ 成功从原数据库加载 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 从原数据库加载失败: {e}")
            return None
    
    def migrate_data(self, df, target_engine):
        """迁移数据到目标数据库"""
        print(f"\n🚀 开始数据迁移 ({len(df)} 条记录)")
        print("=" * 40)
        
        try:
            # 清理数据
            df_clean = df.copy()
            
            # 处理可能的数据类型问题
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str)
                    # 限制文本长度
                    if col in ['Reference', 'Specimen_Preparation']:
                        df_clean[col] = df_clean[col].str[:500]
                    elif col in ['Additional_Notes', 'Appearance_Change']:
                        df_clean[col] = df_clean[col].str[:1000]
            
            # 分批导入数据（每次1000条）
            batch_size = 1000
            total_batches = (len(df_clean) + batch_size - 1) // batch_size
            
            for i in range(0, len(df_clean), batch_size):
                batch_df = df_clean.iloc[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                print(f"正在导入第 {batch_num}/{total_batches} 批 ({len(batch_df)} 条记录)...")
                
                batch_df.to_sql(
                    'research_data', 
                    target_engine, 
                    if_exists='append', 
                    index=False, 
                    method='multi'
                )
                
                print(f"✅ 第 {batch_num} 批导入完成")
            
            print(f"\n🎉 数据迁移完成！总共迁移 {len(df_clean)} 条记录")
            return True
            
        except Exception as e:
            print(f"❌ 数据迁移失败: {e}")
            return False
    
    def verify_migration(self, target_engine):
        """验证数据迁移结果"""
        print("\n🔍 验证数据迁移结果")
        print("=" * 30)
        
        try:
            with target_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as count FROM research_data"))
                count = result.fetchone()[0]
                print(f"✅ 目标数据库中有 {count} 条记录")
                
                # 获取一些统计信息
                stats_result = conn.execute(text("""
                    SELECT 
                        COUNT(DISTINCT Fiber_Type) as fiber_types,
                        COUNT(DISTINCT Matrix_Type) as matrix_types,
                        AVG(Temperature_C) as avg_temp,
                        MIN(Duration_days) as min_duration,
                        MAX(Duration_days) as max_duration
                    FROM research_data 
                    WHERE Fiber_Type IS NOT NULL
                """))
                
                stats = stats_result.fetchone()
                if stats:
                    print(f"📊 数据统计:")
                    print(f"  - 纤维类型: {stats[0]} 种")
                    print(f"  - 基体类型: {stats[1]} 种") 
                    print(f"  - 平均温度: {stats[2]:.1f}°C" if stats[2] else "  - 平均温度: N/A")
                    print(f"  - 持续时间: {stats[3]}-{stats[4]} 天" if stats[3] and stats[4] else "  - 持续时间: N/A")
                
                return True
                
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False
    
    def run_migration(self):
        """运行完整的数据迁移流程"""
        print("🚂 FRP预测平台数据迁移工具")
        print("=" * 60)
        print(f"迁移时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. 获取目标数据库配置
        target_config = self.get_target_database_config()
        target_engine = self.create_database_connection(target_config)
        
        if not target_engine:
            print("❌ 无法连接目标数据库，迁移终止")
            return False
        
        # 2. 创建数据表
        if not self.create_frp_table(target_engine):
            print("❌ 数据表创建失败，迁移终止")
            return False
        
        # 3. 选择数据源
        print("\n📥 选择数据源")
        print("1. 从CSV文件导入（推荐）")
        print("2. 从原数据库导入")
        
        choice = input("请选择 (1 或 2): ").strip()
        
        if choice == "2":
            df = self.load_data_from_database()
        else:
            df = self.load_data_from_csv()
        
        if df is None:
            print("❌ 数据加载失败，迁移终止")
            return False
        
        # 4. 执行数据迁移
        if not self.migrate_data(df, target_engine):
            print("❌ 数据迁移失败")
            return False
        
        # 5. 验证迁移结果
        if not self.verify_migration(target_engine):
            print("⚠️ 数据验证失败，请检查数据完整性")
        
        # 6. 生成配置文件
        self.generate_config_files(target_config)
        
        print("\n🎉 FRP数据迁移完成！")
        print("下一步: 使用生成的配置文件更新您的应用设置")
        return True
    
    def generate_config_files(self, db_config):
        """生成配置文件"""
        print("\n📝 生成配置文件")
        print("=" * 20)
        
        # 生成.env文件
        env_content = f"""# FRP预测平台数据库配置
# 迁移完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DB_HOST={db_config['host']}
DB_PORT={db_config['port']}
DB_USER={db_config['user']}
DB_PASSWORD={db_config['password']}
DB_NAME={db_config['database']}

# 应用配置
SECRET_KEY=your-secret-key-here-{datetime.now().strftime('%Y%m%d')}
"""
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        # 生成Streamlit Secrets配置
        secrets_content = f'''DB_HOST = "{db_config['host']}"
DB_PORT = "{db_config['port']}"
DB_NAME = "{db_config['database']}"
DB_USER = "{db_config['user']}"
DB_PASSWORD = "{db_config['password']}"
SECRET_KEY = "your-secret-key-here-{datetime.now().strftime('%Y%m%d')}"'''
        
        with open('streamlit_secrets.toml', 'w', encoding='utf-8') as f:
            f.write(secrets_content)
        
        print("✅ .env 文件已生成")
        print("✅ streamlit_secrets.toml 文件已生成")
        print("\n🔑 请将 streamlit_secrets.toml 的内容复制到Streamlit Cloud的Secrets配置中")

def main():
    """主函数"""
    migrator = FRPDataMigrator()
    
    try:
        success = migrator.run_migration()
        if success:
            print("\n✨ 迁移成功完成！")
        else:
            print("\n❌ 迁移失败")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户取消了迁移过程")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()