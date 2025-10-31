"""
🛠️ 数据库设置工具
用于创建和配置FRP数据库
"""

import mysql.connector
from mysql.connector import Error
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def connect_to_mysql():
    """连接到MySQL服务器（不指定数据库）"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            port=3306,
            user='root',
            password=''  # XAMPP默认无密码
        )
        return connection
    except Error as e:
        logger.error(f"❌ 连接MySQL失败: {e}")
        return None

def setup_frp_database():
    """设置FRP数据库"""
    logger.info("🚀 开始设置FRP数据库...")
    
    connection = connect_to_mysql()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        # 1. 显示现有数据库
        logger.info("📋 查看现有数据库:")
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        for db in databases:
            logger.info(f"  📂 {db[0]}")
        
        # 2. 创建frp_database（如果不存在）
        logger.info("\n🏗️ 创建frp_database数据库...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS frp_database CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        logger.info("✅ 数据库创建成功!")
        
        # 3. 使用frp_database
        cursor.execute("USE frp_database")
        
        # 4. 创建data表
        logger.info("📊 创建data表...")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS `data` (
            `id` int(11) NOT NULL AUTO_INCREMENT,
            `specimen` varchar(255) DEFAULT NULL,
            `author` varchar(255) DEFAULT NULL,
            `year` int(11) DEFAULT NULL,
            `test_condition` varchar(255) DEFAULT NULL,
            `temperature` float DEFAULT NULL,
            `moisture` varchar(255) DEFAULT NULL,
            `solution` varchar(255) DEFAULT NULL,
            `pH` float DEFAULT NULL,
            `duration` float DEFAULT NULL,
            `fiber_type` varchar(255) DEFAULT NULL,
            `matrix_type` varchar(255) DEFAULT NULL,
            `test_method` varchar(255) DEFAULT NULL,
            `geometry` varchar(255) DEFAULT NULL,
            `diameter` float DEFAULT NULL,
            `cross_sectional_area` float DEFAULT NULL,
            `length` float DEFAULT NULL,
            `fiber_volume_fraction` float DEFAULT NULL,
            `elastic_modulus_initial` float DEFAULT NULL,
            `tensile_strength_initial` float DEFAULT NULL,
            `elastic_modulus_final` float DEFAULT NULL,
            `tensile_strength_final` float DEFAULT NULL,
            `modulus_retention` float DEFAULT NULL,
            `strength_retention` float DEFAULT NULL,
            `notes` text DEFAULT NULL,
            PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        cursor.execute(create_table_sql)
        logger.info("✅ data表创建成功!")
        
        # 5. 验证表结构
        cursor.execute("DESCRIBE data")
        columns = cursor.fetchall()
        logger.info(f"\n📋 data表结构 ({len(columns)}个字段):")
        for col in columns:
            logger.info(f"  📄 {col[0]}: {col[1]}")
        
        # 6. 检查表中数据
        cursor.execute("SELECT COUNT(*) FROM data")
        count = cursor.fetchone()[0]
        logger.info(f"\n📊 data表当前记录数: {count}")
        
        cursor.close()
        connection.close()
        
        logger.info("\n🎉 FRP数据库设置完成!")
        return True
        
    except Error as e:
        logger.error(f"❌ 数据库设置失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("🛠️ FRP数据库设置工具")
    logger.info("=" * 50)
    
    success = setup_frp_database()
    
    if success:
        logger.info("\n✅ 数据库设置成功! 现在可以运行转换器了")
        logger.info("🚀 试试运行: python true_smart_converter.py \"database 4.xlsx\"")
    else:
        logger.info("\n❌ 数据库设置失败! 请检查MySQL服务")
    
    return success

if __name__ == "__main__":
    main()