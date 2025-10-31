"""
📊 你的MySQL数据完整报告生成器
为完全初学者提供详细的数据分析报告
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def connect_to_database():
    """连接到数据库"""
    try:
        config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '',
            'database': 'frp_database'
        }
        
        connection = mysql.connector.connect(**config)
        if connection.is_connected():
            return connection
        
    except Error as e:
        logger.error(f"❌ 数据库连接失败: {e}")
        return None

def generate_complete_report():
    """生成完整的数据报告"""
    
    print("=" * 80)
    print("📊 你的MySQL数据库完整报告")
    print("=" * 80)
    
    connection = connect_to_database()
    if not connection:
        print("❌ 无法连接到数据库，请检查XAMPP是否启动")
        return
    
    try:
        cursor = connection.cursor()
        
        # 1. 基本信息
        print("\n🏷️  数据库基本信息")
        print("-" * 50)
        
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        print(f"📌 MySQL版本: {version}")
        
        cursor.execute("SELECT DATABASE()")
        current_db = cursor.fetchone()[0]
        print(f"📌 数据库名称: {current_db}")
        
        cursor.execute("SELECT COUNT(*) FROM data")
        total_records = cursor.fetchone()[0]
        print(f"📌 总记录数: {total_records:,} 条")
        
        cursor.execute("SHOW COLUMNS FROM data")
        columns = cursor.fetchall()
        print(f"📌 字段数量: {len(columns)} 个")
        
        # 2. 表结构详情
        print("\n🏗️  数据表结构")
        print("-" * 50)
        print("字段编号 | 字段名称                | 数据类型      | 是否必填")
        print("-" * 70)
        
        for i, col in enumerate(columns, 1):
            field_name, field_type, null, key, default, extra = col
            nullable = "可为空" if null == "YES" else "必填"
            print(f"{i:8d} | {field_name:22} | {field_type:12} | {nullable}")
        
        # 3. 数据质量分析
        print("\n📈 数据质量分析")
        print("-" * 50)
        
        # 获取所有数据进行分析
        cursor.execute("SELECT * FROM data")
        all_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(all_data, columns=column_names)
        
        print("字段名称                | 非空记录数 | 完整度   | 数据样例")
        print("-" * 75)
        
        for col in column_names[1:11]:  # 前10个主要字段
            non_null_count = df[col].notna().sum()
            completeness = (non_null_count / len(df)) * 100
            
            # 获取非空样例
            sample_values = df[col].dropna().head(3).tolist()
            if sample_values:
                sample = str(sample_values[0])[:20] + "..." if len(str(sample_values[0])) > 20 else str(sample_values[0])
            else:
                sample = "无数据"
            
            print(f"{col:22} | {non_null_count:8,} | {completeness:6.1f}% | {sample}")
        
        # 4. 关键统计信息
        print("\n📊 关键数据统计")
        print("-" * 50)
        
        # 纤维类型统计
        print("🔸 纤维类型分布 (前8名):")
        cursor.execute("""
            SELECT fiber_type, COUNT(*) as count 
            FROM data 
            WHERE fiber_type IS NOT NULL AND fiber_type != ''
            GROUP BY fiber_type 
            ORDER BY count DESC 
            LIMIT 8
        """)
        fiber_stats = cursor.fetchall()
        for fiber_type, count in fiber_stats:
            percentage = (count / total_records) * 100
            print(f"   • {fiber_type:25} : {count:4,}条 ({percentage:4.1f}%)")
        
        # 作者统计
        print("\n🔸 主要研究作者 (前5名):")
        cursor.execute("""
            SELECT author, COUNT(*) as count 
            FROM data 
            WHERE author IS NOT NULL AND author != ''
            GROUP BY author 
            ORDER BY count DESC 
            LIMIT 5
        """)
        author_stats = cursor.fetchall()
        for author, count in author_stats:
            author_short = author[:30] + "..." if len(author) > 30 else author
            print(f"   • {author_short:33} : {count:4,}条记录")
        
        # 年份分布
        print("\n🔸 数据年份分布:")
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN year < 2000 THEN '2000年前'
                    WHEN year BETWEEN 2000 AND 2010 THEN '2000-2010年'
                    WHEN year BETWEEN 2011 AND 2020 THEN '2011-2020年'
                    WHEN year > 2020 THEN '2020年后'
                    ELSE '未知年份'
                END as period,
                COUNT(*) as count
            FROM data
            GROUP BY period
            ORDER BY period
        """)
        year_stats = cursor.fetchall()
        for period, count in year_stats:
            percentage = (count / total_records) * 100
            print(f"   • {period:15} : {count:4,}条 ({percentage:4.1f}%)")
        
        # 5. 数据存储信息
        print("\n💾 数据存储信息")
        print("-" * 50)
        
        # 计算表大小
        cursor.execute("""
            SELECT 
                table_name,
                ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size_MB'
            FROM information_schema.TABLES 
            WHERE table_schema = 'frp_database' AND table_name = 'data'
        """)
        size_info = cursor.fetchone()
        if size_info:
            table_name, size_mb = size_info
            print(f"📁 表大小: {size_mb} MB")
        
        print(f"📁 存储位置: C:\\xampp\\mysql\\data\\frp_database\\")
        print(f"📁 备份建议: 定期导出SQL文件")
        
        # 6. 如何使用你的数据
        print("\n🎯 如何使用你的数据")
        print("-" * 50)
        print("1. 📊 可视化查看:")
        print("   • 打开浏览器访问: http://localhost/phpmyadmin")
        print("   • 点击左侧 'frp_database' → 'data' 表")
        print("   • 点击 '浏览' 查看所有记录")
        
        print("\n2. 🔍 数据筛选:")
        print("   • 在phpMyAdmin中点击 '搜索' 标签")
        print("   • 可以按纤维类型、作者等条件筛选")
        
        print("\n3. 📤 数据导出:")
        print("   • 点击 '导出' 按钮")
        print("   • 选择Excel格式可用于分析")
        
        print("\n4. 🔄 数据更新:")
        print("   • 运行: python true_smart_converter.py \"新文件.xlsx\"")
        print("   • 自动替换现有数据")
        
        # 7. 推荐的下一步操作
        print("\n🚀 推荐的下一步操作")
        print("-" * 50)
        print("✅ 1. 打开phpMyAdmin熟悉界面")
        print("✅ 2. 浏览前100条记录了解数据结构")  
        print("✅ 3. 尝试按纤维类型筛选数据")
        print("✅ 4. 导出一份Excel备份")
        print("✅ 5. 学习基本的SQL查询语句")
        
        print("\n" + "=" * 80)
        print("🎉 报告生成完成！你现在对你的MySQL数据有了全面了解！")
        print("=" * 80)
        
    except Error as e:
        print(f"❌ 生成报告失败: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    generate_complete_report()