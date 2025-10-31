"""
数据库迁移脚本：从本地XAMPP迁移到Railway MySQL
使用方法：
1. 先在Railway创建MySQL数据库
2. 获取Railway数据库连接信息
3. 修改下面的RAILWAY_DB_CONFIG
4. 运行此脚本进行数据迁移
"""

import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 本地XAMPP数据库配置
LOCAL_DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'haigui_database'
}

# Railway数据库配置（已填入实际信息）
RAILWAY_DB_CONFIG = {
    'host': 'hopper.proxy.rlwy.net',
    'port': 56566,
    'user': 'root',
    'password': 'KujjHdzcQMKuTMoHEOzjRoKIvTKWBfBJ',
    'database': 'railway'
}

def export_from_local():
    """从本地数据库导出数据"""
    print("🔄 正在从本地数据库导出数据...")
    
    try:
        # 连接本地数据库
        local_engine = create_engine(
            f"mysql+pymysql://{LOCAL_DB_CONFIG['user']}:{LOCAL_DB_CONFIG['password']}@{LOCAL_DB_CONFIG['host']}/{LOCAL_DB_CONFIG['database']}"
        )
        
        # 导出research_data表
        df = pd.read_sql("SELECT * FROM research_data", local_engine)
        print(f"✅ 成功导出 {len(df)} 条记录")
        
        # 保存到CSV文件
        df.to_csv('research_data_export.csv', index=False)
        print("✅ 数据已保存到 research_data_export.csv")
        
        return df
    
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return None

def import_to_railway(df):
    """导入数据到Railway数据库"""
    if df is None:
        print("❌ 没有数据需要导入")
        return False
        
    print("🔄 正在导入数据到Railway...")
    
    try:
        # 检查Railway配置
        if RAILWAY_DB_CONFIG['host'] == 'YOUR_RAILWAY_HOST':
            print("❌ 请先配置Railway数据库连接信息！")
            print("在RAILWAY_DB_CONFIG中填入实际的数据库信息")
            return False
        
        # 连接Railway数据库
        railway_engine = create_engine(
            f"mysql+pymysql://{RAILWAY_DB_CONFIG['user']}:{RAILWAY_DB_CONFIG['password']}@{RAILWAY_DB_CONFIG['host']}:{RAILWAY_DB_CONFIG['port']}/{RAILWAY_DB_CONFIG['database']}"
        )
        
        # 导入数据
        df.to_sql('research_data', railway_engine, if_exists='replace', index=False, method='multi')
        print(f"✅ 成功导入 {len(df)} 条记录到Railway数据库")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def main():
    print("🚂 开始数据库迁移到Railway")
    print("=" * 50)
    
    # 第一步：从本地导出
    df = export_from_local()
    
    if df is not None:
        print(f"\n数据概况:")
        print(f"- 总行数: {len(df)}")
        print(f"- 总列数: {len(df.columns)}")
        print(f"- 数据大小: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # 第二步：导入到Railway
        if import_to_railway(df):
            print("\n🎉 数据迁移完成！")
            print("现在可以更新应用配置使用Railway数据库")
        else:
            print("\n❌ 数据迁移失败")
    
    print("=" * 50)

if __name__ == "__main__":
    main()