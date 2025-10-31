"""
FRP数据导出脚本
将当前Railway数据库中的数据导出为CSV文件，供新用户导入使用

使用方法：
1. 确保.env文件配置正确
2. 运行此脚本导出数据
3. 将生成的CSV文件提供给新用户
"""

import pandas as pd
from sqlalchemy import create_engine
import os
from datetime import datetime
from dotenv import load_dotenv

def export_frp_data():
    """导出FRP数据到CSV文件"""
    
    # 加载环境变量
    load_dotenv()
    
    print("📤 FRP数据导出工具")
    print("=" * 40)
    print(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 从环境变量获取数据库配置
    db_config = {
        'host': os.getenv('DB_HOST', 'hopper.proxy.rlwy.net'),
        'port': os.getenv('DB_PORT', '56566'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME', 'railway')
    }
    
    if not db_config['password']:
        print("❌ 数据库密码未配置，请检查.env文件")
        return False
    
    try:
        # 创建数据库连接
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string)
        
        print(f"🔗 连接数据库: {db_config['host']}")
        
        # 查询并导出数据
        query = "SELECT * FROM research_data ORDER BY id"
        df = pd.read_sql(query, engine)
        
        print(f"✅ 成功获取 {len(df)} 条记录")
        print(f"📊 数据列数: {len(df.columns)}")
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'frp_data_export_{timestamp}.csv'
        
        # 导出为CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"💾 数据已导出到: {filename}")
        
        # 显示数据统计
        print("\n📈 数据统计:")
        if 'Fiber_Type' in df.columns:
            print(f"  - 纤维类型数量: {df['Fiber_Type'].nunique()}")
        if 'Matrix_Type' in df.columns:
            print(f"  - 基体类型数量: {df['Matrix_Type'].nunique()}")
        if 'Temperature_C' in df.columns and df['Temperature_C'].notna().any():
            print(f"  - 温度范围: {df['Temperature_C'].min():.1f}°C - {df['Temperature_C'].max():.1f}°C")
        if 'Duration_days' in df.columns and df['Duration_days'].notna().any():
            print(f"  - 持续时间范围: {df['Duration_days'].min()} - {df['Duration_days'].max()} 天")
        
        file_size = os.path.getsize(filename) / 1024 / 1024
        print(f"  - 文件大小: {file_size:.2f} MB")
        
        # 生成数据说明文件
        columns_list = '\n'.join([f'- {col}' for col in df.columns[:10]])
        more_cols = '...(更多列)' if len(df.columns) > 10 else ''
        
        readme_content = f"""# FRP预测平台数据包

## 📋 数据信息
- 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据文件: {filename}
- 记录数量: {len(df)} 条
- 数据列数: {len(df.columns)} 列
- 文件大小: {file_size:.2f} MB

## 🗂️ 数据结构
以下是主要数据列：

{columns_list}
{more_cols}

## 🚀 如何使用此数据
1. 将CSV文件放在新的FRP预测平台项目目录中
2. 运行数据迁移脚本：`python 数据迁移到新Railway.py`
3. 选择"从CSV文件导入"选项
4. 选择此CSV文件进行导入

## 📊 数据来源
此数据包含FRP纤维增强聚合物在各种环境条件下的耐久性测试数据，
包括不同纤维类型、基体材料、温度、湿度、pH值等条件下的性能测试结果。

## ⚠️ 注意事项
- 请确保目标数据库有足够的存储空间
- 导入过程可能需要几分钟时间
- 建议先在测试环境中验证数据完整性
"""
        
        readme_filename = f'数据说明_{timestamp}.md'
        with open(readme_filename, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"📄 数据说明已生成: {readme_filename}")
        
        print(f"\n🎉 导出完成！")
        print(f"请将以下文件提供给新用户:")
        print(f"  ✅ {filename}")
        print(f"  ✅ {readme_filename}")
        print(f"  ✅ 数据迁移到新Railway.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False

if __name__ == "__main__":
    export_frp_data()