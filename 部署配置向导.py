#!/usr/bin/env python3
"""
FRP预测平台 - 新部署配置脚本
用于快速修改部署给他人使用时的必要配置
"""

import os
import re
import secrets
import string

def generate_secret_key(length=32):
    """生成随机密钥"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def update_env_file():
    """更新.env文件中的数据库配置"""
    print("🔐 配置数据库连接信息")
    print("=" * 50)
    
    db_host = input("数据库主机地址 (例: mysql.example.com): ")
    db_port = input("数据库端口 (默认: 3306): ") or "3306"
    db_name = input("数据库名称: ")
    db_user = input("数据库用户名: ")
    db_password = input("数据库密码: ")
    
    env_content = f"""# FRP预测平台数据库配置
# 新部署配置 - {db_host}
DB_HOST={db_host}
DB_PORT={db_port}
DB_USER={db_user}
DB_PASSWORD={db_password}
DB_NAME={db_name}

# 生成的安全密钥
SECRET_KEY={generate_secret_key()}
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print(f"✅ .env 文件已更新！")
    return {
        'DB_HOST': db_host,
        'DB_PORT': db_port, 
        'DB_NAME': db_name,
        'DB_USER': db_user,
        'DB_PASSWORD': db_password,
        'SECRET_KEY': generate_secret_key()
    }

def generate_streamlit_secrets(db_config):
    """生成Streamlit Cloud的Secrets配置"""
    secrets_content = f'''DB_HOST = "{db_config['DB_HOST']}"
DB_PORT = "{db_config['DB_PORT']}"
DB_NAME = "{db_config['DB_NAME']}"
DB_USER = "{db_config['DB_USER']}"
DB_PASSWORD = "{db_config['DB_PASSWORD']}"
SECRET_KEY = "{db_config['SECRET_KEY']}"'''
    
    with open('streamlit_cloud_secrets.toml', 'w', encoding='utf-8') as f:
        f.write(secrets_content)
    
    print("📋 Streamlit Cloud Secrets配置已生成到: streamlit_cloud_secrets.toml")
    print("请将此内容复制到Streamlit Cloud的Secrets配置中！")

def update_page_config():
    """更新页面配置信息"""
    print("\n🎨 配置页面信息")
    print("=" * 50)
    
    org_name = input("组织名称 (将显示在页面标题中): ")
    contact_email = input("技术支持邮箱: ")
    contact_name = input("联系人姓名: ")
    
    # 读取app.py文件
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新页面标题
    if org_name:
        new_title = f"{org_name} - FRP纤维增强聚合物耐久性预测平台"
        content = re.sub(
            r'page_title="[^"]*"',
            f'page_title="{new_title}"',
            content
        )
    
    # 保存修改
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 页面配置已更新！")
    return {
        'org_name': org_name,
        'contact_email': contact_email,
        'contact_name': contact_name
    }

def create_deployment_guide(db_config, page_config):
    """创建部署指南"""
    guide_content = f"""# 🚀 {page_config['org_name']} - FRP预测平台部署指南

## 📊 数据库信息
- 主机: {db_config['DB_HOST']}
- 端口: {db_config['DB_PORT']}
- 数据库: {db_config['DB_NAME']}
- 用户: {db_config['DB_USER']}

## 🔑 Streamlit Cloud配置
请将 `streamlit_cloud_secrets.toml` 文件中的内容复制到Streamlit Cloud的Secrets配置中。

## 👥 技术支持
- 联系人: {page_config['contact_name']}
- 邮箱: {page_config['contact_email']}

## 📋 部署步骤
1. 创建新的GitHub仓库
2. 上传所有文件到新仓库
3. 在Streamlit Cloud中连接新仓库
4. 配置Secrets（使用生成的streamlit_cloud_secrets.toml内容）
5. 部署应用

## ⚠️ 重要提醒
- 数据库需要先创建并导入数据
- 确保数据库用户有完整的读写权限
- 首次部署可能需要5-10分钟

## 🎯 访问地址
部署成功后，访问地址为: https://your-app-name.streamlit.app/

---
配置完成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('新部署指南.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📖 部署指南已生成: 新部署指南.md")

def main():
    """主函数"""
    print("🚀 FRP预测平台 - 新部署配置向导")
    print("=" * 60)
    print("此脚本将帮助您快速配置平台以部署给其他用户使用")
    print()
    
    # 检查当前目录
    if not os.path.exists('app.py'):
        print("❌ 错误：请在FRP预测平台项目根目录下运行此脚本！")
        return
    
    try:
        # 1. 配置数据库
        db_config = update_env_file()
        
        # 2. 生成Streamlit Secrets
        generate_streamlit_secrets(db_config)
        
        # 3. 配置页面信息
        page_config = update_page_config()
        
        # 4. 生成部署指南
        create_deployment_guide(db_config, page_config)
        
        print("\n🎉 配置完成！")
        print("=" * 50)
        print("📁 生成的文件：")
        print("  ├── .env (数据库配置)")
        print("  ├── streamlit_cloud_secrets.toml (Streamlit Cloud配置)")
        print("  ├── 新部署指南.md (部署说明)")
        print("  └── app.py (已更新页面配置)")
        print()
        print("🔄 下一步操作：")
        print("1. 检查生成的配置文件")
        print("2. 创建新的GitHub仓库") 
        print("3. 上传文件到新仓库")
        print("4. 在Streamlit Cloud中部署")
        print("5. 配置Secrets（使用生成的streamlit_cloud_secrets.toml内容）")
        
    except KeyboardInterrupt:
        print("\n\n❌ 配置已取消")
    except Exception as e:
        print(f"\n❌ 配置出错: {e}")

if __name__ == "__main__":
    main()