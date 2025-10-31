#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署前检查脚本 - FRP预测平台
检查部署所需的文件和配置是否完整
"""

import os
import sys
import json
from pathlib import Path

def check_deployment_readiness():
    """检查部署准备情况"""
    
    print("🔍 FRP预测平台部署检查")
    print("=" * 50)
    
    checks = []
    
    # 1. 检查必要文件
    print("📁 检查必要文件...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        '.streamlit/config.toml',
        'README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            checks.append(("✅", f"{file} 存在"))
            
            # 检查文件大小
            size = os.path.getsize(file)
            if size > 0:
                checks.append(("✅", f"{file} 大小正常 ({size} bytes)"))
            else:
                checks.append(("❌", f"{file} 文件为空"))
        else:
            checks.append(("❌", f"{file} 缺失"))
    
    # 2. 检查requirements.txt内容
    print("\n📦 检查依赖包...")
    
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = f.read()
            
        required_packages = [
            'streamlit',
            'pandas',
            'numpy',
            'scikit-learn',
            'sqlalchemy',
            'pymysql'
        ]
        
        for package in required_packages:
            if package in requirements:
                checks.append(("✅", f"依赖包 {package} 已包含"))
            else:
                checks.append(("❌", f"缺少依赖包 {package}"))
    
    # 3. 检查环境变量配置
    print("\n🔧 检查环境变量...")
    
    # 检查app.py中是否有环境变量配置
    if os.path.exists('app.py'):
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
            
        if 'os.environ.get' in app_content:
            checks.append(("✅", "app.py 使用环境变量"))
        else:
            checks.append(("⚠️", "app.py 可能需要环境变量配置"))
            
        if '.env' not in app_content or 'load_dotenv' in app_content:
            checks.append(("✅", "支持生产环境配置"))
        else:
            checks.append(("⚠️", "可能依赖.env文件"))
    
    # 4. 检查数据库配置
    print("\n🗄️ 检查数据库配置...")
    
    if os.path.exists('.env'):
        checks.append(("✅", ".env 文件存在（本地开发）"))
        
        with open('.env', 'r', encoding='utf-8') as f:
            env_content = f.read()
            
        if 'railway' in env_content.lower():
            checks.append(("✅", "配置了Railway数据库"))
        elif 'mysql' in env_content.lower():
            checks.append(("✅", "配置了MySQL数据库"))
    else:
        checks.append(("⚠️", "没有.env文件，确保使用环境变量"))
    
    # 5. 检查Git状态
    print("\n📋 检查Git状态...")
    
    if os.path.exists('.git'):
        checks.append(("✅", "Git仓库初始化"))
        
        # 检查是否有未提交的更改
        import subprocess
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                if result.stdout.strip():
                    checks.append(("⚠️", "有未提交的更改"))
                else:
                    checks.append(("✅", "所有更改已提交"))
            else:
                checks.append(("⚠️", "无法检查Git状态"))
        except:
            checks.append(("⚠️", "Git命令不可用"))
    else:
        checks.append(("❌", "不是Git仓库"))
    
    # 6. 检查文件大小
    print("\n📏 检查文件大小...")
    
    large_files = []
    for root, dirs, files in os.walk('.'):
        # 跳过.git和__pycache__目录
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.streamlit']]
        
        for file in files:
            if file.endswith(('.py', '.pkl', '.csv', '.xlsx')):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                
                if size > 50 * 1024 * 1024:  # 50MB
                    large_files.append((filepath, size))
    
    if large_files:
        checks.append(("⚠️", f"发现大文件 ({len(large_files)} 个)"))
        for filepath, size in large_files:
            checks.append(("   ", f"  {filepath}: {size/1024/1024:.1f}MB"))
    else:
        checks.append(("✅", "文件大小适合部署"))
    
    # 输出检查结果
    print("\n" + "=" * 50)
    print("🎯 检查结果汇总")
    print("=" * 50)
    
    success_count = 0
    warning_count = 0
    error_count = 0
    
    for status, message in checks:
        print(f"{status} {message}")
        if status == "✅":
            success_count += 1
        elif status == "⚠️":
            warning_count += 1
        elif status == "❌":
            error_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 统计: ✅ {success_count} | ⚠️ {warning_count} | ❌ {error_count}")
    
    # 给出部署建议
    if error_count == 0:
        if warning_count == 0:
            print("\n🎉 完美！您的应用已准备好部署！")
            print("🚀 建议使用 Streamlit Cloud 进行部署")
        else:
            print("\n✨ 基本准备就绪！有一些警告需要注意")
            print("🚀 可以尝试部署，但建议先解决警告项")
    else:
        print("\n🔧 需要修复一些问题才能部署")
        print("❌ 请解决所有错误项后再次运行检查")
    
    return error_count == 0

def generate_deployment_guide():
    """生成部署指南"""
    
    guide = """
🚀 快速部署步骤：

1. Streamlit Cloud 部署（推荐）：
   a) 访问 https://share.streamlit.io/
   b) 用GitHub登录
   c) 选择仓库: pengjie123123/frp-streamlit
   d) 设置环境变量（数据库配置）
   e) 点击Deploy

2. 需要设置的环境变量：
   - DB_HOST: 您的Railway数据库地址
   - DB_PORT: 数据库端口
   - DB_NAME: railway
   - DB_USER: root
   - DB_PASSWORD: 您的数据库密码

3. 部署后测试：
   - 访问生成的URL
   - 测试用户注册登录
   - 测试模型训练和预测功能

🌍 部署成功后，全世界都可以访问您的FRP预测平台！
"""
    
    print(guide)

if __name__ == "__main__":
    print("🚀 FRP预测平台部署检查工具")
    print("Version 1.0 | 2025-10-31\n")
    
    if check_deployment_readiness():
        generate_deployment_guide()
    
    print("\n💡 如需帮助，请查看 '部署指南.md' 文件")
    print("📋 详细步骤请参考部署文档")