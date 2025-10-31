#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复SQLAlchemy 2.x中的conn.commit()问题
将所有使用conn.commit()的地方改为使用engine.begin()
"""

import re
import os

def fix_commit_issues(file_path):
    """修复文件中的commit问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_path = file_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 修复模式1: with engine.connect() as conn: ... conn.commit()
    pattern1 = r'with\s+engine\.connect\(\)\s+as\s+conn:\s*\n(.*?)conn\.commit\(\)'
    
    def replace_pattern1(match):
        body = match.group(1)
        # 移除不必要的缩进调整，保持原有缩进
        return f'with engine.begin() as conn:\n{body.rstrip()}'
    
    content = re.sub(pattern1, replace_pattern1, content, flags=re.DOTALL)
    
    # 修复剩余的孤立conn.commit()调用
    content = re.sub(r'\s*conn\.commit\(\)\s*\n', '', content)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 修复完成: {file_path}")
    print(f"📁 备份保存在: {backup_path}")

if __name__ == "__main__":
    app_file = "app.py"
    if os.path.exists(app_file):
        fix_commit_issues(app_file)
    else:
        print(f"❌ 文件不存在: {app_file}")