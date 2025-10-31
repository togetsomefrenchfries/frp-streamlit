#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40参数实验启动脚本 - 使用Python 3.12.7环境
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 启动FRP 40参数优化实验")
    print("="*60)
    
    # 确保使用正确的Python环境
    python_exe = "F:/11/python.exe"
    script_path = "E:/大学/intern/2025-summer-concret/frp-streamlit/frp_local - or/run_40param_experiment.py"
    
    print(f"Python环境: {python_exe}")
    print(f"脚本路径: {script_path}")
    
    # 检查文件是否存在
    if not Path(script_path).exists():
        print(f"❌ 脚本文件不存在: {script_path}")
        return
    
    if not Path(python_exe).exists():
        print(f"❌ Python解释器不存在: {python_exe}")
        return
    
    print("\n✅ 开始执行实验...")
    print("="*60)
    
    try:
        # 使用正确的Python环境运行脚本
        subprocess.run([python_exe, script_path], check=True)
        print("\n🎉 实验完成!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验执行失败: {e}")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断实验")
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")

if __name__ == "__main__":
    main()