#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - FRP 40参数实验

使用方法：
1. 确保数据文件存在
2. 运行此脚本： python quick_start_40param.py
3. 查看实验结果

这个脚本会：
- 自动检查环境和数据
- 运行40参数优化实验
- 生成详细的结果报告
"""

import sys
import os
from pathlib import Path
import subprocess

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的库
    required_packages = ['pandas', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: 未安装")
    
    # 检查可选的库
    optional_packages = ['xgboost', 'lightgbm']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"⚠️  {package}: 未安装（将跳过此模型）")
    
    if missing_packages:
        print(f"\n❌ 缺少必要库: {missing_packages}")
        print("请运行: pip install pandas numpy scikit-learn")
        return False
    
    return True

def check_data_files():
    """检查数据文件"""
    print("\n🔍 检查数据文件...")
    
    possible_paths = [
        "E:/大学/intern/2025-summer-concret/database 4.xlsx",  # 用户指定的路径
        "../database 4.xlsx",
        "../../database 4.xlsx", 
        "../../../database 4.xlsx",
        "data/research_data.xlsx",
        "../data/research_data.xlsx", 
        "../../data/research_data.xlsx",
        "data/train_data.xlsx",
        "../data/train_data.xlsx"
    ]
    
    data_file = None
    for path in possible_paths:
        if Path(path).exists():
            data_file = path
            print(f"✅ 找到数据文件: {path}")
            break
    
    if not data_file:
        print("❌ 未找到数据文件")
        print("请确保以下路径之一存在数据文件:")
        for path in possible_paths:
            print(f"   - {path}")
        return False
    
    return True

def check_script_files():
    """检查脚本文件"""
    print("\n🔍 检查脚本文件...")
    
    required_files = [
        "run_40param_experiment.py",
        "preprocessor.py",
        "data_loader.py",
        "config.py",
        "utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}: 存在")
        else:
            missing_files.append(file)
            print(f"❌ {file}: 缺失")
    
    if missing_files:
        print(f"\n❌ 缺少必要文件: {missing_files}")
        return False
    
    return True

def run_experiment():
    """运行40参数实验"""
    print("\n🚀 启动40参数优化实验...")
    print("=" * 60)
    
    try:
        # 运行实验脚本
        result = subprocess.run([
            sys.executable, "run_40param_experiment.py"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ 实验成功完成！")
            return True
        else:
            print(f"\n❌ 实验失败，退出码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行实验时出错: {e}")
        return False

def show_results():
    """显示实验结果"""
    print("\n📊 查找实验结果...")
    
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("❌ 实验结果目录不存在")
        return
    
    # 查找最新的40参数实验结果
    csv_files = list(experiments_dir.glob("40param_exp_*.csv"))
    if not csv_files:
        print("❌ 未找到实验结果文件")
        return
    
    # 按修改时间排序，获取最新的
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 最新结果文件: {latest_file}")
    print(f"📁 结果目录: {experiments_dir.absolute()}")
    
    try:
        import pandas as pd
        df = pd.read_csv(latest_file)
        
        print(f"\n📈 实验结果快速预览:")
        print(f"   总配置数: {len(df)}")
        print(f"   最佳R²: {df['test_r2'].max():.6f}")
        print(f"   平均R²: {df['test_r2'].mean():.6f}")
        
        print(f"\n🏆 TOP3 配置:")
        top3 = df.nlargest(3, 'test_r2')
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            print(f"   {i}. {row['model']} #{row['config_id']}: R²={row['test_r2']:.6f}")
            
    except Exception as e:
        print(f"❌ 读取结果文件失败: {e}")

def main():
    """主函数"""
    print("🎯 FRP 40参数实验 - 快速启动")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        return
    
    # 检查数据文件
    if not check_data_files():
        return
    
    # 检查脚本文件
    if not check_script_files():
        return
    
    print("\n✅ 所有检查通过，准备运行实验")
    
    # 询问用户是否继续
    response = input("\n是否开始运行40参数实验？(y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("实验取消")
        return
    
    # 运行实验
    success = run_experiment()
    
    if success:
        # 显示结果
        show_results()
        
        print("\n🎉 实验完成！")
        print("\n下一步可以：")
        print("1. 查看 experiments/ 目录中的详细结果")
        print("2. 分析最佳参数配置")
        print("3. 使用最佳配置训练最终模型")
    else:
        print("\n😞 实验失败，请检查错误信息")

if __name__ == "__main__":
    main()