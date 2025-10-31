#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - FRP 450参数实验 (全纤维类型版本)

使用方法：
1. 确保数据文件存在
2. 运行此脚本： python quick_start_450param.py
3. 查看实验结果

这个脚本会：
- 自动检查环境和数据
- 运行450参数优化实验（RF 150个，XGBoost 150个，LightGBM 150个）
- 支持Glass和Basalt纤维类型
- 不限制纤维类型，使用全数据集
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
    """运行450参数实验"""
    print("\n🚀 启动450参数优化实验...")
    
    # 修改run_40param_experiment.py来支持450参数
    experiment_script = "run_450param_experiment.py"
    
    # 如果不存在450参数脚本，先从40参数脚本复制并修改
    if not Path(experiment_script).exists():
        print("📝 创建450参数实验脚本...")
        create_450param_script()
    
    try:
        # 运行实验
        result = subprocess.run([
            sys.executable, experiment_script
        ], 
        capture_output=True, 
        text=True, 
        timeout=7200  # 2小时超时
        )
        
        if result.returncode == 0:
            print("✅ 实验成功完成！")
            print("\n" + result.stdout)
        else:
            print("❌ 实验失败！")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 实验超时（2小时）")
        return False
    except Exception as e:
        print(f"❌ 运行实验时出错: {e}")
        return False
    
    return True

def create_450param_script():
    """创建450参数实验脚本"""
    
    # 读取原始40参数脚本
    source_script = "run_40param_experiment.py"
    if not Path(source_script).exists():
        print(f"❌ 源脚本 {source_script} 不存在")
        return False
    
    with open(source_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改参数配置以支持450个配置
    modifications = [
        # 修改RandomForest配置数量
        ("'n_estimators': [50, 100, 200]", "'n_estimators': [50, 100, 150, 200, 250, 300]"),
        ("'max_depth': [3, 5, 7, 10, None]", "'max_depth': [3, 4, 5, 6, 7, 8, 10, 12, 15, None]"),
        ("'min_samples_split': [2, 5, 10]", "'min_samples_split': [2, 3, 4, 5, 7, 10, 15]"),
        ("'min_samples_leaf': [1, 2, 4]", "'min_samples_leaf': [1, 2, 3, 4, 5, 7]"),
        
        # 修改XGBoost配置数量
        ("'n_estimators': [100, 200, 300]", "'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500]"),
        ("'max_depth': [3, 6, 9]", "'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12]"),
        ("'learning_rate': [0.01, 0.1, 0.2]", "'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]"),
        
        # 修改LightGBM配置数量
        ("'num_leaves': [31, 50, 100]", "'num_leaves': [15, 31, 50, 70, 100, 150, 200, 250]"),
        
        # 修改总配置数量描述
        ("40个配置", "450个配置"),
        ("40param", "450param"),
        ("总配置数: 40", "总配置数: 450"),
        ("RandomForest: 15个配置", "RandomForest: 150个配置"),
        ("XGBoost: 13个配置", "XGBoost: 150个配置"), 
        ("LightGBM: 12个配置", "LightGBM: 150个配置"),
    ]
    
    # 应用修改
    modified_content = content
    for old, new in modifications:
        modified_content = modified_content.replace(old, new)
    
    # 添加注释说明这是全纤维类型版本
    header_comment = '''"""
FRP 450参数实验 - 全纤维类型版本

与basalt版本的区别：
- 不限制纤维类型，包含Glass和Basalt数据
- 使用更大的数据集进行训练
- 450个超参数配置（每个模型150个）

使用预定义特征，与app.py保持一致
"""

'''
    
    # 在文件开头添加说明
    if '"""' in modified_content:
        first_docstring_end = modified_content.find('"""', modified_content.find('"""') + 3) + 3
        modified_content = modified_content[:first_docstring_end] + '\n' + header_comment + modified_content[first_docstring_end:]
    
    # 保存修改后的脚本
    target_script = "run_450param_experiment.py"
    with open(target_script, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ 已创建 {target_script}")
    return True

def show_results():
    """显示实验结果"""
    print(f"\n📊 查找实验结果...")
    
    # 查找最新的实验结果文件
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("❌ experiments目录不存在")
        return
    
    # 查找最新的450参数实验结果
    result_files = list(experiments_dir.glob("450param_exp_*.csv"))
    if not result_files:
        # 如果没有450参数结果，查找其他结果文件
        result_files = list(experiments_dir.glob("*param_exp_*.csv"))
    
    if result_files:
        # 按修改时间排序，获取最新的
        latest_file = max(result_files, key=os.path.getmtime)
        print(f"📁 最新结果文件: {latest_file}")
        print(f"📁 结果目录: {experiments_dir.absolute()}")
        
        # 简单的结果预览
        try:
            import pandas as pd
            df = pd.read_csv(latest_file)
            print(f"📈 实验结果快速预览:")
            print(f"   总配置数: {len(df)}")
            print(f"   最佳R²: {df['test_r2'].max():.6f}")
            print(f"   平均R²: {df['test_r2'].mean():.6f}")
            
            # Top 3 结果
            top3 = df.nlargest(3, 'test_r2')
            print(f"\n🏆 TOP3 配置:")
            for i, row in enumerate(top3.itertuples(), 1):
                print(f"   {i}. {row.model} #{row.config_id}: R²={row.test_r2:.6f}")
                
        except ImportError:
            print("⚠️  无法预览结果（需要pandas库）")
        except Exception as e:
            print(f"⚠️  预览结果时出错: {e}")
    else:
        print("❌ 未找到实验结果文件")

def main():
    """主函数"""
    print("🎯 FRP 450参数实验 - 快速启动 (全纤维类型版本)")
    print("=" * 60)
    print("📋 RandomForest: 150个配置")
    print("📋 XGBoost: 150个配置")
    print("📋 LightGBM: 150个配置")
    print("📋 支持Glass和Basalt纤维类型")
    print("📋 不限制纤维类型，使用全数据集")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return False
    
    # 检查数据文件
    if not check_data_files():
        print("\n❌ 数据文件检查失败，请确保数据文件存在")
        return False
    
    # 检查脚本文件
    if not check_script_files():
        print("\n❌ 脚本文件检查失败，请确保所有必要文件存在")
        return False
    
    # 运行实验
    if not run_experiment():
        print("\n❌ 实验运行失败")
        return False
    
    # 显示结果
    show_results()
    
    print(f"\n🎉 实验完成！")
    print(f"\n下一步可以：")
    print(f"1. 查看 experiments/ 目录中的详细结果")
    print(f"2. 分析最佳参数配置")
    print(f"3. 使用最佳配置训练最终模型")
    print(f"4. 比较全纤维类型版本与basalt限制版本的性能差异")
    
    return True

if __name__ == "__main__":
    main()