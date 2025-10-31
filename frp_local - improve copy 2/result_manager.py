#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果管理器 - 统一管理所有实验结果的输出
按时间和策略分类保存结果文件
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import json

class ResultManager:
    """结果管理器类"""
    
    def __init__(self, base_dir="result"):
        """
        初始化结果管理器
        
        Args:
            base_dir: 结果保存的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def create_result_folder(self, strategy_name, experiment_type="general"):
        """
        创建结果文件夹
        
        Args:
            strategy_name: 策略名称 (如 "7.5_2.5", "7_2_1")
            experiment_type: 实验类型 (如 "parameter_exp", "metrics_analysis", "comparison")
            
        Returns:
            Path: 创建的文件夹路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{strategy_name}_{experiment_type}"
        
        result_folder = self.base_dir / folder_name
        result_folder.mkdir(exist_ok=True)
        
        # 创建元数据文件
        metadata = {
            "timestamp": timestamp,
            "strategy": strategy_name,
            "experiment_type": experiment_type,
            "folder_name": folder_name,
            "created_at": datetime.now().isoformat(),
            "description": f"{strategy_name}策略的{experiment_type}实验结果"
        }
        
        with open(result_folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"📁 创建结果文件夹: {result_folder}")
        return result_folder
    
    def save_results(self, strategy_name, experiment_type, files_to_save, descriptions=None):
        """
        保存结果文件到指定文件夹
        
        Args:
            strategy_name: 策略名称
            experiment_type: 实验类型
            files_to_save: 要保存的文件列表
            descriptions: 文件描述字典
        """
        result_folder = self.create_result_folder(strategy_name, experiment_type)
        
        saved_files = []
        for file_path in files_to_save:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                destination = result_folder / file_name
                
                try:
                    shutil.copy2(file_path, destination)
                    saved_files.append(str(destination))
                    print(f"✅ 已保存: {file_name} -> {destination}")
                except Exception as e:
                    print(f"❌ 保存失败: {file_name} - {e}")
            else:
                print(f"⚠️  文件不存在: {file_path}")
        
        # 更新元数据
        metadata_file = result_folder / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            metadata["saved_files"] = saved_files
            metadata["file_descriptions"] = descriptions or {}
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return result_folder
    
    def list_results(self, strategy_filter=None, experiment_filter=None):
        """
        列出所有结果文件夹
        
        Args:
            strategy_filter: 策略过滤器
            experiment_filter: 实验类型过滤器
        """
        print(f"\n📊 结果文件夹列表:")
        print("="*80)
        
        folders = sorted(self.base_dir.glob("*"), reverse=True)  # 按时间倒序
        
        for folder in folders:
            if folder.is_dir():
                metadata_file = folder / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    strategy = metadata.get("strategy", "unknown")
                    exp_type = metadata.get("experiment_type", "unknown")
                    
                    # 应用过滤器
                    if strategy_filter and strategy_filter not in strategy:
                        continue
                    if experiment_filter and experiment_filter not in exp_type:
                        continue
                    
                    print(f"📁 {folder.name}")
                    print(f"   策略: {strategy}")
                    print(f"   类型: {exp_type}")
                    print(f"   时间: {metadata.get('created_at', 'unknown')}")
                    print(f"   描述: {metadata.get('description', 'N/A')}")
                    
                    if "saved_files" in metadata:
                        print(f"   文件: {len(metadata['saved_files'])} 个")
                    print()
    
    def cleanup_old_results(self, keep_days=30):
        """
        清理旧的结果文件夹
        
        Args:
            keep_days: 保留天数
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        cleaned_count = 0
        for folder in self.base_dir.glob("*"):
            if folder.is_dir():
                try:
                    # 从文件夹名提取时间戳
                    timestamp_str = folder.name.split("_")[0]
                    folder_date = datetime.strptime(timestamp_str, "%Y%m%d")
                    
                    if folder_date < cutoff_date:
                        shutil.rmtree(folder)
                        print(f"🗑️  删除旧结果: {folder.name}")
                        cleaned_count += 1
                except:
                    continue  # 跳过无法解析的文件夹
        
        print(f"✅ 清理完成，删除了 {cleaned_count} 个旧结果文件夹")


def create_enhanced_experiment_scripts():
    """创建增强的实验脚本，自动保存结果"""
    
    # 1. 增强参数实验脚本
    enhanced_param_exp = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版参数实验 - 自动保存结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_parameter_experiments import main as run_param_experiments
from result_manager import ResultManager
import glob

def main():
    """运行参数实验并保存结果"""
    
    # 运行原始实验
    print("🚀 开始参数实验...")
    run_param_experiments()
    
    # 保存结果
    result_manager = ResultManager()
    
    # 查找生成的文件
    files_to_save = []
    
    # 查找图表文件
    chart_files = glob.glob("parameter_experiments_*.png")
    files_to_save.extend(chart_files)
    
    # 查找结果文件
    result_files = glob.glob("*results*.txt") + glob.glob("*results*.csv")
    files_to_save.extend(result_files)
    
    # 保存脚本本身
    files_to_save.append("run_parameter_experiments.py")
    files_to_save.append("enhanced_parameter_experiments.py")
    
    descriptions = {
        "parameter_experiments_*.png": "参数实验对比图表",
        "*results*.txt": "实验结果文本文件",
        "*results*.csv": "实验结果数据文件",
        "run_parameter_experiments.py": "参数实验脚本",
        "enhanced_parameter_experiments.py": "增强版实验脚本"
    }
    
    # 保存到7.5:2.5策略文件夹
    result_folder = result_manager.save_results(
        strategy_name="7.5_2.5",
        experiment_type="parameter_exp",
        files_to_save=files_to_save,
        descriptions=descriptions
    )
    
    print(f"✅ 参数实验结果已保存到: {result_folder}")

if __name__ == "__main__":
    main()
'''
    
    # 2. 增强721实验脚本
    enhanced_721_exp = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版7:2:1实验 - 自动保存结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_experiments_721 import main as run_721_experiments
from result_manager import ResultManager
import glob

def main():
    """运行7:2:1实验并保存结果"""
    
    # 运行原始实验
    print("🚀 开始7:2:1实验...")
    run_721_experiments()
    
    # 保存结果
    result_manager = ResultManager()
    
    # 查找生成的文件
    files_to_save = []
    
    # 查找图表文件
    chart_files = glob.glob("ml_experiments_721_*.png")
    files_to_save.extend(chart_files)
    
    # 查找结果文件
    result_files = glob.glob("*721*.txt") + glob.glob("*721*.csv")
    files_to_save.extend(result_files)
    
    # 保存脚本本身
    files_to_save.append("ml_experiments_721.py")
    files_to_save.append("enhanced_721_experiments.py")
    
    descriptions = {
        "ml_experiments_721_*.png": "7:2:1实验对比图表",
        "*721*.txt": "7:2:1实验结果文本文件",
        "*721*.csv": "7:2:1实验结果数据文件",
        "ml_experiments_721.py": "7:2:1实验脚本",
        "enhanced_721_experiments.py": "增强版7:2:1实验脚本"
    }
    
    # 保存到7:2:1策略文件夹
    result_folder = result_manager.save_results(
        strategy_name="7_2_1",
        experiment_type="ml_experiments",
        files_to_save=files_to_save,
        descriptions=descriptions
    )
    
    print(f"✅ 7:2:1实验结果已保存到: {result_folder}")

if __name__ == "__main__":
    main()
'''
    
    # 3. 增强指标分析脚本
    enhanced_metrics_analysis = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版指标分析 - 自动保存结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_metrics_analysis import main as run_metrics_analysis
from result_manager import ResultManager
import glob

def main():
    """运行指标分析并保存结果"""
    
    # 运行原始分析
    print("🚀 开始指标分析...")
    run_metrics_analysis()
    
    # 保存结果
    result_manager = ResultManager()
    
    # 查找生成的文件
    files_to_save = []
    
    # 查找图表文件
    chart_files = glob.glob("prediction_metrics_*.png")
    files_to_save.extend(chart_files)
    
    # 查找结果文件
    result_files = glob.glob("*metrics*.txt") + glob.glob("*metrics*.csv")
    files_to_save.extend(result_files)
    
    # 保存脚本本身
    files_to_save.append("prediction_metrics_analysis.py")
    files_to_save.append("enhanced_metrics_analysis.py")
    
    descriptions = {
        "prediction_metrics_*.png": "预测指标分析图表",
        "*metrics*.txt": "指标分析结果文本文件",
        "*metrics*.csv": "指标分析结果数据文件",
        "prediction_metrics_analysis.py": "指标分析脚本",
        "enhanced_metrics_analysis.py": "增强版指标分析脚本"
    }
    
    # 保存到metrics分析文件夹
    result_folder = result_manager.save_results(
        strategy_name="metrics",
        experiment_type="analysis",
        files_to_save=files_to_save,
        descriptions=descriptions
    )
    
    print(f"✅ 指标分析结果已保存到: {result_folder}")

if __name__ == "__main__":
    main()
'''
    
    return enhanced_param_exp, enhanced_721_exp, enhanced_metrics_analysis

def main():
    """主函数 - 演示结果管理器的使用"""
    
    print("📁 结果管理器初始化完成！")
    print("="*60)
    
    # 创建增强脚本
    enhanced_param_exp, enhanced_721_exp, enhanced_metrics_analysis = create_enhanced_experiment_scripts()
    
    # 保存增强脚本
    with open("enhanced_parameter_experiments.py", "w", encoding="utf-8") as f:
        f.write(enhanced_param_exp)
    
    with open("enhanced_721_experiments.py", "w", encoding="utf-8") as f:
        f.write(enhanced_721_exp)
    
    with open("enhanced_metrics_analysis.py", "w", encoding="utf-8") as f:
        f.write(enhanced_metrics_analysis)
    
    print("✅ 已创建增强版实验脚本:")
    print("   • enhanced_parameter_experiments.py")
    print("   • enhanced_721_experiments.py") 
    print("   • enhanced_metrics_analysis.py")
    print()
    
    # 演示使用方法
    print("📖 使用方法:")
    print("-" * 40)
    print("1. 运行实验并自动保存结果:")
    print("   python enhanced_parameter_experiments.py")
    print("   python enhanced_721_experiments.py")
    print("   python enhanced_metrics_analysis.py")
    print()
    
    print("2. 手动使用结果管理器:")
    print("   from result_manager import ResultManager")
    print("   rm = ResultManager()")
    print("   rm.save_results('策略名', '实验类型', ['文件1', '文件2'])")
    print()
    
    print("3. 查看结果列表:")
    print("   rm.list_results()")
    print("   rm.list_results(strategy_filter='7_2_1')")
    print()
    
    print("4. 清理旧结果:")
    print("   rm.cleanup_old_results(keep_days=30)")

if __name__ == "__main__":
    main()