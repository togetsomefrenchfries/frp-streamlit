#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动50参数超参数优化实验
"""

import sys
from pathlib import Path
import time

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """启动完整的50参数实验"""
    
    print("🚀 开始50参数超参数优化实验")
    print("=" * 60)
    print("实验配置:")
    print("  - 3个模型: RandomForest, XGBoost, LightGBM")
    print("  - 每个模型50个超参数配置")
    print("  - 5折交叉验证")
    print("  - 每5个配置保存一次中间结果")
    print("  - 预计总实验时间: 30-60分钟")
    print("=" * 60)
    
    # 确认开始
    print("\n⚠️  这将运行150个配置的完整实验，可能需要较长时间")
    confirm = input("确认开始实验吗? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes', '是']:
        print("❌ 实验已取消")
        return
    
    print("\n✅ 开始运行实验...")
    start_time = time.time()
    
    try:
        from run_50param_experiments import FiftyParameterExperiment
        
        # 创建实验实例
        experiment = FiftyParameterExperiment()
        
        # 运行实验
        experiment.run_experiment()
        
        # 计算总时间
        total_time = time.time() - start_time
        print(f"\n🎉 实验完成! 总用时: {total_time/60:.1f} 分钟")
        
        # 显示结果文件
        import os
        results_dir = "results"
        if os.path.exists(results_dir):
            print(f"\n📁 结果文件保存在: {results_dir}/")
            files = os.listdir(results_dir)
            for f in sorted(files):
                if f.endswith('.json') or f.endswith('.csv'):
                    print(f"   - {f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")
        print("💾 中间结果已保存，可以继续查看已完成的配置")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        print("💾 检查是否有中间结果保存")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()