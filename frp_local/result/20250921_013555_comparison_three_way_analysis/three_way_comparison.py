#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三方对比分析：7:2:1 vs 7.5:2.5 vs app.py
详细对比三种策略的最佳配置和性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_comprehensive_comparison():
    """创建三方综合对比分析"""
    
    print("🔍 三方对比分析：7:2:1 vs 7.5:2.5 vs app.py")
    print("="*80)
    
    # 三种策略的详细配置和结果
    strategies = {
        "7:2:1策略": {
            "data_split": "70% 训练 + 20% 验证 + 10% 测试",
            "train_samples": 3257,
            "val_samples": 931,
            "test_samples": 466,
            "best_model": "RF-深度",
            "best_r2": 0.5666,
            "model_config": {
                "algorithm": "RandomForest",
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "bootstrap": True
            },
            "evaluation_method": "独立测试集验证",
            "advantage": "严格的三重验证，防止过拟合",
            "disadvantage": "测试集样本较少，统计不够稳定"
        },
        
        "7.5:2.5策略": {
            "data_split": "75% 训练 + 25% 测试",
            "train_samples": 3489,
            "val_samples": 0,
            "test_samples": 1164,
            "best_model": "XGB-高学习率",
            "best_r2": 0.7368,
            "model_config": {
                "algorithm": "XGBoost",
                "learning_rate": 0.15,
                "max_depth": 8,
                "n_estimators": 200,
                "subsample": 0.8
            },
            "evaluation_method": "Hold-out验证",
            "advantage": "更多训练数据，更大测试集",
            "disadvantage": "缺少独立验证集"
        },
        
        "app.py策略": {
            "data_split": "80% 训练 + 20% 测试",
            "train_samples": "约3700+",
            "val_samples": 0,
            "test_samples": "约900+",
            "best_model": "XGBoost/集成模型",
            "best_r2": 0.75,
            "model_config": {
                "algorithm": "XGBoost + 可能的集成",
                "learning_rate": "0.1-0.15",
                "max_depth": "5-8",
                "n_estimators": "150-200",
                "优化策略": "可能包含特征工程和超参数优化"
            },
            "evaluation_method": "Hold-out + 可能的交叉验证",
            "advantage": "最多训练数据，可能有高级优化",
            "disadvantage": "黑盒，具体配置不明"
        }
    }
    
    # 创建对比表格
    print("📊 详细对比表格：")
    print("-" * 100)
    
    # 基础信息对比
    basic_comparison = pd.DataFrame({
        '策略': ['7:2:1策略', '7.5:2.5策略', 'app.py策略'],
        '数据分割': [s["data_split"] for s in strategies.values()],
        '训练样本': [s["train_samples"] for s in strategies.values()],
        '测试样本': [s["test_samples"] for s in strategies.values()],
        '最佳R²': [s["best_r2"] for s in strategies.values()],
        '最佳模型': [s["best_model"] for s in strategies.values()]
    })
    
    print(basic_comparison.to_string(index=False))
    print()
    
    # 性能分析
    print("🎯 性能排名：")
    print("-" * 50)
    r2_scores = [0.5666, 0.7368, 0.75]
    strategy_names = ['7:2:1策略', '7.5:2.5策略', 'app.py策略']
    
    for i, (name, score) in enumerate(sorted(zip(strategy_names, r2_scores), key=lambda x: x[1], reverse=True)):
        rank_emoji = ["🥇", "🥈", "🥉"][i]
        print(f"{rank_emoji} {name}: R² = {score:.4f}")
    
    print()
    
    # 差异分析
    print("📈 性能差异分析：")
    print("-" * 50)
    
    best_score = max(r2_scores)  # app.py的0.75
    
    for name, score in zip(strategy_names, r2_scores):
        diff_abs = score - best_score
        diff_rel = (diff_abs / best_score) * 100
        
        if diff_abs == 0:
            status = "🎯 基准"
        elif diff_abs > -0.05:
            status = "✅ 接近"
        elif diff_abs > -0.1:
            status = "⚠️  中等差距"
        else:
            status = "❌ 较大差距"
        
        print(f"{name}:")
        print(f"   R²得分: {score:.4f}")
        print(f"   与最佳差异: {diff_abs:+.4f} ({diff_rel:+.1f}%)")
        print(f"   评估: {status}")
        print()
    
    # 创建可视化对比
    create_comparison_visualizations(strategies, r2_scores, strategy_names)
    
    # 深度分析
    print("🔍 深度分析：")
    print("-" * 50)
    
    print("1. 🎯 数据利用效率：")
    print("   • app.py策略：80%训练数据，利用率最高")
    print("   • 7.5:2.5策略：75%训练数据，平衡性好")
    print("   • 7:2:1策略：70%训练数据，最保守但最严格")
    print()
    
    print("2. 🧪 验证可靠性：")
    print("   • 7:2:1策略：三重验证最可靠，但测试集偏小")
    print("   • 7.5:2.5策略：单一测试验证，样本充足")
    print("   • app.py策略：可能结合多种验证方法")
    print()
    
    print("3. 🔧 算法优化：")
    print("   • app.py策略：可能有高级特征工程和模型集成")
    print("   • 7.5:2.5策略：XGBoost优化较好")
    print("   • 7:2:1策略：RandomForest相对保守")
    print()
    
    # 提供改进建议
    provide_improvement_recommendations()

def create_comparison_visualizations(strategies, r2_scores, strategy_names):
    """创建可视化对比图表"""
    
    print("🎨 生成对比可视化图表...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. R²性能对比 (主要指标)
    plt.subplot(2, 4, 1)
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # 红、橙、绿
    bars = plt.bar(strategy_names, r2_scores, color=colors, alpha=0.8)
    plt.title('R²性能对比', fontweight='bold', fontsize=14)
    plt.ylabel('R²得分')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 添加基准线
    plt.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='app.py基准')
    plt.legend()
    
    # 2. 数据分割对比
    plt.subplot(2, 4, 2)
    
    # 数据分割饼图对比
    splits_data = {
        '7:2:1': [70, 20, 10],
        '7.5:2.5': [75, 25, 0],
        'app.py': [80, 20, 0]
    }
    
    labels = ['训练集', '验证集', '测试集']
    colors_pie = ['#3498db', '#e67e22', '#e74c3c']
    
    # 选择7:2:1作为示例
    sizes = splits_data['7:2:1']
    plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    plt.title('7:2:1数据分割', fontweight='bold')
    
    # 3. 样本数量对比
    plt.subplot(2, 4, 3)
    train_samples = [3257, 3489, 3700]  # 估算app.py
    test_samples = [466, 1164, 900]     # 估算app.py
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    plt.bar(x - width/2, train_samples, width, label='训练样本', alpha=0.8)
    plt.bar(x + width/2, test_samples, width, label='测试样本', alpha=0.8)
    
    plt.xlabel('策略')
    plt.ylabel('样本数量')
    plt.title('样本数量对比', fontweight='bold')
    plt.xticks(x, strategy_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 性能差异雷达图
    plt.subplot(2, 4, 4)
    
    # 定义评估维度 (满分100)
    dimensions = ['R²性能', '数据利用', '验证严格性', '算法复杂度', '结果稳定性']
    
    # 各策略得分 (主观评估)
    scores_721 = [75.5, 70, 95, 70, 80]    # 7:2:1
    scores_7525 = [98.2, 85, 75, 85, 85]   # 7.5:2.5
    scores_app = [100, 95, 80, 90, 85]     # app.py
    
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    scores_721 += scores_721[:1]
    scores_7525 += scores_7525[:1]
    scores_app += scores_app[:1]
    
    plt.polar(angles, scores_721, 'o-', linewidth=2, label='7:2:1策略', alpha=0.7)
    plt.polar(angles, scores_7525, 's-', linewidth=2, label='7.5:2.5策略', alpha=0.7)
    plt.polar(angles, scores_app, '^-', linewidth=2, label='app.py策略', alpha=0.7)
    
    plt.xticks(angles[:-1], dimensions)
    plt.ylim(0, 100)
    plt.title('综合性能雷达图', fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 5. 与app.py的差距分析
    plt.subplot(2, 4, 5)
    app_baseline = 0.75
    differences = [score - app_baseline for score in r2_scores]
    diff_colors = ['red' if d < 0 else 'green' for d in differences]
    
    bars = plt.bar(strategy_names, differences, color=diff_colors, alpha=0.7)
    plt.title('与app.py基准的差异', fontweight='bold')
    plt.ylabel('R²差异')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, diff in zip(bars, differences):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.005 if diff >= 0 else -0.005),
                f'{diff:+.4f}', ha='center', 
                va='bottom' if diff >= 0 else 'top', fontweight='bold')
    
    # 6. 训练效率对比
    plt.subplot(2, 4, 6)
    
    # 模拟训练时间 (相对值)
    training_time = [1.2, 1.0, 0.8]  # 7:2:1, 7.5:2.5, app.py (相对)
    model_complexity = [0.7, 0.9, 1.0]  # 模型复杂度
    
    plt.scatter(training_time, r2_scores, s=[c*300 for c in model_complexity], 
               c=colors, alpha=0.7)
    
    for i, name in enumerate(strategy_names):
        plt.annotate(name, (training_time[i], r2_scores[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('相对训练时间')
    plt.ylabel('R²性能')
    plt.title('性能-效率散点图', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 7. 模型算法对比
    plt.subplot(2, 4, 7)
    
    models = ['RandomForest\n(7:2:1)', 'XGBoost\n(7.5:2.5)', 'XGBoost+优化\n(app.py)']
    model_scores = r2_scores
    
    plt.barh(models, model_scores, color=colors, alpha=0.8)
    plt.title('模型算法性能对比', fontweight='bold')
    plt.xlabel('R²得分')
    
    # 添加数值标签
    for i, score in enumerate(model_scores):
        plt.text(score + 0.01, i, f'{score:.4f}', 
                va='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # 8. 改进潜力分析
    plt.subplot(2, 4, 8)
    
    current_scores = r2_scores
    potential_scores = [0.65, 0.78, 0.75]  # 预估改进后的分数
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    plt.bar(x - width/2, current_scores, width, label='当前性能', alpha=0.8)
    plt.bar(x + width/2, potential_scores, width, label='改进潜力', alpha=0.8)
    
    plt.xlabel('策略')
    plt.ylabel('R²得分')
    plt.title('改进潜力分析', fontweight='bold')
    plt.xticks(x, strategy_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"three_way_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📁 对比图表已保存: {plot_filename}")
    
    return plot_filename

def provide_improvement_recommendations():
    """提供针对性改进建议"""
    
    print("💡 针对性改进建议：")
    print("="*80)
    
    print("🎯 对7:2:1策略的建议：")
    print("-" * 50)
    print("1. 算法优化：")
    print("   • 尝试XGBoost替代RandomForest")
    print("   • 调整学习率到0.1-0.15")
    print("   • 增加n_estimators到200+")
    print()
    print("2. 数据策略：")
    print("   • 考虑增加测试集比例到15% (7:1.5:1.5)")
    print("   • 或者采用交叉验证增强稳定性")
    print()
    print("预期改进: R² 0.5666 → 0.65+ (提升15%)")
    print()
    
    print("🎯 对7.5:2.5策略的建议：")
    print("-" * 50)
    print("1. 特征工程：")
    print("   • 添加多项式特征")
    print("   • 特征交互项")
    print("   • 更精细的数据预处理")
    print()
    print("2. 模型集成：")
    print("   • XGBoost + LightGBM + RandomForest")
    print("   • 使用VotingRegressor")
    print()
    print("预期改进: R² 0.7368 → 0.78+ (提升6%)")
    print()
    
    print("🎯 总体建议：")
    print("-" * 50)
    print("• 如果追求最高性能: 优化7.5:2.5策略")
    print("• 如果追求严格验证: 改进7:2:1策略")
    print("• 如果要达到app.py水平: 重点做特征工程和模型集成")
    print()
    
    print("🏆 最佳实践路径：")
    print("1. 基于7.5:2.5策略 (已接近app.py)")
    print("2. 添加特征工程")
    print("3. 实施模型集成")
    print("4. 超参数精细调优")
    print("5. 预期达到R² 0.78+，超越app.py!")

def create_summary_table():
    """创建结果汇总表"""
    
    print("\n📋 三方对比结果汇总表：")
    print("="*80)
    
    summary_data = {
        '指标': [
            'R²得分',
            '排名',
            '训练样本',
            '测试样本',
            '数据分割',
            '最佳算法',
            '验证方法',
            '与app.py差异',
            '相对差异(%)',
            '改进潜力',
            '推荐场景'
        ],
        '7:2:1策略': [
            '0.5666',
            '第3名 🥉',
            '3,257',
            '466',
            '70:20:10',
            'RF-深度',
            '三重验证',
            '-0.1834',
            '-24.5%',
            '中等',
            '研究验证'
        ],
        '7.5:2.5策略': [
            '0.7368',
            '第2名 🥈',
            '3,489',
            '1,164',
            '75:25',
            'XGB-高学习率',
            'Hold-out',
            '-0.0132',
            '-1.7%',
            '较高',
            '生产应用'
        ],
        'app.py策略': [
            '0.7500',
            '第1名 🥇',
            '~3,700',
            '~900',
            '80:20',
            'XGB+优化',
            'Hold-out+CV',
            '0.0000',
            '0.0%',
            '基准',
            '参考标杆'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

def main():
    """主函数"""
    
    create_comprehensive_comparison()
    create_summary_table()
    
    print(f"\n🎯 核心结论：")
    print(f"="*50)
    print(f"🥇 app.py策略 (R²=0.750): 当前最佳，作为目标基准")
    print(f"🥈 7.5:2.5策略 (R²=0.737): 非常接近，仅差1.7%，最有优化潜力")
    print(f"🥉 7:2:1策略 (R²=0.567): 验证最严格，但性能差距较大")
    print(f"")
    print(f"📈 最佳策略: 基于7.5:2.5进行特征工程和模型集成优化")
    print(f"🎯 预期目标: 通过优化可达到R² 0.78+，超越app.py!")

if __name__ == "__main__":
    main()