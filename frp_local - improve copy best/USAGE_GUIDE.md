# 🚀 FRP 40参数实验使用指南

## 📋 概述

这个实验包含**40个参数配置**，分布在3种机器学习模型中：
- **RandomForest**: 14个配置
- **XGBoost**: 13个配置  
- **LightGBM**: 13个配置

每个配置都会进行5折交叉验证，并在测试集上评估性能。

## 🎯 推荐运行方式

### 方式1: 快速启动（推荐）
```bash
cd "frp_local - or"
python quick_start_40param.py
```

这个脚本会：
- ✅ 自动检查环境和依赖
- ✅ 检查数据文件是否存在
- ✅ 运行40参数实验
- ✅ 显示结果摘要

### 方式2: 直接运行实验
```bash
cd "frp_local - or"
python run_40param_experiment.py
```

### 方式3: 测试修改效果
```bash
cd "frp_local - or"
python test_modifications.py
```

## 📂 数据要求

脚本会自动查找以下位置的数据文件：
- `data/research_data.xlsx`
- `../data/research_data.xlsx`
- `../../data/research_data.xlsx`
- `data/train_data.xlsx`
- `../data/train_data.xlsx`

确保至少有一个位置存在Excel数据文件。

## 🔧 环境要求

### 必需库：
```bash
pip install pandas numpy scikit-learn
```

### 可选库（获得完整功能）：
```bash
pip install xgboost lightgbm
```

如果没有安装XGBoost或LightGBM，实验会自动跳过相应的模型。

## 📊 实验配置详情

### RandomForest配置 (14个)
- n_estimators: 50-500
- max_depth: 4-20
- min_samples_split: 2-6
- 重点测试不同的树数量和深度组合

### XGBoost配置 (13个)
- n_estimators: 100-500
- max_depth: 3-10
- learning_rate: 0.02-0.2
- 重点测试不同的学习率和树深度

### LightGBM配置 (13个)
- n_estimators: 100-500
- max_depth: 3-12
- learning_rate: 0.02-0.2
- num_leaves: 7-4095
- 重点测试叶子数量和深度的平衡

## 📈 结果输出

实验完成后会生成：

### 1. CSV结果文件
- 位置: `experiments/40param_exp_YYYYMMDD_HHMMSS.csv`
- 包含所有配置的性能指标

### 2. 详细JSON文件
- 位置: `experiments/40param_exp_YYYYMMDD_HHMMSS_detailed.json`
- 包含完整的实验配置和结果

### 3. 控制台输出
- 实时显示每个配置的训练进度
- 最终结果摘要
- TOP5最佳配置

## 🎯 数据筛选逻辑

根据你的要求，实验使用以下数据筛选策略：

### ✅ 保留的筛选条件：
1. **第一列验证**: 只使用Comments=1的数据
2. **pH自动分配**: sea water等自动分配pH值
3. **SMD处理**: 将SMD转换为NaN

### ❌ 移除的筛选条件：
1. **BU列SMD检测**: 不再检查BU列是否为SMD

## ⏱️ 预计运行时间

- **小数据集** (< 1000行): 5-15分钟
- **中等数据集** (1000-5000行): 15-45分钟  
- **大数据集** (> 5000行): 45分钟-2小时

实验每5个配置会保存一次中间结果，避免意外中断丢失数据。

## 🎉 使用建议

1. **首次运行**: 使用`quick_start_40param.py`，它会做全面检查
2. **调试问题**: 先运行`test_modifications.py`确保修改正确
3. **大数据集**: 建议在性能较好的机器上运行
4. **结果分析**: 关注R²指标，通常>0.7表示较好的预测性能

## 🆘 常见问题

### Q: 提示找不到数据文件？
A: 确保数据文件在指定位置，或修改脚本中的路径

### Q: 某些模型被跳过？
A: 检查是否安装了xgboost和lightgbm库

### Q: 内存不足？
A: 可以减少数据量或使用更少的参数配置

### Q: 实验中断了？
A: 检查experiments目录，中间结果已保存

## 📞 下一步

实验完成后，你可以：
1. 分析最佳参数配置
2. 使用最佳配置训练最终模型
3. 根据结果调整参数搜索范围
4. 与之前的实验结果对比

祝实验顺利！🎯