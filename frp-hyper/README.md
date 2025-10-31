# FRP超参数优化实验# FRP钢筋耐久性预测系统 - 本地版本



专门用于FRP钢筋耐久性预测的超参数优化实验项目。## 📚 项目概述



## 主要特性这是一个从原始Streamlit web应用中提取的本地版本机器学习系统，专门用于预测FRP（纤维增强聚合物）钢筋在各种环境条件下的耐久性。系统采用模块化设计，易于维护和扩展。



- **7:2:1数据分割**: 使用训练集(70%):验证集(20%):测试集(10%)的分割策略## 🏗️ 系统架构

- **超参数优化**: 支持网格搜索和随机搜索

- **验证集使用**: 正确使用验证集进行超参数调优### 核心模块

- **多模型支持**: 随机森林、XGBoost、LightGBM```

- **结果管理**: 自动保存和汇总实验结果frp_local/

├── config.py          # 配置管理

## 使用方法├── data_loader.py      # 数据加载

├── preprocessor.py     # 数据预处理

```bash├── model_trainer.py    # 模型训练

# 运行超参数优化实验├── predictor.py        # 预测模块

python run_parameter_experiments.py├── utils.py           # 工具函数

```├── main.py            # 主程序入口

├── example_usage.py   # 使用示例

## 输出文件└── data/              # 数据文件夹

```

- `hyperopt_results.json`: 完整的实验结果

- `hyperopt_summary.csv`: 结果汇总表### 功能特点

- ✅ **模块化设计**: 核心功能独立封装，易于维护

## 与frp_local的区别- ✅ **多算法支持**: Random Forest, XGBoost, LightGBM, 集成学习

- ✅ **专业预处理**: 针对FRP材料数据的特征工程

- frp_local: 使用7.5:2.5分割，专注于标准ML实验- ✅ **批量处理**: 支持单样本和批量预测

- frp-hyper: 使用7:2:1分割，专注于超参数优化，正确使用验证集- ✅ **命令行界面**: 完整的CLI工具

- ✅ **灵活配置**: 可配置的参数和路径

这个版本展示了如何正确使用验证集进行模型选择和超参数调优，避免在测试集上过拟合。
## 🚀 快速开始

### 1. 环境要求
```bash
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0 (可选)
lightgbm >= 3.0.0 (可选)
```

### 2. 安装依赖
```bash
pip install pandas numpy scikit-learn
pip install xgboost lightgbm  # 可选，用于高级算法
```

### 3. 运行示例
```bash
# 查看系统信息
python main.py info

# 运行完整示例
python example_usage.py

# 交互式预测
python main.py predict --model models/your_model.pkl --interactive
```

## 📊 使用指南

### 数据加载
```python
from frp_local import DataLoader

# 从CSV加载数据
loader = DataLoader("csv")
df = loader.load_data("your_data.csv")

# 数据摘要
loader.print_data_summary(df)
```

### 数据预处理
```python
from frp_local import preprocess_frp_data

# 完整预处理流程
processed_df, feature_info = preprocess_frp_data(df)
print(f"处理后数据形状: {processed_df.shape}")
```

### 模型训练
```python
from frp_local import train_frp_models

# 训练所有可用模型
results = train_frp_models(processed_df, enable_hyperparameter_tuning=True)

# 获取最佳模型
best_model = max(results.items(), key=lambda x: x[1]['test_metrics']['r2'])
print(f"最佳模型: {best_model[0]}")
```

### 预测
```python
from frp_local import FRPPredictor, create_sample_input

# 加载模型
predictor = FRPPredictor("models/best_model.pkl")

# 单样本预测
sample_input = create_sample_input()
prediction = predictor.predict_single(sample_input)
print(f"预测结果: {prediction:.4f}")

# 批量预测
predictions = predictor.predict_batch(test_df)
```

## 🎯 命令行工具

### 基本命令
```bash
# 显示系统信息
python main.py info

# 数据预处理
python main.py preprocess --input raw_data.csv --output processed_data.csv

# 模型训练
python main.py train --input processed_data.csv --tune

# 预测
python main.py predict --model models/best_model.pkl --input test_data.csv
```

### 高级用法
```bash
# 训练特定模型
python main.py train --model xgboost --tune --input data.csv

# 交互式预测
python main.py predict --model models/best_model.pkl --interactive

# 批量预测
python main.py predict --model models/best_model.pkl --input test.csv --output results.csv
```

## 📈 特征说明

### 输入特征 (12个核心特征)
1. **pH_of_condition_enviroment**: 环境pH值 (6.0-14.0)
2. **condition_time**: 暴露时间 (天)
3. **fiber_content**: 纤维含量 (%)
4. **Temperature**: 暴露温度 (°C)
5. **diameter**: 钢筋直径 (mm)
6. **concrete**: 混凝土环境 (0=否, 1=是)
7. **load_value**: 相对载荷 (0.0-1.0)
8. **Chloride_ion**: 氯离子存在 (0=否, 1=是)
9. **Glass_or_Basalt**: 纤维类型 (1=玻璃纤维, 0=玄武岩纤维)
10. **Vinyl_ester_or_Epoxy**: 基体类型 (1=乙烯基酯, 0=环氧树脂)
11. **surface_treatment**: 表面处理 (0=砂涂层, 1=光滑)
12. **max_strength**: 初始强度 (MPa)

### 输出
- **Tensile_strength_retention**: 拉伸强度保持率 (0.0-1.0)

## 🔧 配置说明

### 配置文件 (config.py)
```python
# 修改数据路径
config.DEFAULT_DATA_FILE = "path/to/your/data.csv"

# 修改模型参数
config.MODEL_PARAMS['xgboost']['n_estimators'] = 500

# 修改特征列表
config.CORE_FEATURES.append('new_feature')
```

## 📝 原始代码映射

以下是从原始`app.py`中提取的代码段对应关系：

### 数据预处理 (`preprocessor.py`)
- **原始位置**: app.py 第2619-4090行
- **核心类**: `FRPDataPreprocessor` (第3295-4090行)
- **主要功能**: 
  - `change_smd_to_nan()` - 缺失值处理
  - `parse_range_to_mean()` - 范围值解析  
  - `create_selected_features()` - 特征工程
  - `create_model_dataset()` - 模型数据集构建

### 模型训练 (`model_trainer.py`)
- **原始位置**: app.py 第171-250行 + 第2854-3294行
- **核心函数**:
  - `create_enhanced_preprocessor()` (第181-220行)
  - `diagnose_model_performance()` (第222-249行)
  - `ModelCacheManager` (第2854-3294行)

### 预测模块 (`predictor.py`)
- **原始位置**: app.py 第2624-2850行
- **核心函数**:
  - `standardize_prediction_features()` (第2624-2774行)
  - `emergency_prediction_fallback()` (第2777-2850行)

### 工具函数 (`utils.py`)
- **原始位置**: app.py 第37-170行
- **核心函数**:
  - `apply_sklearn_compatibility_patch()` (第37-62行)
  - `safe_pickle_load()` (第146-170行)

## 🆚 与原版差异

### 移除的部分 (不适合本地运行)
- ❌ **Streamlit UI组件** - 所有st.*相关代码
- ❌ **用户认证系统** - 登录/注册/权限管理
- ❌ **Web安全功能** - IP验证/邮件通知
- ❌ **数据库缓存** - 复杂的缓存表管理
- ❌ **CSS样式注入** - 界面美化代码

### 保留的部分 (核心功能)
- ✅ **数据预处理流程** - 完整的FRP数据处理
- ✅ **特征工程** - 13个核心特征构建
- ✅ **模型训练** - 多算法支持和优化
- ✅ **预测功能** - 单样本和批量预测
- ✅ **兼容性处理** - sklearn版本兼容

### 新增的部分 (本地优化)
- ✨ **命令行界面** - 完整的CLI工具
- ✨ **配置管理** - 集中化配置系统
- ✨ **文件I/O** - CSV文件读写支持
- ✨ **使用示例** - 完整的演示代码
- ✨ **错误处理** - 更强的错误恢复能力

## 🔍 故障排除

### 常见问题

1. **依赖缺失**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

2. **数据文件找不到**
```python
# 检查文件路径
from pathlib import Path
print(Path("your_data.csv").exists())
```

3. **模型加载失败**
```python
# 检查模型文件
from frp_local.utils import load_model_safely
model, info = load_model_safely("your_model.pkl")
```

4. **特征不匹配**
```bash
# 使用命令行工具检查
python main.py info
```

## 📞 技术支持

如有问题或建议，请联系开发团队或查看示例代码中的详细注释。

---
**版本**: 1.0.0  
**更新时间**: 2025年9月  
**兼容性**: Python 3.8+