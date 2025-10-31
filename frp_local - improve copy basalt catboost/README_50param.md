# 50参数超参数优化实验

这个项目包含了针对FRP数据的大规模超参数优化实验，使用3种机器学习模型各50个配置进行5折交叉验证。

## 📁 文件结构

```
frp_local/
├── run_50param_experiments.py  # 主实验脚本
├── start_experiment.py         # 实验启动脚本  
├── test_50param.py             # 功能测试脚本
├── analyze_results.py          # 结果分析脚本
├── data_loader.py              # 数据加载模块
├── preprocessor.py             # 数据预处理模块
├── config.py                   # 配置文件
└── results/                    # 实验结果目录 (自动创建)
```

## 🚀 快速开始

### 1. 运行功能测试
首先验证所有功能是否正常：
```bash
python test_50param.py
```

### 2. 启动完整实验
```bash
python start_experiment.py
```
或者直接运行：
```bash
python run_50param_experiments.py
```

### 3. 分析实验结果
```bash
python analyze_results.py
```

## 📊 实验配置

### 模型与参数
- **RandomForest**: 50个配置
  - n_estimators: [50, 100, 150, 200, 250]
  - max_depth: [3, 5, 10, 15, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
  - max_features: ['sqrt', 'log2']

- **XGBoost**: 50个配置  
  - n_estimators: [50, 100, 150, 200]
  - max_depth: [3, 4, 5, 6]
  - learning_rate: [0.01, 0.03, 0.05, 0.1]
  - subsample: [0.6, 0.7, 0.8, 0.9]
  - colsample_bytree: [0.6, 0.7, 0.8, 0.9]

- **LightGBM**: 50个配置
  - n_estimators: [50, 100, 150, 200]
  - max_depth: [3, 4, 5, 6]  
  - learning_rate: [0.01, 0.03, 0.05, 0.1]
  - num_leaves: [15, 31, 50, 100]
  - subsample: [0.6, 0.7, 0.8, 0.9]

### 数据配置
- **数据分割**: 8:2 (训练:测试)
- **交叉验证**: 5折CV
- **目标变量**: Tensile strength retention
- **特征数量**: ~8个 (移除完全缺失的特征)
- **有效样本**: ~2720个

## 📈 实验流程

1. **数据加载与预处理**
   - 加载Excel数据 
   - 应用FRPDataPreprocessor进行预处理
   - 移除完全缺失的特征列
   - 过滤缺失值

2. **参数网格生成**
   - 每个模型生成50个不同的参数配置
   - 使用combinatorial和random sampling

3. **模型训练与评估**
   - 5折交叉验证
   - 计算CV均值和标准差
   - 在测试集上评估
   - 记录训练时间

4. **结果保存**
   - 每5个配置保存一次中间结果
   - JSON格式保存实验摘要
   - CSV格式保存详细结果

## 📁 输出文件

### results/ 目录包含：
- `experiment_summary_YYYYMMDD_HHMMSS.json` - 实验摘要
- `detailed_results_YYYYMMDD_HHMMSS.csv` - 详细结果
- `RandomForest_results_YYYYMMDD_HHMMSS.json` - RandomForest结果  
- `XGBoost_results_YYYYMMDD_HHMMSS.json` - XGBoost结果
- `LightGBM_results_YYYYMMDD_HHMMSS.json` - LightGBM结果

### 结果格式示例：
```json
{
  "best_models": {
    "RandomForest": {
      "config": {"n_estimators": 200, "max_depth": 10, ...},
      "cv_mean": 0.5234,
      "cv_std": 0.0156,
      "test_r2": 0.5456
    }
  },
  "model_statistics": {
    "RandomForest": {
      "best_cv_score": 0.5234,
      "mean_cv_score": 0.4523,
      "total_configs": 50
    }
  }
}
```

## ⏱️ 预计运行时间

- **测试脚本**: ~2分钟
- **完整实验**: 30-60分钟 (取决于硬件)
- **单个配置**: ~10-30秒

## 🔧 自定义配置

### 修改参数网格
编辑 `run_50param_experiments.py` 中的 `generate_parameter_grids()` 方法：

```python
def generate_parameter_grids(self):
    # 修改参数范围
    rf_base_grid = {
        'n_estimators': [50, 100, 200],  # 自定义值
        'max_depth': [5, 10, None],      # 自定义值
        # ...
    }
```

### 修改保存频率
修改 `SAVE_EVERY` 常量：
```python
SAVE_EVERY = 10  # 每10个配置保存一次
```

### 修改交叉验证折数
修改 `cv_folds` 参数：
```python
cv_folds = 3  # 使用3折CV加速实验
```

## 🛠️ 故障排除

### 常见问题

1. **数据加载失败**
   - 检查 `database 4.xlsx` 文件是否存在
   - 确保文件路径正确

2. **内存不足**
   - 减少参数网格大小
   - 降低交叉验证折数

3. **实验中断**
   - 中间结果已保存在results/目录
   - 可以查看已完成的配置

4. **特征完全缺失**
   - 实验会自动移除完全缺失的特征
   - 检查preprocessor.py的预处理逻辑

### 调试模式
运行小规模测试：
```bash
python test_50param.py
```

查看详细日志，在脚本中添加：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 使用建议

1. **首次使用**: 先运行 `test_50param.py` 确保所有功能正常
2. **长时间实验**: 使用 `start_experiment.py` 获得更好的用户体验
3. **结果分析**: 实验完成后运行 `analyze_results.py` 获得详细分析
4. **参数调优**: 根据结果分析调整参数网格进行进一步实验

## 🔍 下一步

- 根据最佳配置进行模型集成
- 分析特征重要性
- 进行模型解释和可视化
- 部署最佳模型到生产环境