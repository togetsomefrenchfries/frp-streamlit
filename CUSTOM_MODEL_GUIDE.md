# 🤖 自定义模型导入功能使用指南

## 📋 功能概述

FRP预测平台现在支持导入您自己训练的机器学习模型！这个功能允许您：

- ✅ 上传预训练的pickle格式模型文件
- ✅ 可选择上传预处理器文件
- ✅ 配置模型的特征要求
- ✅ 直接使用自定义模型进行预测
- ✅ 管理多个自定义模型

## 🚀 使用步骤

### 第1步：准备模型文件

您的模型必须满足以下要求：
- **格式**：pickle (.pkl) 文件
- **接口**：模型必须有 `predict()` 方法
- **兼容性**：建议使用scikit-learn、XGBoost或LightGBM模型

**支持的模型类型：**
```python
# ✅ 支持的模型示例
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 模型必须实现predict方法
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 保存模型
import pickle
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 第2步：在平台中导入模型

1. **登录应用** - 使用admin账号登录
2. **进入Model Configuration页面**
3. **启用自定义模型模式**：
   - 勾选 "Use Custom Pre-trained Model"
4. **上传文件**：
   - 上传模型文件 (.pkl格式)
   - 可选：上传预处理器文件
5. **配置模型信息**：
   - 输入模型名称
   - 选择模型类型（Regression/Classification）
   - 配置期望特征列表

### 第3步：配置特征

在"Expected Feature Names"文本框中，每行输入一个特征名称：

```
pH
Fiber_content_wt
Diameter_mm
Temp_C
Duration_hours
```

### 第4步：验证和保存

1. 点击 **"Validate and Save Custom Model"**
2. 系统会验证模型文件的有效性
3. 成功后模型将保存到会话中

### 第5步：使用模型进行预测

1. **在Model Training页面**：
   - 点击"Start Training"
   - 系统检测到自定义模型后会跳过训练过程
   - 直接准备模型用于预测

2. **在Predictions页面**：
   - 选择您的自定义模型
   - 系统会自动生成特征输入界面
   - 输入特征值进行预测

## 📝 示例：使用测试模型

平台提供了一个测试模型供您体验功能：

### 测试文件
- `test_frp_model.pkl` - 测试模型文件
- `test_frp_preprocessor.pkl` - 预处理器文件（可选）
- `model_features.txt` - 特征说明文档

### 测试步骤
1. 在Model Configuration中上传 `test_frp_model.pkl`
2. 模型名称：`Test FRP Model`
3. 模型类型：`Regression`
4. 期望特征：
   ```
   pH
   Fiber_content_wt
   Diameter_mm
   Temp_C
   Duration_hours
   ```
5. 保存模型后，在Predictions页面测试

### 测试输入示例
- **pH**: 7.0
- **Fiber_content_wt**: 0.5
- **Diameter_mm**: 12.0
- **Temp_C**: 25.0
- **Duration_hours**: 100.0

预期预测结果约为 **80.66**

## ⚠️ 注意事项

### 模型要求
- 模型必须是已训练的（fitted）状态
- 必须支持pandas DataFrame输入
- 建议模型具有良好的泛化能力

### 特征匹配
- 确保特征名称与训练时一致
- 特征顺序必须正确
- 数据类型需要匹配

### 数据预处理
- 如果训练时使用了预处理器，建议一同上传
- 预处理器将自动应用于输入数据
- 支持sklearn的StandardScaler、MinMaxScaler等

### 安全提醒
- 只上传来源可信的模型文件
- 模型文件可能包含执行代码，请确保安全

## 🔧 高级功能

### 批量预测
支持上传CSV文件进行批量预测：
1. 在Predictions页面选择自定义模型
2. 上传包含特征数据的CSV文件
3. 系统自动处理并返回预测结果
4. 可下载完整的预测结果

### 模型管理
- 支持同时管理多个自定义模型
- 可以随时切换使用不同的模型
- 可以删除不需要的模型

### 预处理器支持
- 支持上传配套的预处理器
- 自动应用数据标准化/归一化
- 支持特征工程流水线

## 🐛 故障排除

### 常见错误

1. **"Invalid model: Model must have a 'predict' method"**
   - 确保模型已正确训练
   - 检查模型类是否实现predict方法

2. **"Failed to load model: pickle load error"**
   - 检查文件是否损坏
   - 确认pickle版本兼容性
   - 尝试重新保存模型文件

3. **预测失败**
   - 检查输入特征名称是否匹配
   - 确认数据类型正确
   - 验证特征数量是否一致

### 调试建议
- 使用提供的测试模型验证功能
- 检查模型训练时的特征列表
- 确认预处理步骤的一致性

## 📞 技术支持

如果遇到问题：
1. 检查应用日志中的错误信息
2. 验证模型文件的完整性
3. 确认特征配置的正确性
4. 使用测试模型进行功能验证

---

**🎉 现在您可以充分利用自己训练的模型来进行FRP耐久性预测了！**