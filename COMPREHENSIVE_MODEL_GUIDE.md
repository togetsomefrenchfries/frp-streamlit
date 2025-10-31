# FRP 测试模型集合使用指南

## 可用模型类型

以下是已创建的测试模型及其特点：

### Random Forest
- **描述**: 随机森林 - 集成学习，适用于大多数回归任务
- **文件**: `test_random_forest_model.pkl`
- **预处理**:  (可选预处理器)
- **测试预测**: 2.08

### Decision Tree
- **描述**: 决策树 - 简单易理解的树模型
- **文件**: `test_decision_tree_model.pkl`
- **预处理**:  (可选预处理器)
- **测试预测**: 2.43

### Extra Trees
- **描述**: 极端随机树 - 随机森林的变种
- **文件**: `test_extra_trees_model.pkl`
- **预处理**:  (可选预处理器)
- **测试预测**: 2.43

### Gradient Boosting
- **描述**: 梯度提升 - scikit-learn的梯度提升实现
- **文件**: `test_gradient_boosting_model.pkl`
- **预处理**:  (可选预处理器)
- **测试预测**: 2.38

### AdaBoost
- **描述**: 自适应提升 - 经典的提升算法
- **文件**: `test_adaboost_model.pkl`
- **预处理**:  (可选预处理器)
- **测试预测**: 1.50

### Neural Network (MLP)
- **描述**: 多层感知器 - 深度神经网络
- **文件**: `test_neural_network_mlp_model.pkl`
- **预处理**:  (需要预处理器)
- **测试预测**: 2.33

### Support Vector Machine
- **描述**: 支持向量机 - 核方法回归
- **文件**: `test_support_vector_machine_model.pkl`
- **预处理**:  (需要预处理器)
- **测试预测**: 2.34

### Linear Model
- **描述**: 线性回归 - 最基础的线性模型
- **文件**: `test_linear_model_model.pkl`
- **预处理**:  (需要预处理器)
- **测试预测**: 2.31

### XGBoost
- **描述**: XGBoost梯度提升 - 高性能梯度提升框架
- **文件**: `test_xgboost_model.pkl`
- **预处理**:  (可选预处理器)
- **测试预测**: 2.43

### LightGBM
- **描述**: LightGBM梯度提升 - 微软开源的高效梯度提升
- **文件**: `test_lightgbm_model.pkl`
- **预处理**:  (可选预处理器)
- **测试预测**: 2.43


## 输入特征

所有模型期望以下5个特征（按顺序）：

1. **pH** - 环境pH值 (0-14)
2. **Fiber_content_wt** - 纤维含量重量比 (0-1)
3. **Diameter_mm** - 直径(毫米) (1-50)
4. **Temp_C** - 温度(摄氏度) (-50-200)
5. **Duration_hours** - 持续时间(小时) (0-100000)


## 在Streamlit中使用步骤

1. **进入Model Configuration页面**
2. **启用自定义模型**
   - 勾选 'Use Custom Pre-trained Model'
3. **上传模型文件**
   - 选择对应的 `.pkl` 模型文件
   - 对于需要预处理的模型，同时上传 `test_preprocessor.pkl`
4. **配置模型信息**
   - 输入模型名称
   - 选择对应的模型类型（现在支持13种类型！）
5. **设置特征**
   - 在特征配置区域输入以下特征（每行一个）：
     ```
     pH
     Fiber_content_wt
     Diameter_mm
     Temp_C
     Duration_hours
     ```
6. **验证并保存**
   - 点击 'Validate and Save Custom Model'
7. **使用预测**
   - 在 Predictions 标签页中选择自定义模型进行预测

## 支持的模型类型

现在平台支持以下13种模型类型：

### 基础分类
- **Regression** - 通用回归模型
- **Classification** - 通用分类模型

### 树模型
- **Random Forest** - 随机森林
- **Decision Tree** - 决策树  
- **Extra Trees** - 极端随机树

### 梯度提升
- **XGBoost** - XGBoost梯度提升
- **LightGBM** - LightGBM梯度提升
- **Gradient Boosting** - scikit-learn梯度提升
- **AdaBoost** - 自适应提升

### 高级模型
- **Neural Network (MLP)** - 多层感知器神经网络
- **Support Vector Machine** - 支持向量机
- **Linear Model** - 线性模型
- **Ensemble Model** - 集成模型

## 使用建议

### 对于树模型：
- 无需预处理器，但上传也不会出错
- 适合处理非线性关系和特征交互
- 训练速度快，解释性好

### 对于神经网络模型：
- **必须**上传预处理器进行数据标准化
- 适合复杂的非线性模式学习
- 需要更多的训练数据

### 对于线性模型：
- 建议使用预处理器
- 适合线性关系明显的数据
- 计算速度最快

## 注意事项

1. 选择正确的模型类型很重要，这会影响预测界面的显示
2. 神经网络和SVM模型强烈建议使用预处理器
3. 所有模型文件都必须是pickle格式(.pkl)
4. 特征名称和顺序必须与训练时一致
5. 可以上传多个不同类型的模型进行比较

