# 🚀 FRP预测平台

一个基于机器学习的FRP（纤维增强聚合物）钢筋耐久性预测平台，支持多种模型算法和自定义模型导入。

## ✨ 主要功能

- 🔬 **数据管理**: 支持FRP材料数据的上传、编辑和管理
- 🤖 **模型训练**: 集成Random Forest、XGBoost、LightGBM等主流ML算法
- 📊 **预测分析**: 提供单样本和批量预测功能
- 🔧 **自定义模型**: 支持导入13种不同类型的预训练模型
- 👥 **用户管理**: 完整的用户注册、认证和权限系统
- 📈 **数据可视化**: 丰富的图表和分析报告

## 🚀 快速部署

### 方法1: Streamlit Cloud（推荐，免费）

1. **Fork此仓库到您的GitHub账号**
2. **访问 [Streamlit Cloud](https://share.streamlit.io/)**
3. **用GitHub账号登录**
4. **创建新应用**:
   - Repository: `your-username/frp-streamlit`
   - Branch: `main`
   - Main file path: `app.py`
5. **配置环境变量**（Advanced settings中）:
   ```
   DB_HOST=your-railway-host
   DB_PORT=7866
   DB_NAME=railway
   DB_USER=root
   DB_PASSWORD=your-password
   SECRET_KEY=your-secret-key
   ```
6. **点击Deploy** - 几分钟后即可全球访问！

### 方法2: 本地运行

1. **安装依赖**:
```bash
pip install -r requirements.txt
```

2. **配置数据库**:
复制 `.env.example` 到 `.env` 并填入数据库信息

3. **启动应用**:
```bash
streamlit run app.py
```

## 📊 支持的模型类型（13种）

### 🌳 树模型系列
- Random Forest、Decision Tree、Extra Trees

### 🚀 梯度提升系列  
- XGBoost、LightGBM、Gradient Boosting、AdaBoost

### 🧠 高级模型系列
- Neural Network (MLP)、Support Vector Machine、Linear Model、Ensemble Model

### 📋 基础分类
- Regression、Classification

## 🗄️ 数据库配置

支持多种数据库后端：
- ☁️ **Railway Cloud** - 云端MySQL（生产推荐）
- 🏠 **本地MySQL** - 开发环境
- 📁 **CSV导入** - 支持Excel/CSV数据

## 👥 用户权限系统

- **Admin** - 完整权限，数据管理、模型训练
- **Editor** - 数据编辑、模型配置  
- **Viewer** - 查看和预测功能

## 🧪 测试模型

项目包含10个预构建测试模型：
- `test_random_forest_model.pkl`
- `test_xgboost_model.pkl` 
- `test_neural_network_mlp_model.pkl`
- `test_preprocessor.pkl`
- 等更多...

## 📚 部署文档

- [详细部署指南](部署指南.md) 
- [自定义模型格式要求](自定义模型文件格式要求.txt)
- [快速参考指南](模型格式要求_快速参考.txt)

## 🔧 部署检查

运行部署检查脚本：
```bash
python check_deployment.py
```

## 🌍 环境变量

部署时需要配置：
```
DB_HOST=your-database-host
DB_PORT=your-database-port  
DB_NAME=railway
DB_USER=root
DB_PASSWORD=your-password
SECRET_KEY=your-secret-key
```

**🎯 5分钟内即可部署您的FRP预测平台，让全世界访问！**
