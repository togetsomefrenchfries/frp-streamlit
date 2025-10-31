# 🚀 Streamlit Cloud 部署步骤清单

## 📋 您的仓库信息
- **GitHub用户名**: `togetsomefrenchfries`
- **仓库名称**: `frp-streamlit`
- **仓库地址**: https://github.com/togetsomefrenchfries/frp-streamlit
- **分支**: `main`
- **主文件**: `app.py`

---

## 🚀 部署步骤（总共6步，约5分钟）

### 第1步：访问 Streamlit Cloud 🌐
1. 打开浏览器，访问：**https://share.streamlit.io/**
2. 如果还没有账号，用您的GitHub账号登录（推荐）

### 第2步：创建新应用 ➕
1. 登录后，点击右上角的 **"New app"** 按钮
2. 选择 **"From existing repo"**

### 第3步：配置仓库信息 📁
在应用配置页面，填入以下信息：

**Repository（仓库）**:
```
togetsomefrenchfries/frp-streamlit
```

**Branch（分支）**:
```
main
```

**Main file path（主文件路径）**:
```
app.py
```

**App URL（应用网址，可自定义）**:
```
frp-prediction（或您喜欢的名字）
```

### 第4步：配置环境变量 🔧
点击 **"Advanced settings"** 展开高级设置，在 **"Environment variables"** 部分添加：

```
DB_HOST=containers-us-west-xxx.railway.app
DB_PORT=7866
DB_NAME=railway
DB_USER=root
DB_PASSWORD=您的Railway数据库密码
SECRET_KEY=frp-prediction-secret-2025
```

> ⚠️ 重要：请替换为您实际的Railway数据库信息！

### 第5步：部署应用 🚀
1. 检查所有配置信息无误
2. 点击 **"Deploy!"** 按钮
3. 等待部署完成（通常2-5分钟）

### 第6步：测试应用 ✅
部署成功后，您将获得一个网址，类似：
```
https://frp-prediction.streamlit.app/
```

立即测试以下功能：
- [ ] 网站可以正常访问
- [ ] 用户注册/登录功能
- [ ] 数据加载正常
- [ ] 模型训练功能
- [ ] 自定义模型上传
- [ ] 预测功能

---

## 🔧 如果需要Railway数据库信息

### 查看Railway数据库连接信息：
1. 访问 https://railway.app/
2. 进入您的数据库项目
3. 点击 **"Variables"** 标签页
4. 复制以下信息：
   - `MYSQLHOST` → DB_HOST
   - `MYSQLPORT` → DB_PORT  
   - `MYSQLDATABASE` → DB_NAME
   - `MYSQLUSER` → DB_USER
   - `MYSQLPASSWORD` → DB_PASSWORD

---

## 🎯 部署后的网址示例

根据您选择的应用名称，您的网址可能是：
- https://frp-prediction.streamlit.app/
- https://togetsomefrenchfries-frp-streamlit.streamlit.app/
- 或其他您自定义的名称

---

## 🚨 常见问题解决

### 问题1：部署失败
**解决方案**：
- 检查requirements.txt是否包含所有依赖
- 确认app.py在仓库根目录
- 检查环境变量是否正确设置

### 问题2：数据库连接失败
**解决方案**：
- 确认Railway数据库正在运行
- 检查数据库连接信息是否正确
- 确认网络策略允许外部连接

### 问题3：应用启动慢
**解决方案**：
- 首次启动需要安装依赖，约2-3分钟是正常的
- 后续访问会快很多

---

## 🎉 部署成功后您将获得

✅ **全球访问的网址** - 任何人都可以通过网址访问您的FRP预测平台
✅ **自动HTTPS加密** - 安全的数据传输
✅ **自动更新** - 当您更新GitHub代码时，应用会自动重新部署
✅ **免费托管** - 无需服务器维护成本
✅ **使用统计** - Streamlit Cloud提供访问统计

---

## 📱 分享您的应用

部署成功后，您可以：
1. **发送链接**给同事和研究人员
2. **在学术论文**中引用您的预测平台
3. **在社交媒体**上分享您的研究成果
4. **生成QR码**方便手机用户访问

---

## 🔄 后续更新流程

当您需要更新应用时：
1. 在本地修改代码
2. 提交并推送到GitHub：
   ```bash
   git add .
   git commit -m "更新说明"
   git push origin main
   ```
3. Streamlit Cloud会自动检测更新并重新部署

---

**🎯 现在就开始部署吧！5分钟后，全世界都能访问您的FRP预测平台！**