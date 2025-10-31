@echo off
chcp 65001
echo ========================================
echo   FRP钢筋耐久性预测系统 - 环境设置
echo ========================================
echo.

echo 🔧 开始环境设置...

echo.
echo 1️⃣ 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% equ 0 (
        echo ✅ 虚拟环境创建成功
    ) else (
        echo ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
) else (
    echo ✅ 虚拟环境已存在
)

echo.
echo 2️⃣ 激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo 3️⃣ 升级pip...
python -m pip install --upgrade pip

echo.
echo 4️⃣ 安装项目依赖...
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if %errorlevel% equ 0 (
    echo ✅ 依赖安装完成
) else (
    echo ❌ 依赖安装失败，请检查网络连接
    pause
    exit /b 1
)

echo.
echo 5️⃣ 创建环境配置文件...
if not exist ".env" (
    copy ".env.example" ".env"
    echo ✅ 已创建 .env 文件，请根据需要修改配置
    echo ⚠️  请特别注意修改数据库密码和文件路径
) else (
    echo ✅ .env 文件已存在
)

echo.
echo 6️⃣ 创建必要目录...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "temp" mkdir temp
echo ✅ 目录创建完成

echo.
echo 🎉 环境设置完成！
echo.
echo 📋 下一步操作:
echo 1. 编辑 .env 文件，配置数据库连接信息
echo 2. 确保MySQL数据库已安装并运行
echo 3. 准备Excel数据文件
echo 4. 运行 "启动应用.bat" 开始使用

echo.
echo ❓ 是否现在打开配置文件进行编辑? (y/N)
set /p edit_config=
if /i "%edit_config%"=="y" (
    notepad .env
)

echo.
echo 按任意键退出...
pause >nul