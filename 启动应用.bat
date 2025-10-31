@echo off
chcp 65001
echo ========================================
echo    FRP钢筋耐久性预测系统 - 启动脚本
echo ========================================
echo.

echo 🔍 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

echo.
echo 📦 检查虚拟环境...
if exist ".venv\Scripts\activate.bat" (
    echo ✅ 发现虚拟环境，正在激活...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo ✅ 发现虚拟环境，正在激活...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  未发现虚拟环境，使用系统Python
)

echo.
echo 📋 选择要启动的应用:
echo 1. 主预测应用 (app.py)
echo 2. 平台管理系统 (platform code.py)
echo 3. 数据导入工具 (dataset code.py)
echo 4. 数据分析应用 (app_dataset_relationship_analysis.py)
echo 5. 安装/更新依赖
echo 6. 退出

set /p choice=请输入选择 (1-6): 

if "%choice%"=="1" (
    echo.
    echo 🚀 启动主预测应用...
    echo 访问地址: http://localhost:8501
    streamlit run app.py
) else if "%choice%"=="2" (
    echo.
    echo 🚀 启动平台管理系统...
    echo 访问地址: http://localhost:8502
    streamlit run "platform code.py" --server.port 8502
) else if "%choice%"=="3" (
    echo.
    echo 🚀 启动数据导入工具...
    python "dataset code（excel to SQL）.py"
    pause
) else if "%choice%"=="4" (
    echo.
    echo 🚀 启动数据分析应用...
    echo 访问地址: http://localhost:8503
    streamlit run app_dataset_relationship_analysis.py --server.port 8503
) else if "%choice%"=="5" (
    echo.
    echo 📦 安装/更新依赖...
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    if %errorlevel% equ 0 (
        echo ✅ 依赖安装完成
    ) else (
        echo ❌ 依赖安装失败
    )
    pause
) else if "%choice%"=="6" (
    echo 👋 再见!
    exit /b 0
) else (
    echo ❌ 无效选择
    pause
)

echo.
echo 按任意键返回主菜单...
pause >nul
goto :start