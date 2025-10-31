#!/usr/bin/env python3
"""
FRP本地预测系统安装脚本
自动检查依赖并运行测试
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version}")
    return True

def install_requirements():
    """安装依赖包"""
    print("\n📦 安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def check_core_imports():
    """检查核心模块导入"""
    print("\n🔍 检查核心模块...")
    core_modules = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
    ]
    
    for module_name, alias in core_modules:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                importlib.import_module(module_name)
            print(f"✅ {module_name} 导入成功")
        except ImportError as e:
            print(f"❌ {module_name} 导入失败: {e}")
            return False
    
    return True

def check_optional_imports():
    """检查可选模块导入"""
    print("\n🔧 检查可选模块...")
    optional_modules = ['xgboost', 'lightgbm', 'matplotlib', 'seaborn']
    
    available = []
    for module_name in optional_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} 可用")
            available.append(module_name)
        except ImportError:
            print(f"⚠️  {module_name} 不可用 (可选)")
    
    return available

def test_local_modules():
    """测试本地模块导入"""
    print("\n🏠 测试本地模块...")
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, str(Path.cwd()))
        
        from frp_local import config
        print("✅ config 模块导入成功")
        
        from frp_local import DataLoader
        print("✅ DataLoader 导入成功")
        
        from frp_local import FRPDataPreprocessor
        print("✅ FRPDataPreprocessor 导入成功")
        
        from frp_local import ModelTrainer
        print("✅ ModelTrainer 导入成功")
        
        from frp_local import FRPPredictor
        print("✅ FRPPredictor 导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 本地模块导入失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")
    directories = ['data', 'models', 'outputs']
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {dir_name}")
        else:
            print(f"ℹ️  目录已存在: {dir_name}")

def run_basic_test():
    """运行基础功能测试"""
    print("\n🧪 运行基础测试...")
    try:
        # 测试配置加载
        from frp_local import config
        assert len(config.CORE_FEATURES) > 0, "核心特征列表为空"
        print("✅ 配置测试通过")
        
        # 测试数据加载器
        from frp_local import DataLoader
        loader = DataLoader("csv")
        print("✅ 数据加载器测试通过")
        
        # 测试预处理器
        from frp_local import FRPDataPreprocessor
        preprocessor = FRPDataPreprocessor()
        print("✅ 预处理器测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")
        return False

def main():
    """主安装流程"""
    print("🚀 FRP本地预测系统 - 安装向导")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        print("\n⚠️  依赖安装失败，尝试手动安装:")
        print("pip install pandas numpy scikit-learn xgboost lightgbm")
        sys.exit(1)
    
    # 检查导入
    if not check_core_imports():
        sys.exit(1)
    
    # 检查可选模块
    available_optional = check_optional_imports()
    
    # 创建目录
    create_directories()
    
    # 测试本地模块
    if not test_local_modules():
        sys.exit(1)
    
    # 运行基础测试
    if not run_basic_test():
        sys.exit(1)
    
    # 安装完成
    print("\n" + "=" * 50)
    print("🎉 安装完成!")
    print("\n📚 快速开始:")
    print("   python example_usage.py")
    print("   python main.py info")
    print("   python main.py predict --interactive")
    print("\n💡 更多信息请查看 README.md")
    
    if available_optional:
        print(f"\n🔧 可用的高级功能: {', '.join(available_optional)}")

if __name__ == "__main__":
    main()