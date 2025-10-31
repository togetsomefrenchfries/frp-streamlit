"""
Platform Code.py 代码组织结构映射
=====================================

这个文档详细说明了 platform code.py 中数据读取和预处理代码的位置分布

文件总体信息：
- 总行数：12156行
- 主要功能：企业级FRP钢筋耐久性预测平台
- 包含：数据读取、预处理、机器学习、用户管理、缓存等功能
"""

# ========================================
# 1. 数据读取相关代码 (Data Reading)
# ========================================

class DataReadingCodeMap:
    """数据读取代码分布"""
    
    # 主要数据读取函数
    main_data_loading = {
        "function": "load_default_data()",
        "line": 3876,
        "decorator": "@st.cache_data",
        "description": "主要数据读取函数，带缓存装饰器",
        "sql_query": "SELECT * FROM research_data",
        "sql_line": 3895,
        "return_type": "pandas DataFrame"
    }
    
    # 其他数据读取相关函数
    other_reading_functions = [
        {
            "pattern": "pd.read_sql",
            "locations": [3895],  # 基于之前的搜索结果
            "purpose": "从MySQL数据库读取research_data表"
        },
        {
            "pattern": "engine.connect()",
            "purpose": "数据库连接管理"
        },
        {
            "pattern": "SELECT.*FROM",
            "purpose": "SQL查询语句"
        }
    ]
    
    # 缓存相关的数据读取
    cached_functions = [
        {
            "decorator": "@st.cache_data",
            "line": 3876,
            "function": "load_default_data()",
            "purpose": "缓存主要数据读取，提高性能"
        }
    ]

# ========================================
# 2. 预处理代码分布 (Preprocessing)
# ========================================

class PreprocessingCodeMap:
    """预处理代码分布映射"""
    
    # 主要预处理类 - 发现了3个重复的FRPDataPreprocessor类！
    frp_preprocessor_classes = [
        {
            "class_name": "FRPDataPreprocessor",
            "line": 3185,
            "version": "第一版本",
            "description": "主要的FRP数据预处理类"
        },
        {
            "class_name": "FRPDataPreprocessor", 
            "line": 7462,
            "version": "第二版本（重复）",
            "description": "重复的预处理类定义"
        },
        {
            "class_name": "FRPDataPreprocessor",
            "line": 7758, 
            "version": "第三版本（重复）",
            "description": "又一个重复的预处理类定义"
        }
    ]
    
    # 第一版本 FRPDataPreprocessor (Line 3185) 的主要方法
    first_version_methods = {
        "create_selected_features": {
            "line": 3447,
            "purpose": "创建选定的13个特征",
            "description": "主要的特征工程入口函数"
        },
        
        # 具体的特征处理方法
        "feature_processing_methods": [
            {
                "method": "_process_ph_and_chloride",
                "line": 3519,
                "features": "pH值和氯离子浓度处理"
            },
            {
                "method": "_process_concrete_indicator", 
                "line": 3627,
                "features": "混凝土指标处理"
            },
            {
                "method": "_process_diameter",
                "line": 3642, 
                "features": "直径相关特征处理"
            },
            {
                "method": "_process_load",
                "line": 3658,
                "features": "荷载相关特征处理" 
            },
            {
                "method": "_process_fiber_content",
                "line": 3690,
                "features": "纤维含量特征处理"
            },
            {
                "method": "_process_material_types",
                "line": 3729,
                "features": "材料类型特征处理"
            },
            {
                "method": "_process_surface_treatment",
                "line": 3747,
                "features": "表面处理特征处理"
            },
            {
                "method": "_process_other_features",
                "line": 3756,
                "features": "其他特征处理"
            }
        ]
    }
    
    # 第二版本的预处理方法
    second_version_methods = {
        "preprocess": {
            "line": 7598,
            "purpose": "数据预处理主函数",
            "description": "第二版本的预处理入口"
        }
    }
    
    # 第三版本的预处理方法  
    third_version_methods = {
        "create_selected_features": {
            "line": 7819,
            "purpose": "创建选定特征（重复版本）",
            "description": "第三版本的特征创建方法"
        }
    }
    
    # 其他预处理相关函数
    other_preprocessing_functions = [
        {
            "function": "create_enhanced_preprocessor",
            "line": 47,
            "purpose": "创建增强的预处理器",
            "parameters": "categorical_cols, numeric_cols, add_polynomial, polynomial_degree"
        },
        {
            "function": "standardize_prediction_features", 
            "line": 2516,
            "purpose": "标准化预测特征",
            "description": "用于预测时的特征标准化"
        }
    ]

# ========================================
# 3. 代码重复性分析
# ========================================

class CodeDuplicationAnalysis:
    """代码重复性分析"""
    
    major_duplications = [
        {
            "component": "FRPDataPreprocessor类",
            "instances": 3,
            "lines": [3185, 7462, 7758],
            "duplication_level": "严重重复",
            "impact": "维护困难，容易出现不一致"
        },
        {
            "component": "create_selected_features方法",
            "instances": 2, 
            "lines": [3447, 7819],
            "duplication_level": "重复",
            "impact": "特征工程逻辑重复"
        }
    ]
    
    recommendation = """
    建议重构：
    1. 合并重复的FRPDataPreprocessor类定义
    2. 统一特征工程接口
    3. 采用配置驱动的方式区分不同版本的处理逻辑
    4. 建立清晰的代码模块化结构
    """

# ========================================
# 4. 使用指南
# ========================================

class UsageGuide:
    """代码使用指南"""
    
    data_reading_workflow = """
    数据读取流程：
    1. 调用 load_default_data() (第3876行)
    2. 该函数使用 @st.cache_data 缓存结果
    3. 内部执行 pd.read_sql("SELECT * FROM research_data", engine) (第3895行)
    4. 返回完整的research_data表数据
    """
    
    preprocessing_workflow = """
    预处理流程：
    1. 实例化 FRPDataPreprocessor (推荐使用第3185行的版本)
    2. 调用 create_selected_features(df) (第3447行)
    3. 该方法内部调用各种 _process_* 方法处理不同特征
    4. 最终返回13个工程特征的DataFrame
    """
    
    feature_processing_details = """
    特征处理细节：
    - pH和氯离子：_process_ph_and_chloride (第3519行)
    - 混凝土指标：_process_concrete_indicator (第3627行)  
    - 直径特征：_process_diameter (第3642行)
    - 荷载特征：_process_load (第3658行)
    - 纤维含量：_process_fiber_content (第3690行)
    - 材料类型：_process_material_types (第3729行)
    - 表面处理：_process_surface_treatment (第3747行)
    - 其他特征：_process_other_features (第3756行)
    """

# ========================================
# 5. 关键发现总结
# ========================================

print("Platform Code.py 代码组织关键发现：")
print("="*50)
print("📊 数据读取代码：")
print("   - 主函数：load_default_data() (第3876行)")
print("   - SQL查询：第3895行")
print("   - 缓存机制：@st.cache_data装饰器")
print()
print("🔧 预处理代码：")
print("   - 主类：FRPDataPreprocessor (第3185行) - 推荐使用")
print("   - 重复类：第7462行和第7758行 - 需要清理")
print("   - 特征工程：create_selected_features (第3447行)")
print("   - 8个专门的_process_*方法处理不同特征类型")
print()
print("⚠️  代码重复问题：")
print("   - FRPDataPreprocessor类重复定义3次")
print("   - 严重影响代码维护性")
print("   - 建议重构统一接口")
print()
print("💡 推荐使用路径：")
print("   - 数据读取：第3876行的load_default_data()")
print("   - 预处理：第3185行的FRPDataPreprocessor类")
print("   - 特征工程：第3447行的create_selected_features()")