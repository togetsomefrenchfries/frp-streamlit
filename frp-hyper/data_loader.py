# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 数据加载模块
Data Loading Module for FRP Rebar Durability Prediction

支持从CSV文件和数据库加载数据
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings

# 可选的数据库支持
try:
    import pymysql
    from sqlalchemy import create_engine, text
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Warning: Database dependencies not available. CSV loading only.")

try:
    from .config import config
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from config import config
    except ImportError:
        print("Warning: Could not import config module")

class DataLoader:
    """数据加载器 - 支持多种数据源"""
    
    def __init__(self, data_source: str = "csv"):
        """
        初始化数据加载器
        
        Args:
            data_source: 数据源类型 ('csv' 或 'database')
        """
        self.data_source = data_source
        self.engine = None
        
        if data_source == "database" and DATABASE_AVAILABLE:
            self._init_database_connection()
    
    def _init_database_connection(self):
        """初始化数据库连接"""
        try:
            db_config = config.DATABASE_CONFIG
            connection_string = (
                f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("✅ Database connection established")
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.engine = None
    
    def load_data(self, 
                  source_path: Optional[str] = None,
                  table_name: str = "research_data",
                  **kwargs) -> Optional[pd.DataFrame]:
        """
        加载数据
        
        Args:
            source_path: CSV文件路径（当使用CSV时）
            table_name: 数据库表名（当使用数据库时）
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        
        if self.data_source == "csv":
            return self._load_from_csv(source_path, **kwargs)
        elif self.data_source == "database":
            return self._load_from_database(table_name, **kwargs)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
    
    def _load_from_csv(self, file_path: Optional[str] = None, **kwargs) -> Optional[pd.DataFrame]:
        """从CSV或Excel文件加载数据"""
        
        if file_path is None:
            file_path = config.DEFAULT_DATA_FILE
        
        try:
            if not os.path.exists(file_path):
                print(f"❌ Data file not found: {file_path}")
                return None
            
            # 检测文件类型
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.xlsx' or file_ext == '.xls':
                # 读取Excel文件，跳过前3行标题
                print(f"Loading Excel file: {file_path}")
                df = pd.read_excel(file_path, skiprows=3, **kwargs)
                file_type = "Excel"
            elif file_ext == '.csv':
                # 读取CSV文件
                print(f"Loading CSV file: {file_path}")
                df = pd.read_csv(file_path, **kwargs)
                file_type = "CSV"
            else:
                print(f"Unsupported file format: {file_ext}")
                return None
            
            print(f"Successfully loaded data from {file_type}: {df.shape}")
            print(f"   Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            
            # 应用数据过滤和清理
            cleaned_df = self._apply_data_filtering(df)
            
            return self._basic_data_cleaning(cleaned_df)
            
        except Exception as e:
            print(f"Failed to load data file: {e}")
            return None
    
    def _load_from_database(self, table_name: str = "research_data", **kwargs) -> Optional[pd.DataFrame]:
        """从数据库加载数据"""
        
        if not DATABASE_AVAILABLE:
            print("❌ Database functionality not available")
            return None
            
        if self.engine is None:
            print("❌ Database connection not established")
            return None
        
        try:
            # 检查表是否存在
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'")).fetchone()
                if not result:
                    print(f"❌ Table '{table_name}' does not exist")
                    return None
            
            # 查询数据
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine, **kwargs)
            
            print(f"✅ Successfully loaded data from database: {df.shape}")
            print(f"   Table: {table_name}")
            print(f"   Columns: {list(df.columns)}")
            
            return self._basic_data_cleaning(df)
            
        except Exception as e:
            print(f"❌ Failed to load from database: {e}")
            return None
    
    def _apply_data_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用数据过滤规则"""
        
        print("Applying data filtering rules...")
        
        # 记录原始形状
        original_shape = df.shape
        
        # 检查第一列是否存在
        if df.empty or df.shape[1] == 0:
            print("Warning: No data to filter")
            return df
        
        first_col = df.iloc[:, 0]
        print(f"   First column name: {df.columns[0]}")
        print(f"   First column data type: {first_col.dtype}")
        
        # 分析第一列的值分布
        print("   First column value distribution:")
        try:
            value_counts = first_col.value_counts()
            print(f"     {dict(value_counts.head(10))}")
        except Exception as e:
            print(f"     Error analyzing distribution: {e}")
        
        # 转换第一列为数值类型以便过滤
        numeric_first_col = pd.to_numeric(first_col, errors='coerce')
        
        # 过滤规则：只保留第一列值为1的行
        valid_mask = (numeric_first_col == 1)
        filtered_df = df[valid_mask].copy()
        
        # 统计过滤结果
        total_rows = len(df)
        valid_rows = valid_mask.sum()
        invalid_rows = len(df) - valid_rows
        
        print(f"   Total rows: {total_rows}")
        print(f"   Valid rows (first_col=1): {valid_rows}")
        print(f"   Filtered out rows (first_col≠1): {invalid_rows}")
        print(f"   Retention rate: {valid_rows/total_rows*100:.1f}%")
        
        if valid_rows == 0:
            print("Warning: No valid rows found after filtering!")
            return df  # 返回原始数据以避免空数据集
        
        return filtered_df

    def _basic_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础数据清理"""
        
        print("Performing basic data cleaning...")
        
        # 记录原始形状
        original_shape = df.shape
        
        # 1. 移除完全空的行和列
        df = df.dropna(how='all')  # 移除全空行
        df = df.dropna(axis=1, how='all')  # 移除全空列
        
        # 2. 清理列名
        df.columns = df.columns.str.strip()  # 移除前后空格
        
        # 3. 处理明显的缺失值标记
        missing_markers = ['SMD', 'smd', 'Notreported', 'not reported', 'Not reported', 'NOT REPORTED']
        for marker in missing_markers:
            df = df.replace(marker, np.nan)
        
        # 4. 数据类型优化
        for col in df.columns:
            # 尝试将object类型的数值列转换为数值类型
            if df[col].dtype == 'object':
                # 检查是否可以转换为数值
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    # 如果转换后非空值比例大于50%，则认为是数值列
                    if numeric_series.notna().sum() / len(df) > 0.5:
                        df[col] = numeric_series
                except:
                    pass
        
        print(f"   Original shape: {original_shape}")
        print(f"   Cleaned shape:  {df.shape}")
        print(f"   Removed {original_shape[0] - df.shape[0]} rows")
        print(f"   Removed {original_shape[1] - df.shape[1]} columns")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, 
                           output_path: Optional[str] = None) -> bool:
        """保存预处理后的数据"""
        
        if output_path is None:
            output_path = config.PROCESSED_DATA_FILE
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存数据
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ Processed data saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save processed data: {e}")
            return False
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据集信息"""
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }
        
        return info
    
    def print_data_summary(self, df: pd.DataFrame):
        """打印数据摘要"""
        
        info = self.get_data_info(df)
        
        print("\n📊 Data Summary:")
        print("=" * 50)
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        
        print(f"\nNumeric Columns ({len(info['numeric_columns'])}):")
        for col in info['numeric_columns'][:10]:  # 显示前10个
            missing = info['missing_values'][col]
            print(f"  - {col}: {missing} missing values")
        
        print(f"\nCategorical Columns ({len(info['categorical_columns'])}):")
        for col in info['categorical_columns'][:10]:  # 显示前10个
            missing = info['missing_values'][col]
            print(f"  - {col}: {missing} missing values")
        
        print("=" * 50)

# 便捷函数
def load_default_data() -> Optional[pd.DataFrame]:
    """加载默认数据"""
    loader = DataLoader("csv")
    return loader.load_data()

def load_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """从指定CSV文件加载数据"""
    loader = DataLoader("csv")
    return loader.load_data(file_path)

def load_data_from_database(table_name: str = "research_data") -> Optional[pd.DataFrame]:
    """从数据库加载数据"""
    loader = DataLoader("database")
    return loader.load_data(table_name=table_name)