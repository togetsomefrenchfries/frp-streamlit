# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 模型训练模块
Model Training Module for FRP Rebar Durability Prediction

包含：
- 多种机器学习算法
- 超参数优化
- 模型评估和验证
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

# 可选的高级算法
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

try:
    from .config import config
    from .utils import create_enhanced_preprocessor, diagnose_model_performance, print_model_performance, save_model_safely
    from .preprocessor import FRPDataPreprocessor
except ImportError:
    from config import config
    from utils import create_enhanced_preprocessor, diagnose_model_performance, print_model_performance, save_model_safely
    from preprocessor import FRPDataPreprocessor

class ModelTrainer:
    """模型训练器 - 支持多种算法和自动化训练流程"""
    
    def __init__(self, enable_hyperparameter_tuning: bool = True):
        """
        初始化模型训练器
        
        Args:
            enable_hyperparameter_tuning: 是否启用超参数优化
        """
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        self.models = {}
        self.trained_models = {}
        self.evaluation_results = {}
        self.feature_info = None
        
        # 初始化支持的模型
        self._init_models()
    
    def _init_models(self):
        """初始化支持的模型"""
        
        # Random Forest（始终可用）
        self.models['random_forest'] = RandomForestRegressor(
            **config.get_model_params('random_forest')
        )
        
        # XGBoost（如果可用）
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = XGBRegressor(
                **config.get_model_params('xgboost')
            )
        
        # LightGBM（如果可用）
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = LGBMRegressor(
                **config.get_model_params('lightgbm')
            )
        
        print(f"✅ Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def prepare_train_val_test_splits(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备7:2:1数据分割（训练:验证:测试）
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # 首先分离出测试集(10%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=42
        )
        
        # 从剩余90%中分离训练集(77.8%≈70%)和验证集(22.2%≈20%)
        val_ratio = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)  # 0.2/0.9 ≈ 0.222
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )
        
        print(f"📊 数据分割完成:")
        print(f"   训练集: {len(X_train)} 样本 ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   验证集: {len(X_val)} 样本 ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   测试集: {len(X_test)} 样本 ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def hyperparameter_search(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray) -> object:
        """
        使用验证集进行超参数搜索
        
        Args:
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征  
            y_val: 验证目标
            
        Returns:
            优化后的模型
        """
        if not config.ENABLE_HYPERPARAMETER_TUNING:
            return self.models[model_name]
        
        if model_name not in config.HYPERPARAMETER_SPACES:
            print(f"⚠️ No hyperparameter space defined for {model_name}, using default parameters")
            return self.models[model_name]
        
        print(f"🔍 开始{model_name}超参数优化...")
        
        base_model = self.models[model_name]
        param_space = config.HYPERPARAMETER_SPACES[model_name]
        
        # 创建预处理器
        if self.feature_info:
            preprocessor = create_enhanced_preprocessor(
                categorical_cols=self.feature_info['categorical_features'],
                numeric_cols=self.feature_info['numeric_features']
            )
            
            # 预处理训练和验证数据
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
        else:
            X_train_processed = X_train
            X_val_processed = X_val
        
        # 选择搜索方法
        if config.HYPERPARAMETER_SEARCH_METHOD == 'grid':
            search = GridSearchCV(
                base_model, 
                param_space,
                cv=config.TUNING_CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
        elif config.HYPERPARAMETER_SEARCH_METHOD == 'random':
            search = RandomizedSearchCV(
                base_model,
                param_space, 
                n_iter=config.TUNING_N_ITER,
                cv=config.TUNING_CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            print(f"⚠️ Unsupported search method: {config.HYPERPARAMETER_SEARCH_METHOD}")
            return base_model
        
        # 执行搜索
        search.fit(X_train_processed, y_train)
        
        # 在验证集上评估最佳模型
        best_model = search.best_estimator_
        val_score = best_model.score(X_val_processed, y_val)
        
        print(f"✅ 超参数优化完成")
        print(f"   最佳参数: {search.best_params_}")
        print(f"   验证集R²: {val_score:.4f}")
        
        return best_model
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        准备训练数据
        
        Args:
            df: 预处理后的数据
            target_column: 目标变量列名
            
        Returns:
            X, y, feature_info
        """
        
        if target_column is None:
            target_column = config.TARGET_VARIABLE
            # 尝试匹配可能的目标列名
            possible_targets = ['Tensile strength retention', 'Tensile_strength_retention', 'retention1']
            for col in possible_targets:
                if col in df.columns:
                    target_column = col
                    break
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        print(f"🎯 Using target variable: {target_column}")
        
        # 分离特征和目标变量
        feature_columns = [col for col in df.columns if col not in [target_column, 'Title']]
        X = df[feature_columns]
        y = df[target_column]
        
        # 移除目标变量为空的行
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📊 Data shape after removing missing targets: X={X.shape}, y={y.shape}")
        
        # 分离数值和分类特征
        numeric_features = []
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # 检查是否为二进制分类特征
                unique_values = X[col].dropna().unique()
                if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values if not pd.isna(v)):
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        # 保存特征信息
        self.feature_info = {
            'feature_columns': feature_columns,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'target_variable': target_column,
            'feature_names': list(X.columns)
        }
        
        print(f"📈 Feature analysis:")
        print(f"   - Numeric features: {len(numeric_features)}")
        print(f"   - Categorical features: {len(categorical_features)}")
        print(f"   - Total features: {len(feature_columns)}")
        
        return X.values, y.values, self.feature_info
    
    def train_model_with_hyperopt(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        使用超参数优化训练单个模型（7:2:1分割）
        
        Args:
            model_name: 模型名称
            X: 特征数据
            y: 目标变量
            
        Returns:
            训练结果
        """
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not supported. Available: {list(self.models.keys())}")
        
        print(f"🚀 Training {model_name} with hyperparameter optimization...")
        
        # 使用7:2:1分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_val_test_splits(X, y)
        
        try:
            # 超参数优化
            optimized_model = self.hyperparameter_search(model_name, X_train, y_train, X_val, y_val)
            
            # 创建预处理器
            if self.feature_info:
                preprocessor = create_enhanced_preprocessor(
                    categorical_cols=self.feature_info['categorical_features'],
                    numeric_cols=self.feature_info['numeric_features']
                )
                
                # 创建管道
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', optimized_model)
                ])
                
                # 在训练+验证集上重新训练最终模型
                X_train_val = np.vstack([X_train, X_val])
                y_train_val = np.concatenate([y_train, y_val])
                
                pipeline.fit(X_train_val, y_train_val)
                
                # 在测试集上评估
                y_pred = pipeline.predict(X_test)
                
                # 保存训练好的模型
                self.trained_models[model_name] = pipeline
                
            else:
                # 没有特征信息时的简单处理
                X_train_val = np.vstack([X_train, X_val])
                y_train_val = np.concatenate([y_train, y_val])
                
                optimized_model.fit(X_train_val, y_train_val)
                y_pred = optimized_model.predict(X_test)
                
                # 保存训练好的模型
                self.trained_models[model_name] = optimized_model
            
            # 计算评估指标
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            result = {
                'model_name': model_name,
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'test_size': len(y_test),
                'train_size': len(y_train),
                'val_size': len(y_val),
                'hyperparameter_optimized': config.ENABLE_HYPERPARAMETER_TUNING
            }
            
            # 保存结果
            self.evaluation_results[model_name] = result
            
            # 打印结果
            print(f"✅ {model_name} training completed!")
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   Test samples: {len(y_test)}")
            
            return result
            
        except Exception as e:
            print(f"❌ {model_name} training failed: {str(e)}")
            return {
                'model_name': model_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                   test_size: float = None) -> Dict[str, Any]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            
        Returns:
            训练结果
        """
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not supported. Available: {list(self.models.keys())}")
        
        if test_size is None:
            test_size = config.TEST_SIZE
        
        print(f"🚀 Training {model_name}...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 创建预处理器
        if self.feature_info:
            preprocessor = create_enhanced_preprocessor(
                categorical_cols=self.feature_info['categorical_features'],
                numeric_cols=self.feature_info['numeric_features']
            )
            
            # 创建管道
            model = self.models[model_name]
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
        else:
            # 如果没有特征信息，直接使用模型
            pipeline = self.models[model_name]
        
        # 超参数优化
        if self.enable_hyperparameter_tuning:
            pipeline = self._optimize_hyperparameters(pipeline, X_train, y_train, model_name)
        
        # 训练模型
        pipeline.fit(X_train, y_train)
        
        # 预测
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # 评估
        train_metrics = diagnose_model_performance(y_train, y_pred_train, f"{model_name}_train")
        test_metrics = diagnose_model_performance(y_test, y_pred_test, f"{model_name}_test")
        
        # 交叉验证
        cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                   cv=config.CV_FOLDS, scoring='r2')
        
        # 保存训练结果
        result = {
            'model': pipeline,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_info': self.feature_info
        }
        
        self.trained_models[model_name] = result
        
        # 打印结果
        print_model_performance(test_metrics)
        print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return result
    
    def _optimize_hyperparameters(self, pipeline, X_train, y_train, model_name):
        """超参数优化"""
        
        print(f"🔧 Optimizing hyperparameters for {model_name}...")
        
        # 定义搜索空间
        param_grids = {
            'random_forest': {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [3, 6, 10, None],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [3, 6, 9],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [3, 6, 9],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__subsample': [0.8, 0.9, 1.0]
            }
        }
        
        param_grid = param_grids.get(model_name, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                pipeline, param_grid,
                cv=config.TUNING_CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        
        return pipeline
    
    def train_all_models(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        训练所有可用模型
        
        Args:
            df: 预处理后的数据
            target_column: 目标变量列名
            
        Returns:
            所有模型的训练结果
        """
        
        print("🚀 Training all available models...")
        
        # 准备数据
        X, y, feature_info = self.prepare_data(df, target_column)
        
        # 训练每个模型
        results = {}
        for model_name in self.models.keys():
            try:
                # 使用新的超参数优化训练方法
                result = self.train_model_with_hyperopt(model_name, X, y)
                results[model_name] = result
                print(f"✅ {model_name} training completed")
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
                continue
        
        # 创建集成模型
        if len(results) > 1:
            try:
                ensemble_result = self._create_ensemble_model(X, y, results)
                results['ensemble'] = ensemble_result
                print("✅ Ensemble model created")
            except Exception as e:
                print(f"❌ Ensemble model creation failed: {e}")
        
        self.evaluation_results = results
        return results
    
    def _create_ensemble_model(self, X, y, individual_results):
        """创建集成模型"""
        
        print("🔗 Creating ensemble model...")
        
        # 准备基学习器
        estimators = []
        for name, result in individual_results.items():
            if 'model' in result:
                estimators.append((name, result['model']))
        
        # 创建投票回归器
        ensemble = VotingRegressor(estimators=estimators)
        
        # 分割数据进行评估
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=42
        )
        
        # 训练集成模型
        ensemble.fit(X_train, y_train)
        
        # 预测和评估
        y_pred_train = ensemble.predict(X_train)
        y_pred_test = ensemble.predict(X_test)
        
        train_metrics = diagnose_model_performance(y_train, y_pred_train, "ensemble_train")
        test_metrics = diagnose_model_performance(y_test, y_pred_test, "ensemble_test")
        
        # 交叉验证
        cv_scores = cross_val_score(ensemble, X_train, y_train, 
                                   cv=config.CV_FOLDS, scoring='r2')
        
        result = {
            'model': ensemble,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_info': self.feature_info
        }
        
        return result
    
    def compare_models(self) -> pd.DataFrame:
        """比较所有训练的模型"""
        
        if not self.evaluation_results:
            print("❌ No trained models to compare")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, result in self.evaluation_results.items():
            test_metrics = result['test_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'R²': test_metrics['r2'],
                'RMSE': test_metrics['rmse'],
                'MAE': test_metrics['mae'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        print("📊 Model Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        print("=" * 80)
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[str, Any]:
        """获取最佳模型"""
        
        if not self.evaluation_results:
            return None, None
        
        best_model_name = None
        best_score = -np.inf
        
        for model_name, result in self.evaluation_results.items():
            score = result['test_metrics']['r2']
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name, self.evaluation_results[best_model_name]
    
    def save_models(self, output_dir: str = None):
        """保存所有训练的模型"""
        
        if output_dir is None:
            output_dir = config.MODELS_DIR
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_count = 0
        
        for model_name, result in self.evaluation_results.items():
            try:
                model_file = output_path / f"{model_name}_model.pkl"
                
                # 保存模型和相关信息
                save_data = {
                    'model': result['model'],
                    'feature_info': result['feature_info'],
                    'metrics': result['test_metrics'],
                    'cv_scores': result['cv_scores']
                }
                
                if save_model_safely(save_data, str(model_file)):
                    saved_count += 1
                    
            except Exception as e:
                print(f"❌ Failed to save {model_name}: {e}")
        
        print(f"✅ Saved {saved_count}/{len(self.evaluation_results)} models to {output_path}")

# 便捷函数
def train_frp_models(df: pd.DataFrame, target_column: str = None, 
                     enable_hyperparameter_tuning: bool = True) -> Dict[str, Any]:
    """便捷的模型训练函数"""
    
    trainer = ModelTrainer(enable_hyperparameter_tuning=enable_hyperparameter_tuning)
    results = trainer.train_all_models(df, target_column)
    
    # 显示比较结果
    trainer.compare_models()
    
    # 获取最佳模型
    best_name, best_result = trainer.get_best_model()
    if best_name:
        print(f"🏆 Best model: {best_name} (R² = {best_result['test_metrics']['r2']:.4f})")
    
    return results