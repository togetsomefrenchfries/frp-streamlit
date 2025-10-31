# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 主程序入口
Main Entry Point for FRP Rebar Durability Prediction

提供完整的命令行界面和API
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from config import config
from data_loader import DataLoader, load_default_data
from preprocessor import FRPDataPreprocessor, preprocess_frp_data
from model_trainer import ModelTrainer, train_frp_models
from predictor import FRPPredictor, FRPPredictionPipeline, create_sample_input
from utils import apply_sklearn_compatibility_patch

def setup_environment():
    """设置运行环境"""
    print("🔧 Setting up environment...")
    
    # 应用兼容性补丁
    apply_sklearn_compatibility_patch()
    
    # 创建必要的目录
    config.DATA_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.LOGS_DIR.mkdir(exist_ok=True)
    
    print("✅ Environment setup completed")

def load_data_command(args):
    """数据加载命令"""
    print("📂 Loading data...")
    
    if args.source == 'csv':
        data_path = args.path or config.DEFAULT_DATA_FILE
        if not Path(data_path).exists():
            print(f"❌ Data file not found: {data_path}")
            return
        
        loader = DataLoader("csv")
        df = loader.load_data(data_path)
    
    elif args.source == 'database':
        loader = DataLoader("database")
        df = loader.load_data(table_name=args.table)
    
    else:
        print(f"❌ Unsupported data source: {args.source}")
        return
    
    if df is not None:
        loader.print_data_summary(df)
        
        # 保存处理后的数据
        if args.save:
            output_path = args.output or config.PROCESSED_DATA_FILE
            loader.save_processed_data(df, output_path)
    else:
        print("❌ Failed to load data")

def preprocess_command(args):
    """数据预处理命令"""
    print("🔄 Preprocessing data...")
    
    # 加载数据
    input_path = args.input or config.DEFAULT_DATA_FILE
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    loader = DataLoader("csv")
    df = loader.load_data(input_path)
    
    if df is None:
        print("❌ Failed to load input data")
        return
    
    # 预处理
    processed_df, feature_info = preprocess_frp_data(df)
    
    if processed_df is not None:
        print(f"✅ Preprocessing completed: {processed_df.shape}")
        
        # 保存结果
        output_path = args.output or config.PROCESSED_DATA_FILE
        loader.save_processed_data(processed_df, output_path)
        
        print(f"📊 Processed data saved to: {output_path}")
        
        # 打印特征信息
        print("\\n📈 Feature Information:")
        for key, value in feature_info.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")
    else:
        print("❌ Preprocessing failed")

def train_command(args):
    """模型训练命令"""
    print("🚀 Training models...")
    
    # 加载数据
    input_path = args.input or config.PROCESSED_DATA_FILE
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        print("Hint: Run preprocessing first or specify correct input file")
        return
    
    loader = DataLoader("csv")
    df = loader.load_data(input_path)
    
    if df is None:
        print("❌ Failed to load training data")
        return
    
    # 训练模型
    trainer = ModelTrainer(enable_hyperparameter_tuning=args.tune)
    
    if args.model == 'all':
        results = trainer.train_all_models(df, args.target)
    else:
        # 训练单个模型
        X, y, feature_info = trainer.prepare_data(df, args.target)
        result = trainer.train_model(args.model, X, y)
        results = {args.model: result}
    
    # 保存模型
    if args.save and results:
        trainer.evaluation_results = results
        output_dir = args.output or config.MODELS_DIR
        trainer.save_models(output_dir)

def predict_command(args):
    """预测命令"""
    print("🔮 Making predictions...")
    
    # 检查模型文件
    model_path = args.model
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    # 创建预测器
    predictor = FRPPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return
    
    # 预测方式
    if args.interactive:
        interactive_prediction(predictor)
    elif args.input:
        batch_prediction(predictor, args.input, args.output)
    else:
        # 使用样本数据进行演示
        sample_input = create_sample_input()
        print("\\n📝 Using sample input data:")
        for key, value in sample_input.items():
            print(f"   {key}: {value}")
        
        prediction = predictor.predict_single(sample_input)
        
        if prediction is not None:
            explanation = predictor.explain_prediction(sample_input, prediction)
            print_prediction_result(explanation)
        else:
            print("❌ Prediction failed")

def interactive_prediction(predictor):
    """交互式预测"""
    print("\\n🎯 Interactive Prediction Mode")
    print("Enter values for each feature (press Enter for default):")
    
    sample_input = create_sample_input()
    user_input = {}
    
    for key, default_value in sample_input.items():
        try:
            user_value = input(f"{key} (default: {default_value}): ").strip()
            if user_value:
                # 尝试转换为适当的类型
                if isinstance(default_value, (int, float)):
                    user_input[key] = float(user_value)
                else:
                    user_input[key] = user_value
            else:
                user_input[key] = default_value
        except (ValueError, KeyboardInterrupt):
            user_input[key] = default_value
    
    # 进行预测
    prediction = predictor.predict_single(user_input)
    
    if prediction is not None:
        explanation = predictor.explain_prediction(user_input, prediction)
        print_prediction_result(explanation)
    else:
        print("❌ Prediction failed")

def batch_prediction(predictor, input_path, output_path):
    """批量预测"""
    import pandas as pd
    
    # 加载输入数据
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    try:
        input_df = pd.read_csv(input_path)
        print(f"📂 Loaded {len(input_df)} samples for prediction")
        
        # 批量预测
        predictions = predictor.predict_batch(input_df)
        
        if predictions is not None:
            # 添加预测结果到DataFrame
            result_df = input_df.copy()
            result_df['predicted_retention'] = predictions
            
            # 保存结果
            if output_path:
                result_df.to_csv(output_path, index=False)
                print(f"✅ Predictions saved to: {output_path}")
            
            # 显示统计信息
            print(f"\\n📊 Prediction Statistics:")
            print(f"   Mean: {predictions.mean():.4f}")
            print(f"   Std:  {predictions.std():.4f}")
            print(f"   Min:  {predictions.min():.4f}")
            print(f"   Max:  {predictions.max():.4f}")
        
        else:
            print("❌ Batch prediction failed")
    
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")

def print_prediction_result(explanation):
    """打印预测结果"""
    print("\\n" + "="*60)
    print("🎯 PREDICTION RESULT")
    print("="*60)
    
    prediction = explanation.get('prediction')
    if prediction is not None:
        print(f"Tensile Strength Retention: {prediction:.4f} ({prediction*100:.2f}%)")
        print(f"Durability Assessment: {explanation.get('durability_assessment', 'Unknown')}")
        print(f"Recommendation: {explanation.get('recommendation', 'No recommendation')}")
    else:
        print("❌ No prediction available")
    
    print("="*60)

def info_command(args):
    """显示系统信息"""
    print("\\n📋 FRP Prediction System Information")
    print("="*50)
    
    # 配置信息
    print("Configuration:")
    print(f"   Data Directory: {config.DATA_DIR}")
    print(f"   Models Directory: {config.MODELS_DIR}")
    print(f"   Supported Models: {list(config.SUPPORTED_MODELS.keys())}")
    print(f"   Core Features: {len(config.CORE_FEATURES)}")
    
    # 检查数据文件
    print("\\nData Files:")
    if config.DEFAULT_DATA_FILE.exists():
        print(f"   ✅ Default data: {config.DEFAULT_DATA_FILE}")
    else:
        print(f"   ❌ Default data: {config.DEFAULT_DATA_FILE} (not found)")
    
    # 检查模型文件
    print("\\nAvailable Models:")
    if config.MODELS_DIR.exists():
        model_files = list(config.MODELS_DIR.glob("*.pkl"))
        if model_files:
            for model_file in model_files:
                print(f"   ✅ {model_file.name}")
        else:
            print("   ❌ No trained models found")
    else:
        print("   ❌ Models directory not found")
    
    print("="*50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FRP Rebar Durability Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py info                                    # Show system information
  python main.py load --source csv --path data.csv      # Load data from CSV
  python main.py preprocess --input data.csv            # Preprocess data
  python main.py train --input processed_data.csv       # Train all models
  python main.py predict --model models/best_model.pkl  # Make predictions
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Show system information')
    
    # Load data command
    parser_load = subparsers.add_parser('load', help='Load and examine data')
    parser_load.add_argument('--source', choices=['csv', 'database'], default='csv', help='Data source')
    parser_load.add_argument('--path', help='Path to CSV file')
    parser_load.add_argument('--table', default='research_data', help='Database table name')
    parser_load.add_argument('--save', action='store_true', help='Save processed data')
    parser_load.add_argument('--output', help='Output file path')
    
    # Preprocess command
    parser_preprocess = subparsers.add_parser('preprocess', help='Preprocess data')
    parser_preprocess.add_argument('--input', help='Input CSV file path')
    parser_preprocess.add_argument('--output', help='Output CSV file path')
    
    # Train command
    parser_train = subparsers.add_parser('train', help='Train models')
    parser_train.add_argument('--input', help='Input CSV file path')
    parser_train.add_argument('--model', default='all', help='Model to train (all, random_forest, xgboost, lightgbm)')
    parser_train.add_argument('--target', help='Target variable column name')
    parser_train.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser_train.add_argument('--save', action='store_true', default=True, help='Save trained models')
    parser_train.add_argument('--output', help='Output directory for models')
    
    # Predict command
    parser_predict = subparsers.add_parser('predict', help='Make predictions')
    parser_predict.add_argument('--model', required=True, help='Path to trained model file')
    parser_predict.add_argument('--input', help='Input CSV file for batch prediction')
    parser_predict.add_argument('--output', help='Output CSV file for batch prediction')
    parser_predict.add_argument('--interactive', action='store_true', help='Interactive prediction mode')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 执行命令
    if args.command == 'info':
        info_command(args)
    elif args.command == 'load':
        load_data_command(args)
    elif args.command == 'preprocess':
        preprocess_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()