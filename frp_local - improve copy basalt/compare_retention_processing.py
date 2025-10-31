#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def compare_retention_processing():
    """比较40参数实验和random prediction test中retention的处理"""
    try:
        # 读取数据库文件
        df = pd.read_excel('../../database 4.xlsx')
        
        print(f"🔍 比较retention处理方式:")
        print(f"=" * 60)
        
        # 检查第101列 (索引100) - 应该是retention1
        if len(df.columns) > 100:
            retention_col = df.iloc[:, 100]
            col_name = df.columns[100]
            
            print(f"第101列 (索引100): {col_name}")
            print(f"  数据类型: {retention_col.dtype}")
            print(f"  非空数据: {retention_col.notna().sum()}/{len(df)}")
            
            # 转换为数值并分析
            numeric_retention = pd.to_numeric(retention_col, errors='coerce')
            valid_retention = numeric_retention.dropna()
            
            if len(valid_retention) > 0:
                print(f"  数值统计:")
                print(f"    有效数值: {len(valid_retention)}")
                print(f"    范围: {valid_retention.min():.6f} - {valid_retention.max():.6f}")
                print(f"    均值: {valid_retention.mean():.6f}")
                print(f"    中位数: {valid_retention.median():.6f}")
                print(f"    前10个值: {valid_retention.head(10).tolist()}")
            
            # 特别检查包含'tensile'的行
            print(f"\n检查Target_parameter='tensile'的行中的retention值:")
            target_col = df.iloc[:, 30]  # AE列
            tensile_mask = target_col.astype(str).str.lower().str.contains('tensile', na=False)
            
            if tensile_mask.any():
                tensile_retention = numeric_retention[tensile_mask].dropna()
                print(f"  tensile行数: {tensile_mask.sum()}")
                print(f"  tensile中有retention值的行数: {len(tensile_retention)}")
                if len(tensile_retention) > 0:
                    print(f"  tensile中retention范围: {tensile_retention.min():.6f} - {tensile_retention.max():.6f}")
                    print(f"  tensile中retention均值: {tensile_retention.mean():.6f}")
                    print(f"  tensile中retention前10个值: {tensile_retention.head(10).tolist()}")
                    
            # 检查retention的数据质量
            print(f"\n数据质量分析:")
            non_numeric_values = retention_col[pd.to_numeric(retention_col, errors='coerce').isna() & retention_col.notna()]
            if len(non_numeric_values) > 0:
                print(f"  非数值的retention值: {non_numeric_values.value_counts().head(10).to_dict()}")
            else:
                print(f"  所有非空retention值都是数值型")
                
            # 检查retention在合理范围内的数据
            reasonable_retention = valid_retention[(valid_retention >= 0) & (valid_retention <= 2)]
            print(f"  合理范围(0-2)的retention值: {len(reasonable_retention)}/{len(valid_retention)} ({len(reasonable_retention)/len(valid_retention)*100:.1f}%)")
            
        else:
            print("第101列不存在")
            
        # 对比检查在训练时应该使用的数据
        print(f"\n📊 模拟训练时的数据筛选:")
        
        # 1. Target_parameter = 'tensile'
        target_mask = df.iloc[:, 30].astype(str).str.lower().str.contains('tensile', na=False)
        
        # 2. retention1有数值
        retention_mask = pd.to_numeric(df.iloc[:, 100], errors='coerce').notna()
        
        # 3. condition_time有值 (假设第52列是condition_time)
        condition_time_mask = df.iloc[:, 51].notna() if len(df.columns) > 51 else pd.Series(True, index=df.index)
        
        # 综合筛选
        final_mask = target_mask & retention_mask & condition_time_mask
        
        print(f"  原始数据: {len(df)} 行")
        print(f"  Target_parameter='tensile': {target_mask.sum()} 行")
        print(f"  retention1有数值: {retention_mask.sum()} 行")
        print(f"  condition_time有值: {condition_time_mask.sum()} 行")
        print(f"  满足所有条件: {final_mask.sum()} 行")
        print(f"  数据保留率: {final_mask.sum()/len(df)*100:.1f}%")
        
        # 分析最终筛选后的retention分布
        if final_mask.sum() > 0:
            final_retention = pd.to_numeric(df.iloc[:, 100], errors='coerce')[final_mask].dropna()
            if len(final_retention) > 0:
                print(f"\n最终数据中的retention分布:")
                print(f"  数量: {len(final_retention)}")
                print(f"  范围: {final_retention.min():.6f} - {final_retention.max():.6f}")
                print(f"  均值: {final_retention.mean():.6f}")
                print(f"  标准差: {final_retention.std():.6f}")
                
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_retention_processing()