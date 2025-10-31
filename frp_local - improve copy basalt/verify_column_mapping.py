#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Python索引与Excel列的对应关系
"""

def show_index_mapping():
    """显示Python索引与Excel列的对应关系"""
    print("Python索引与Excel列对应关系:")
    print("=" * 50)
    print(f"{'Python索引':<10} | {'Excel列名':<8} | {'Excel列号':<8} | 说明")
    print("-" * 50)
    
    # 显示前30列的对应关系
    for i in range(30):
        excel_col = chr(ord('A') + i) if i < 26 else f"A{chr(ord('A') + i - 26)}"
        excel_num = i + 1
        print(f"{i:<10} | {excel_col:<8} | 第{excel_num}列{'':<3} | pandas.iloc[:, {i}]")

def explain_feature_mappings():
    """解释特征映射中的索引含义"""
    print("\n\n代码中特征映射的实际含义:")
    print("=" * 60)
    
    # 从原代码中提取的特征映射
    feature_mappings = {
        'pH_of_condition_enviroment': {
            'column_indices': [54, 59, 60],
            'comment': '第55列, 第60列, 第61列'
        },
        'Chloride_ion': {
            'column_indices': [61, 64, 77],
            'comment': '第62列, 第65列, 第78列'
        },
        'concrete': {
            'column_indices': [53, 56, 57],
            'comment': '第54列, 第57列, 第58列'
        },
        'diameter': {
            'column_indices': [18, 20],
            'comment': '第19列, 第21列'
        },
        'Glass_or_Basalt': {
            'column_indices': [8],
            'comment': '第9列'
        },
        'Vinyl_ester_or_Epoxy': {
            'column_indices': [10],
            'comment': '第11列'
        },
        'fiber_content': {
            'column_indices': [15, 16],
            'comment': '第16列, 第17列'
        },
        'surface_treatment': {
            'column_indices': [22],
            'comment': '第23列'
        }
    }
    
    for feature_name, config in feature_mappings.items():
        print(f"\n{feature_name}:")
        print(f"  Python索引: {config['column_indices']}")
        print(f"  代码注释: {config['comment']}")
        
        excel_cols = []
        for idx in config['column_indices']:
            if idx < 26:
                excel_col = chr(ord('A') + idx)
            elif idx < 52:
                excel_col = f"A{chr(ord('A') + idx - 26)}"
            else:
                # 对于超过AZ的列，使用更复杂的计算
                first = (idx - 26) // 26
                second = (idx - 26) % 26
                excel_col = f"{chr(ord('A') + first)}{chr(ord('A') + second)}"
            excel_cols.append(excel_col)
        
        print(f"  对应Excel列: {excel_cols}")
        print(f"  访问方式: data.iloc[:, {config['column_indices']}]")

def verify_specific_examples():
    """验证具体的例子"""
    print("\n\n具体例子验证:")
    print("=" * 40)
    
    examples = [
        (0, "A", "第1列"),
        (8, "I", "第9列"),
        (10, "K", "第11列"),
        (15, "P", "第16列"),
        (18, "S", "第19列"),
        (22, "W", "第23列"),
        (54, "BC", "第55列"),
        (90, "CM", "第91列")
    ]
    
    for python_idx, excel_col, excel_desc in examples:
        print(f"Python索引 {python_idx:2d} = Excel {excel_col:3s}列 = {excel_desc}")

if __name__ == "__main__":
    show_index_mapping()
    explain_feature_mappings()
    verify_specific_examples()
    
    print("\n\n总结:")
    print("🔹 Python索引从0开始")
    print("🔹 Excel列从A(第1列)开始")
    print("🔹 Python索引N = Excel第(N+1)列")
    print("🔹 代码中的注释'第X列'指的是Excel中的列号")
    print("🔹 实际Python代码使用的是(X-1)作为索引")