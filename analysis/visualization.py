import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_csv(csv_df, crud_df=None, direct_llm_df=None):
    """
    可视化CSV中的所有数值指标，按question_type分组求均值并显示柱状图
    """
    # 确保question_type列存在
    if 'question_type' not in csv_df.columns:
        print("警告: CSV中没有question_type列，无法按类型分组")
        return
    
    def safe_convert_to_numeric(df):
        """安全地将数据框中的列转换为数值类型，返回转换后的数据框和数值列列表"""
        df_copy = df.copy()
        numeric_cols = []
        
        for col in df_copy.columns:
            if col == 'question_type':
                continue
                
            try:
                original_values = df_copy[col].copy()
                converted_series = pd.to_numeric(df_copy[col], errors='coerce')
                
                # 如果转换后至少有一个有效数值，则认为是数值列
                if converted_series.notna().any():
                    df_copy[col] = converted_series
                    numeric_cols.append(col)
            except Exception:
                continue
                
        return df_copy, numeric_cols
    
    # 转换主实验数据
    csv_df_conv, metric_cols = safe_convert_to_numeric(csv_df)
    print(f"主实验数值指标: {metric_cols}")
    
    if not metric_cols:
        print("警告: 没有找到可视化的数值指标列")
        return
    
    # 按question_type分组计算均值
    grouped = csv_df_conv.groupby('question_type')[metric_cols].mean()
    question_types = grouped.index.tolist()
    
    # 转换并处理baseline数据
    crud_grouped = None
    direct_llm_grouped = None
    
    if crud_df is not None and 'question_type' in crud_df.columns:
        crud_df_conv, crud_numeric_cols = safe_convert_to_numeric(crud_df)
        common_crud_cols = [col for col in metric_cols if col in crud_numeric_cols]
        if common_crud_cols:
            crud_grouped = crud_df_conv.groupby('question_type')[common_crud_cols].mean()
            print(f"CRUD baseline指标: {common_crud_cols}")
    
    if direct_llm_df is not None and 'question_type' in direct_llm_df.columns:
        direct_llm_df_conv, direct_llm_numeric_cols = safe_convert_to_numeric(direct_llm_df)
        common_llm_cols = [col for col in metric_cols if col in direct_llm_numeric_cols]
        if common_llm_cols:
            direct_llm_grouped = direct_llm_df_conv.groupby('question_type')[common_llm_cols].mean()
            print(f"Direct LLM baseline指标: {common_llm_cols}")
    
    # 计算数据集数量
    n_datasets = 1
    if crud_grouped is not None:
        n_datasets += 1
    if direct_llm_grouped is not None:
        n_datasets += 1
    
    # 创建图表
    n_metrics = len(metric_cols)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # 创建图形对象
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # 确保axes是数组形式
    if n_metrics == 1:
        axes = [axes]
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = ['Main']
    
    # 为每个指标创建柱状图
    for idx, metric in enumerate(metric_cols):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        x = np.arange(len(question_types))
        width = 0.8 / n_datasets
        
        # 收集所有数据集的数据
        all_data = []
        
        # 主实验数据
        main_data = [grouped.loc[qt, metric] if qt in grouped.index and pd.notna(grouped.loc[qt, metric]) else 0 
                    for qt in question_types]
        all_data.append(main_data)
        
        # CRUD数据
        if crud_grouped is not None and metric in crud_grouped.columns:
            crud_data = [crud_grouped.loc[qt, metric] if qt in crud_grouped.index and pd.notna(crud_grouped.loc[qt, metric]) else 0 
                        for qt in question_types]
            all_data.append(crud_data)
            if len(labels) < 2:
                labels.append('CRUD')
        else:
            all_data.append([0] * len(question_types))
        
        # Direct LLM数据
        if direct_llm_grouped is not None and metric in direct_llm_grouped.columns:
            llm_data = [direct_llm_grouped.loc[qt, metric] if qt in direct_llm_grouped.index and pd.notna(direct_llm_grouped.loc[qt, metric]) else 0 
                       for qt in question_types]
            all_data.append(llm_data)
            if len(labels) < 3:
                labels.append('Direct LLM')
        else:
            all_data.append([0] * len(question_types))
        
        # 绘制柱状图
        for i, data in enumerate(all_data[:n_datasets]):
            if any(val != 0 for val in data):  # 只有有数据时才绘制
                ax.bar(x + i * width, data, width, label=labels[i] if i < len(labels) else f'Dataset {i+1}', 
                      color=colors[i] if i < len(colors) else None, alpha=0.8)
        
        # 设置图表属性
        ax.set_xlabel('Question Type', fontsize=10)
        ax.set_ylabel('Mean Value', fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (n_datasets - 1) / 2)
        ax.set_xticklabels(question_types, rotation=45, ha='right')
        
        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        if idx < len(axes):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Metrics Comparison by Question Type', 
                 fontsize=16, fontweight='bold', y=1.002)
    plt.subplots_adjust(top=0.96)
    
    return fig