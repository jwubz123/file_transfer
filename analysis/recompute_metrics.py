#!/usr/bin/env python3
"""
重新计算评估指标脚本

此脚本可以从已有的samples目录重新计算所有指标，无需重新运行RAG agent。
适用于调试Metrics时快速迭代。

用法:
    python recompute_metrics.py <experiment_folder>
    
示例:
    python recompute_metrics.py ../results/experiment_20240101_120000
"""

import os
import sys
import json
import pandas as pd
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加父目录到path以导入评估模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.elsuite.rag_eval.rag_metrics import calculate_metrics, calculate_retrieval_metrics
from evals.elsuite.rag_eval.process_intermediate import (
    process_sample_messages,
    extract_corpus_ids_from_search_agent_response,
    metric_with_token,
    calculate_intermediate_metrics,
    process_intermediate  # 添加process_intermediate函数
)


def load_samples_data(experiment_folder: str) -> List[Dict[str, Any]]:
    """
    从samples目录加载所有样本数据
    
    Args:
        experiment_folder: 实验文件夹路径
        
    Returns:
        样本数据列表
    """
    samples_dir = os.path.join(experiment_folder, 'samples')
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Samples目录不存在: {samples_dir}")
    
    sample_files = glob.glob(os.path.join(samples_dir, "sample_*.json"))
    print(f"找到 {len(sample_files)} 个样本文件")
    
    samples_data = []
    for sample_file in sorted(sample_files):
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            samples_data.append(sample_data)
    
    return samples_data


def extract_retrieved_ids_from_response(response: str) -> List[str]:
    """
    从response中提取retrieved_ids
    
    Args:
        response: RAG系统的响应
        
    Returns:
        检索到的文档ID列表
    """
    retrieved_ids = []
    
    try:
        parsed_response = json.loads(response) if isinstance(response, str) and response.startswith(('{', '[')) else response
        if isinstance(parsed_response, dict):
            sources = parsed_response.get('sources', [])
            retrieved_ids = [os.path.splitext(os.path.basename(path))[0] for path in sources]
    except json.JSONDecodeError:
        pass
    
    return retrieved_ids


def extract_answer_from_response(response: str) -> str:
    """
    从response中提取答案文本
    
    Args:
        response: RAG系统的响应
        
    Returns:
        答案文本
    """
    try:
        parsed_response = json.loads(response) if isinstance(response, str) and response.startswith(('{', '[')) else response
        if isinstance(parsed_response, dict):
            return parsed_response.get('direct_answer', response)
    except json.JSONDecodeError:
        pass
    
    return response


def extract_pre_search_retrieved_ids(retrieval_content: str) -> List[str]:
    """
    从pre-search的检索结果中提取文档ID列表
    
    Args:
        retrieval_content: 检索结果的原始文本
        
    Returns:
        提取的文档ID列表
    """
    if not retrieval_content:
        return []
    
    retrieved_ids = []
    pattern = r'file_path:\s*([^\n]+)'
    matches = re.findall(pattern, retrieval_content)
    
    for file_path in matches:
        filename = os.path.basename(file_path.strip())
        doc_id = os.path.splitext(filename)[0]
        if doc_id:
            retrieved_ids.append(doc_id)
    
    return retrieved_ids


def compute_sample_metrics(
    sample_data: Dict[str, Any],
    enable_llm_judge: bool = True,
    llm_judge_config: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    为单个样本计算所有指标
    
    Args:
        sample_data: 样本数据（从sample_*.json加载）
        enable_llm_judge: 是否启用LLM judge
        llm_judge_config: LLM judge配置
        
    Returns:
        包含所有指标的字典
    """
    sample_info = sample_data.get('sample_info', {})
    response = sample_data.get('response', '')
    execution_time = sample_data.get('execution_time', 0)
    
    # 提取基本信息
    question_id = sample_info.get('question_id', 'unknown')
    question_type = sample_info.get('question_type', '')
    user_query = sample_info.get('user_query', '')
    ideal = sample_info.get('ideal', '')
    expected_context_ids = sample_info.get('expected_context_ids', [])
    corpus_id = sample_info.get('corpus_id', 'CRUD_80000')
    knowledge_source = sample_info.get('knowledge_source', 'unknown')
    
    # 检测是否为Direct LLM模式（无intermediate_results）
    is_direct_llm = 'messages' not in sample_data or not sample_data.get('messages')
    
    # 提取retrieved_ids和answer
    retrieved_ids = extract_retrieved_ids_from_response(response)
    answer_text = extract_answer_from_response(response)
    
    # 计算pre-search检索指标
    pre_search_metrics = {}
    pre_search_retrieved_ids = []
    if not is_direct_llm and 'messages' in sample_data and len(sample_data['messages']) > 1:
        try:
            intermediate_item = sample_data['messages'][1]
            if isinstance(intermediate_item, dict):
                pre_search_content = intermediate_item.get('content', '')
            else:
                pre_search_content = str(intermediate_item)
            
            pre_search_retrieved_ids = extract_pre_search_retrieved_ids(pre_search_content)
            
            if pre_search_retrieved_ids and expected_context_ids:
                pre_search_metrics = calculate_retrieval_metrics(
                    expected_ids=expected_context_ids,
                    retrieved_ids=pre_search_retrieved_ids,
                    k_values=[3, 5]
                )
        except Exception as e:
            print(f"Warning: 计算pre-search指标失败 for {question_id}: {e}")
    
    # 计算主要指标
    metrics = calculate_metrics(
        answer_text,
        question_type,
        ideal,
        expected_context_ids,
        user_query=user_query,
        enable_llm_judge=enable_llm_judge,
        llm_judge_config=llm_judge_config,
        retrieved_ids=retrieved_ids,
        is_direct_llm=is_direct_llm,
        retrieved_documents=None  # 不读取文档内容以加快速度
    )
    
    # 构建CSV行
    csv_row = {
        'question_id': question_id,
        'question_type': question_type,
        'user_query': user_query,
        'knowledge_source': knowledge_source,
        'requested_corpus_id': corpus_id,
        'actual_corpus_id': corpus_id,
        'execution_time_seconds': execution_time,
        'overall_score': metrics.get('overall_score', 0.0),
    }
    
    # 添加各类指标
    base_metrics = ['mrr', 'rejection_recall', 'refusal_choice', 'refusal_reasoning', 
                   'is_refusal', 'retrieved_count']
    extended_metrics = ['llm_judge_score', 'llm_judge_choice', 'llm_judge_reasoning',
                       'ndcg', 'ndcg_at_3', 'ndcg_at_5', 'ncg']
    
    for metric_key in base_metrics + extended_metrics:
        csv_row[metric_key] = metrics.get(metric_key, 'N/A')
    
    # Pre-search检索指标
    csv_row['pre_search_ndcg'] = pre_search_metrics.get('ndcg', 'N/A') if pre_search_metrics else 'N/A'
    csv_row['pre_search_mrr'] = pre_search_metrics.get('mrr', 'N/A') if pre_search_metrics else 'N/A'
    csv_row['pre_search_ncg'] = pre_search_metrics.get('ncg', 'N/A') if pre_search_metrics else 'N/A'
    csv_row['pre_search_retrieved_ids'] = ";".join(pre_search_retrieved_ids) if pre_search_retrieved_ids else "N/A"
    
    # 检索相关信息
    csv_row['retrieved_id'] = ";".join(retrieved_ids) if retrieved_ids else "N/A"
    csv_row['expected_retrieval_id'] = (";".join(expected_context_ids) if isinstance(expected_context_ids, list) 
                                       else expected_context_ids) if expected_context_ids != "NA" else "N/A"
    
    # 解析response为各个组件
    try:
        if isinstance(response, str) and response.startswith(('{', '[')):
            parsed_response = json.loads(response)
            if isinstance(parsed_response, dict):
                csv_row.update({
                    'direct_answer': parsed_response.get('direct_answer', 'N/A'),
                    'response_sources': ";".join(parsed_response.get('sources', [])) if parsed_response.get('sources') else 'N/A',
                    'llm_judge_reasoning': metrics.get('llm_judge_reasoning', 'N/A'),
                    'response_raw': response,
                })
            else:
                csv_row.update({
                    'direct_answer': 'N/A',
                    'response_sources': 'N/A',
                    'llm_judge_reasoning': metrics.get('llm_judge_reasoning', 'N/A'),
                    'response_raw': response,
                })
        else:
            csv_row.update({
                'direct_answer': response if response else 'N/A',
                'response_sources': 'N/A',
                'llm_judge_reasoning': metrics.get('llm_judge_reasoning', 'N/A'),
                'response_raw': response,
            })
    except Exception:
        csv_row.update({
            'direct_answer': response if response else 'N/A',
            'response_sources': 'N/A',
            'llm_judge_reasoning': metrics.get('llm_judge_reasoning', 'N/A'),
            'response_raw': response,
        })
    
    csv_row['gt_answer'] = ideal if ideal != 'NA' else 'N/A'
    
    # 处理token统计和per-task metrics（如果存在messages）
    if 'messages' in sample_data and sample_data['messages']:
        try:
            processed_data, token_stats, per_task_metrics = process_sample_messages(
                sample_data, question_id, expected_context_ids
            )
            
            if token_stats:
                csv_row['rag_agent_input_tokens'] = token_stats['rag_agent_input_tokens']
                csv_row['rag_agent_output_tokens'] = token_stats['rag_agent_output_tokens']
                csv_row['rag_agent_tokens'] = token_stats['rag_agent_tokens']
                csv_row['search_agent_input_tokens'] = sum(token_stats['search_agent_input_tokens'])
                csv_row['search_agent_output_tokens'] = sum(token_stats['search_agent_output_tokens'])
                csv_row['search_agent_tokens'] = sum(token_stats['search_agent_tokens'])
                csv_row['average_search_agent_input_token'] = token_stats['average_search_agent_input_token']
                csv_row['average_search_agent_output_token'] = token_stats['average_search_agent_output_token']
                csv_row['average_search_agent_token'] = token_stats['average_search_agent_token']
                csv_row['num_search_calls'] = token_stats['num_search_calls']
                csv_row['total_num_tokens'] = token_stats['total_num_tokens']
                csv_row['total_num_input_tokens'] = token_stats['total_num_input_tokens']
                csv_row['total_num_output_tokens'] = token_stats['total_num_output_tokens']
                
                # 添加per_task_metrics
                if per_task_metrics:
                    csv_row['per_task_retrieved_ids'] = json.dumps(
                        [';'.join(ids) for ids in per_task_metrics['per_task_retrieved_ids']]
                    )
                    csv_row['per_task_ndcg'] = json.dumps(per_task_metrics['per_task_ndcg'])
                    csv_row['per_task_ncg'] = json.dumps(per_task_metrics['per_task_ncg'])
                    csv_row['per_task_input_tokens'] = json.dumps(per_task_metrics['per_task_input_tokens'])
                    csv_row['per_task_output_tokens'] = json.dumps(per_task_metrics['per_task_output_tokens'])
                    csv_row['per_task_total_tokens'] = json.dumps(per_task_metrics['per_task_total_tokens'])
                    csv_row['per_task_metric_with_token'] = json.dumps(per_task_metrics['per_task_metric_with_token'])
                    csv_row['per_task_perf_score'] = json.dumps(per_task_metrics['per_task_perf_score'])
                else:
                    for key in ['per_task_retrieved_ids', 'per_task_ndcg', 'per_task_ncg',
                               'per_task_input_tokens', 'per_task_output_tokens', 'per_task_total_tokens',
                               'per_task_metric_with_token', 'per_task_perf_score']:
                        csv_row[key] = 'N/A'
                
                # 计算metric_with_token
                base_accuracy = pre_search_metrics.get('ndcg', 0) if pre_search_metrics else 0
                final_accuracy = metrics.get('overall_score', 0)
                input_tokens = token_stats['rag_agent_input_tokens']
                output_tokens = token_stats['rag_agent_output_tokens']
                
                if pd.isna(base_accuracy) or base_accuracy == 'N/A':
                    base_accuracy = 0.0
                else:
                    base_accuracy = float(base_accuracy)
                
                if pd.isna(final_accuracy) or final_accuracy == 'N/A':
                    final_accuracy = 0.0
                else:
                    final_accuracy = float(final_accuracy)
                
                metric_score, perf_score = metric_with_token(
                    accuracy=final_accuracy,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    max_expected_tokens=1000,
                    lambda_acc=1.0,
                    lambda_token=0.4
                )
                csv_row['metric_with_token'] = metric_score
                csv_row['perf_score'] = perf_score
                
                # 计算time_estimate_sec
                tokens_used = (input_tokens / 20.0) + output_tokens
                time_estimate = tokens_used / 20.0  # 假设处理速度为20 tokens/秒
                csv_row['time_estimate_sec'] = round(time_estimate, 2)
        except Exception as e:
            print(f"Warning: 处理token统计失败 for {question_id}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Direct LLM模式，没有token统计
        for key in ['rag_agent_input_tokens', 'rag_agent_output_tokens', 'rag_agent_tokens',
                   'search_agent_input_tokens', 'search_agent_output_tokens', 'search_agent_tokens',
                   'average_search_agent_input_token', 'average_search_agent_output_token',
                   'average_search_agent_token', 'num_search_calls',
                   'total_num_tokens', 'total_num_input_tokens', 'total_num_output_tokens',
                   'per_task_retrieved_ids', 'per_task_ndcg', 'per_task_ncg',
                   'per_task_input_tokens', 'per_task_output_tokens', 'per_task_total_tokens',
                   'per_task_metric_with_token', 'per_task_perf_score',
                   'perf_score', 'time_estimate_sec']:
            csv_row[key] = 'N/A'
    
    return csv_row


def recompute_all_metrics(
    experiment_folder: str,
    enable_llm_judge: bool = True,
    llm_judge_config: Optional[Dict[str, str]] = None,
    output_csv_name: Optional[str] = None,
    intermediate_only: bool = False
) -> pd.DataFrame:
    """
    重新计算所有样本的指标并保存为CSV
    
    Args:
        experiment_folder: 实验文件夹路径
        enable_llm_judge: 是否启用LLM judge
        llm_judge_config: LLM judge配置
        output_csv_name: 输出CSV文件名（可选，默认自动生成）
        
    Returns:
        包含所有指标的DataFrame
    """
    print(f"\n{'='*80}")
    print(f"重新计算指标: {experiment_folder}")
    print(f"{'='*80}\n")
    
    # 加载样本数据
    samples_data = load_samples_data(experiment_folder)
    
    if not samples_data:
        print("错误: 未找到任何样本数据")
        return None
    
    # 设置LLM judge配置（如果未提供）
    if llm_judge_config is None:
        llm_judge_config = {
            "base_url": "http://hk01dgx022:8005/v1",
            "model": "Qwen/Qwen3-30B-A3B-FP8",
            "api_key": "EMPTY",
            "modelgraded_template": "llm_judger",
        }
    if not intermediate_only:
        # 计算每个样本的指标
        csv_results = []
        print(f"开始处理 {len(samples_data)} 个样本...")
        for i, sample_data in enumerate(samples_data, 1):
            question_id = sample_data.get('sample_info', {}).get('question_id', f'unknown_{i}')
            print(f"  [{i}/{len(samples_data)}] 处理样本: {question_id}")
            try:
                csv_row = compute_sample_metrics(
                    sample_data,
                    enable_llm_judge=enable_llm_judge,
                    llm_judge_config=llm_judge_config
                )
                csv_results.append(csv_row)
            except Exception as e:
                print(f"    错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 创建DataFrame
        df = pd.DataFrame(csv_results)
        
        # 如果无法从现有CSV读取，尝试从rag_eval.py导入

        try:
            # 导入RAGEval类以获取preferred_csv_order
            from evals.elsuite.rag_eval.rag_eval import RAGEval
            # 创建一个临时实例只为获取preferred_csv_order
            # 使用最小参数避免复杂初始化
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write('{}')
                temp_jsonl = f.name
            try:
                from evals.api import DummyCompletionFn
                dummy_fn = DummyCompletionFn()
                temp_eval = RAGEval([dummy_fn], temp_jsonl, corpus_root='.')
                preferred_csv_order = temp_eval.preferred_csv_order
                print(f"  使用rag_eval.py的preferred_csv_order")
            finally:
                os.unlink(temp_jsonl)
        except Exception as e:
            print(f"  警告: 无法从rag_eval.py导入列顺序: {e}")
    
        # 调整列顺序
        if preferred_csv_order:
            available_columns = df.columns.tolist()
            ordered_columns = [col for col in preferred_csv_order if col in available_columns]
            remaining_columns = [col for col in available_columns if col not in ordered_columns]
            final_columns = ordered_columns + remaining_columns
            df = df[final_columns]
        else:
            print(f"  使用DataFrame默认列顺序")
        
        # 删除旧的CSV文件（保留原始的，只删除recomputed版本）
        existing_csv_files = glob.glob(os.path.join(experiment_folder, "rag_evaluation_results_recomputed_*.csv"))
        for old_csv in existing_csv_files:
            try:
                os.remove(old_csv)
                print(f"  删除旧CSV: {os.path.basename(old_csv)}")
            except Exception as e:
                print(f"  警告: 无法删除 {old_csv}: {e}")
        
        # 保存CSV
        if output_csv_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_name = f"rag_evaluation_results_recomputed_{timestamp}.csv"
        
        # 保存到原experiment_folder中（不创建新文件夹）
        csv_path = os.path.join(experiment_folder, output_csv_name)
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ CSV已保存: {csv_path}")
        print(f"  样本数: {len(df)}")
        
        # 打印统计信息
        if 'overall_score' in df.columns:
            df['overall_score_num'] = pd.to_numeric(df['overall_score'], errors='coerce')
            overall_mean = df['overall_score_num'].mean()
            print(f"  平均overall_score: {overall_mean:.4f}")
            
            if 'question_type' in df.columns:
                type_averages = df.groupby('question_type')['overall_score_num'].mean()
                print(f"\n分类型平均分:")
                for q_type, avg_score in type_averages.items():
                    print(f"  {q_type}: {avg_score:.4f}")
    
    # 注意：Token统计已在compute_sample_metrics中实时处理，无需批量处理
    
    # 首先生成processed_json（如果不存在）
    print(f"\n生成processed_json文件...")
    processed_json_dir = os.path.join(experiment_folder, 'processed_json')
    samples_dir = os.path.join(experiment_folder, 'samples')
    
    # 检查是否需要生成processed_json
    if os.path.exists(samples_dir):
        # 检查processed_json是否已存在且完整
        if not os.path.exists(processed_json_dir):
            print(f"  processed_json目录不存在，开始生成...")
            process_intermediate(experiment_folder)
        else:
            # 检查processed_json文件数量是否与samples匹配
            sample_files = glob.glob(os.path.join(samples_dir, "*.json"))
            processed_files = glob.glob(os.path.join(processed_json_dir, "processed_*.json"))
            if len(processed_files) < len(sample_files):
                print(f"  processed_json文件不完整 ({len(processed_files)}/{len(sample_files)})，重新生成...")
                process_intermediate(experiment_folder)
            else:
                print(f"  processed_json已存在且完整 ({len(processed_files)} 个文件)")
    else:
        print(f"  警告: samples目录不存在，跳过processed_json生成")
    
    # 计算中间结果指标并合并
    print(f"\n计算中间结果指标...")
    updated_df = calculate_intermediate_metrics(experiment_folder)
    if updated_df is not None:
        print(f"✓ 中间结果指标已成功合并到CSV")
        return updated_df

    
    return updated_df


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python recompute_metrics.py <experiment_folder> [--no-llm-judge]")
        print("\n示例:")
        print("  python recompute_metrics.py ../results/experiment_20240101_120000")
        print("  python recompute_metrics.py ../results/experiment_20240101_120000 --no-llm-judge")
        sys.exit(1)
    
    experiment_folder = sys.argv[1]
    enable_llm_judge = '--no-llm-judge' not in sys.argv
    intermediate_only = '--intermediate-only' in sys.argv

    if not os.path.exists(experiment_folder):
        print(f"错误: 实验文件夹不存在: {experiment_folder}")
        sys.exit(1)
    
    # 重新计算指标
    
    df = recompute_all_metrics(
        experiment_folder,
        enable_llm_judge=enable_llm_judge,
        intermediate_only=intermediate_only
    )
    
    if df is not None:
        print(f"\n{'='*80}")
        print(f"✅ 指标重新计算完成！")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
