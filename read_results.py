#!/usr/bin/env python3
"""
RAG系统结果可视化工具
用于将低于特定分数阈值的样本以HTML格式可视化，便于调试分析

用法:
    python read_results.py <experiment_dir> --threshold <score> [--score-column <column_name>] [--baseline <baseline_csv>] [--samples <id1> <id2> ...]

参数:
    experiment_dir: 实验结果目录名称（如 experiment_20251224_104240）
    --threshold: 分数阈值（默认0.8）
    --score-column: 分数列名（默认 overall_score）
    --baseline: Baseline CSV文件路径（可选）
    --samples: 手动指定要显示的样本ID列表（可选，如果不指定则自动选择每类4个样本）

输出:
    - html/visualization_processed.html: processed_json的可视化
    - html/visualization_samples.html: samples的可视化
"""

import argparse
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict


# 定义不同关键词的颜色 - 按组设计，每组5个相近色
# 组1: 橙红系
GROUP1_COLORS = ["#f5939c", '#ff5722', "#fa1707", "#7f0300", "#623F41"]
# 组2: 蓝色系
GROUP2_COLORS = ["#008cff", "#2e70a5", "#aed4f4", "#3f4f60", "#224970"]
# 组3: 绿色系 (特别预算/管理/使用)
GROUP3_COLORS = ["#adeeaf", "#02f50e", "#048A0A", "#4c724c", "#107015"]
# 组4: 紫色系
GROUP4_COLORS = ["#eeaef9", "#d900ff", "#693E70", "#8f6f8f", "#932EA5"]
# 组5: 黄色系 (柯文哲)
GROUP5_COLORS = ["#F9F0A2", "#FFC400", "#A19A85", "#9D973D", "#615716"]
# 组6: 青色系
GROUP6_COLORS = ['#00bcd4', '#26c6da', '#4dd0e1', "#d4ebee", "#1C5E66"]

ALL_GROUP_COLORS = [GROUP1_COLORS, GROUP2_COLORS, GROUP3_COLORS, GROUP4_COLORS, GROUP5_COLORS, GROUP6_COLORS]


def build_keyword_colors(keywords):
    """
    统一的关键词颜色分配函数
    Args:
        keywords: [[group1_kw1, group1_kw2], [group2_kw1], ...] 或 [kw1, kw2, ...]
    Returns:
        Dict[str, str]: 关键词到颜色的映射
    """
    keyword_colors = {}
    
    if keywords and isinstance(keywords, list) and keywords and isinstance(keywords[0], list):
        # 多组关键词
        for group_idx, group in enumerate(keywords):
            group_colors = ALL_GROUP_COLORS[group_idx % len(ALL_GROUP_COLORS)]
            for kw_idx, kw in enumerate(group):
                keyword_colors[kw] = group_colors[kw_idx % len(group_colors)]
    else:
        # 单层列表
        for i, kw in enumerate(keywords):
            keyword_colors[kw] = GROUP1_COLORS[i % len(GROUP1_COLORS)]
    
    return keyword_colors


class RAGResultsVisualizer:
    """RAG结果可视化器"""
    
    def __init__(self, results_dir: Path, threshold: float, score_column: str = "overall_score", baseline_csv: Optional[Path] = None, manual_samples: Optional[List[str]] = None):
        self.results_dir = results_dir
        self.threshold = threshold
        self.score_column = score_column
        self.processed_dir = results_dir / "processed_json"
        self.samples_dir = results_dir / "samples"
        self.html_dir = results_dir / "html"
        self.html_dir.mkdir(exist_ok=True)
        self.baseline_csv = baseline_csv
        self.manual_samples = manual_samples
        
        self.csv_data = self._load_csv_data()
        self.scores = {row['question_id']: row for row in self.csv_data}
        self.baseline_data = self._load_baseline_data() if baseline_csv else None
        self.stats = self._compute_statistics()
        
    def _load_csv_data(self) -> List[Dict]:
        """从CSV文件加载所有数据"""
        csv_files = list(self.results_dir.glob("rag_evaluation_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"未找到评分CSV文件在 {self.results_dir}")
        
        csv_file = csv_files[0]
        data = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值列
                for key in row:
                    if key in ['metric_with_token', 'perf_score', 'overall_score', 'llm_judge_score',
                              'rejection_recall', 'ndcg', 'ncg', 'mrr', 'pre_search_ndcg', 
                              'pre_search_mrr', 'pre_search_ncg']:
                        try:
                            row[key] = float(row[key]) if row[key] else 0.0
                        except:
                            row[key] = 0.0
                data.append(row)
        
        return data
    
    def _load_baseline_data(self) -> Optional[List[Dict]]:
        """加载Baseline CSV数据"""
        if not self.baseline_csv or not self.baseline_csv.exists():
            return None
        
        data = []
        try:
            with open(self.baseline_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 转换数值列
                    for key in row:
                        if key in ['Score', 'score']:
                            try:
                                row[key] = float(row[key]) if row[key] else 0.0
                            except:
                                row[key] = 0.0
                    data.append(row)
            return data
        except Exception as e:
            print(f"警告: 无法加载Baseline数据: {e}")
            return None
    
    def _compute_statistics(self) -> Dict:
        """计算统计信息"""
        stats = {
            'overall': {},
            'by_type': defaultdict(lambda: {}),
            'by_difficulty': defaultdict(lambda: {})
        }
        
        stat_columns = ['llm_judge_score']
        
        # 整体统计
        for col in stat_columns:
            values = [row[col] for row in self.csv_data if isinstance(row.get(col), (int, float))]
            if values:
                values_sorted = sorted(values)
                stats['overall'][col] = {
                    'mean': sum(values) / len(values),
                    'median': values_sorted[len(values) // 2],
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        # 按question_type分类统计
        type_groups = defaultdict(list)
        for row in self.csv_data:
            q_type = row.get('question_type', 'Unknown')
            type_groups[q_type].append(row)
        
        for q_type, rows in type_groups.items():
            for col in stat_columns:
                values = [row[col] for row in rows if isinstance(row.get(col), (int, float))]
                if values:
                    values_sorted = sorted(values)
                    stats['by_type'][q_type][col] = {
                        'mean': sum(values) / len(values),
                        'median': values_sorted[len(values) // 2],
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
        
        # 按difficulty分类统计
        difficulty_groups = defaultdict(list)
        for row in self.csv_data:
            difficulty = row.get('difficulty', row.get('question_type', 'Unknown'))  # 如果没有difficulty，使用question_type
            difficulty_groups[difficulty].append(row)
        
        for difficulty, rows in difficulty_groups.items():
            for col in stat_columns:
                values = [row[col] for row in rows if isinstance(row.get(col), (int, float))]
                if values:
                    values_sorted = sorted(values)
                    stats['by_difficulty'][difficulty][col] = {
                        'mean': sum(values) / len(values),
                        'median': values_sorted[len(values) // 2],
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
        
        return stats
    
    def get_low_score_samples(self) -> List[str]:
        """获取低于阈值的样本ID列表"""
        low_samples = []
        for row in self.csv_data:
            qid = row['question_id']
            score = row.get(self.score_column, 0.0)
            if isinstance(score, (int, float)) and score < self.threshold:
                low_samples.append(qid)
        return low_samples
    
    def get_selected_samples_by_difficulty(self, num_per_category: int = 2) -> List[str]:
        """对每个difficulty/question_type类别选取指定数量的llm_judge_score=0和1的样本
        
        Args:
            num_per_category: 每个类别每种分数选取的样本数量（默认2个）
        """
        difficulty_groups = defaultdict(lambda: {'zero': [], 'one': []})
        
        for row in self.csv_data:
            qid = row['question_id']
            difficulty = row.get('difficulty', row.get('question_type', 'Unknown'))
            score = row.get('llm_judge_score', 0.0)
            
            if isinstance(score, (int, float)):
                if score == 0.0:
                    difficulty_groups[difficulty]['zero'].append(qid)
                elif score == 1.0:
                    difficulty_groups[difficulty]['one'].append(qid)
        
        selected = []
        for difficulty, groups in difficulty_groups.items():
            # 每个difficulty选num_per_category个score=0和num_per_category个score=1
            selected.extend(groups['zero'][:num_per_category])
            selected.extend(groups['one'][:num_per_category])
        
        return selected
    
    def generate_visualizations(self):
        """生成两个HTML可视化文件"""
        # 使用manual_samples或自动选取逻辑
        if self.manual_samples:
            low_score_samples = self.manual_samples
            print(f"使用手动指定的 {len(low_score_samples)} 个样本")
        else:
            # 默认每类选2个，总共4个（2个score=0 + 2个score=1）
            low_score_samples = self.get_selected_samples_by_difficulty(num_per_category=2)
            print(f"自动选取了 {len(low_score_samples)} 个样本")
        # 读取JSONL文件并保存低分样本
        input_jsonl = self.results_dir / "data.jsonl"
        output_jsonl = self.html_dir / "low_score_samples.jsonl"
        
        if not input_jsonl.exists():
            print(f"未找到JSONL文件: {input_jsonl}")
        else:
            with open(input_jsonl, 'r', encoding='utf-8') as infile, open(output_jsonl, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    sample = json.loads(line)
                    if sample.get('question_id') in low_score_samples:
                        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        
            print(f"低分样本已保存到: {output_jsonl}")
        if not low_score_samples:
            print(f"没有找到分数低于 {self.threshold} 的样本")
            return
        
        print(f"找到 {len(low_score_samples)} 个低于阈值的样本")
        
        # 生成processed_json可视化
        processed_html = self._generate_processed_html(low_score_samples)
        output_processed = self.html_dir / "visualization_processed.html"
        with open(output_processed, 'w', encoding='utf-8') as f:
            f.write(processed_html)
        print(f"已生成: {output_processed}")
        
        # 生成samples可视化
        samples_html = self._generate_samples_html(low_score_samples)
        output_samples = self.html_dir / "visualization_samples.html"
        with open(output_samples, 'w', encoding='utf-8') as f:
            f.write(samples_html)
        print(f"已生成: {output_samples}")
    
    def _generate_statistics_html(self) -> str:
        """生成统计信息HTML"""
        html_parts = ['<div class="statistics">']
        
        # Baseline对比（如果有）
        if self.baseline_data:
            html_parts.append(self._generate_baseline_comparison())
            html_parts.append('<hr>')
        html_parts.append('<h2>📊 统计信息</h2>')
        
        # 整体统计 - 横向排列
        html_parts.append('<div class="stat-section">')
        html_parts.append('<h3>整体统计 (Total Samples: {})</h3>'.format(len(self.csv_data)))
        html_parts.append('<div class="stat-table-container">')
        html_parts.append('<table class="stat-table">')
        
        # 表头：第一列是空的，后面是所有指标名
        html_parts.append('<thead><tr><th></th>')
        for col in self.stats['overall'].keys():
            html_parts.append(f'<th>{col}</th>')
        html_parts.append('</tr></thead>')
        
        # Mean行
        html_parts.append('<tbody><tr><td>Mean</td>')
        for col, values in self.stats['overall'].items():
            html_parts.append(f'<td>{values["mean"]:.4f}</td>')
        html_parts.append('</tr>')
        
        # Median行
        html_parts.append('<tr><td>Median</td>')
        for col, values in self.stats['overall'].items():
            html_parts.append(f'<td>{values["median"]:.4f}</td>')
        html_parts.append('</tr>')
        
        html_parts.append('</tbody></table></div></div>')
        
        # 按类型统计 - 可展开收缩
        if self.stats['by_type']:
            html_parts.append('<div class="stat-section">')
            html_parts.append('<h3>按Question Type分类统计</h3>')
            
            for idx, (q_type, type_stats) in enumerate(self.stats['by_type'].items()):
                type_id = f"type_{idx}"
                html_parts.append(f'<div class="type-section">')
                html_parts.append(f'<div class="type-header" onclick="toggleType(\"{type_id}\")">')
                html_parts.append(f'Type: {q_type} ▼')
                html_parts.append(f'</div>')
                html_parts.append(f'<div class="type-body" id="{type_id}">')
                
                html_parts.append('<div class="stat-table-container">')
                html_parts.append('<table class="stat-table">')
                
                # 表头
                html_parts.append('<thead><tr><th></th>')
                for col in type_stats.keys():
                    html_parts.append(f'<th>{col}</th>')
                html_parts.append('</tr></thead>')
                
                # Mean行
                html_parts.append('<tbody><tr><td>Mean</td>')
                for col, values in type_stats.items():
                    html_parts.append(f'<td>{values["mean"]:.4f}</td>')
                html_parts.append('</tr>')
                
                # Median行
                html_parts.append('<tr><td>Median</td>')
                for col, values in type_stats.items():
                    html_parts.append(f'<td>{values["median"]:.4f}</td>')
                html_parts.append('</tr>')
                
                html_parts.append('</tbody></table></div>')
                html_parts.append('</div>')  # type-body
                html_parts.append('</div>')  # type-section
            
            html_parts.append('</div>')
        
        # 按difficulty统计 - 可展开收缩
        if self.stats['by_difficulty']:
            html_parts.append('<div class="stat-section">')
            html_parts.append('<h3>按Difficulty分类统计</h3>')
            
            for idx, (difficulty, diff_stats) in enumerate(self.stats['by_difficulty'].items()):
                diff_id = f"diff_{idx}"
                html_parts.append(f'<div class="type-section">')
                html_parts.append(f'<div class="type-header" onclick="toggleType(\"{diff_id}\")">')
                html_parts.append(f'Difficulty: {difficulty} ▼')
                html_parts.append(f'</div>')
                html_parts.append(f'<div class="type-body" id="{diff_id}">')
                
                html_parts.append('<div class="stat-table-container">')
                html_parts.append('<table class="stat-table">')
                
                # 表头
                html_parts.append('<thead><tr><th></th>')
                for col in diff_stats.keys():
                    html_parts.append(f'<th>{col}</th>')
                html_parts.append('</tr></thead>')
                
                # Mean行
                html_parts.append('<tbody><tr><td>Mean</td>')
                for col, values in diff_stats.items():
                    html_parts.append(f'<td>{values["mean"]:.4f}</td>')
                html_parts.append('</tr>')
                
                # Median行
                html_parts.append('<tr><td>Median</td>')
                for col, values in diff_stats.items():
                    html_parts.append(f'<td>{values["median"]:.4f}</td>')
                html_parts.append('</tr>')
                
                html_parts.append('</tbody></table></div>')
                html_parts.append('</div>')  # type-body
                html_parts.append('</div>')  # type-section
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _generate_baseline_comparison(self) -> str:
        """生成Baseline对比HTML"""
        html_parts = ['<div class="baseline-comparison">']
        html_parts.append('<h2>📊 与Baseline对比</h2>')
        
        # 计算我们的llm_judge_score均值
        our_scores = [row['llm_judge_score'] for row in self.csv_data if isinstance(row.get('llm_judge_score'), (int, float))]
        our_mean = sum(our_scores) / len(our_scores) if our_scores else 0
        
        # 计算Baseline的Score均值
        baseline_scores = []
        for row in self.baseline_data:
            score = row.get('Score') or row.get('score')
            if isinstance(score, (int, float)):
                baseline_scores.append(score)
        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
        
        # 显示均值对比
        html_parts.append('<div class="mean-comparison">')
        html_parts.append(f'<div class="mean-item"><strong>我们的llm_judge_score均值:</strong> <span class="score-value">{our_mean:.4f}</span></div>')
        html_parts.append(f'<div class="mean-item"><strong>Baseline Score均值:</strong> <span class="score-value">{baseline_mean:.4f}</span></div>')
        html_parts.append('</div>')
        
        # 按difficulty对比的柱状图
        html_parts.append(self._generate_difficulty_chart())
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _generate_difficulty_chart(self) -> str:
        """生成按difficulty分类的对比柱状图"""
        # 计算我们的数据按difficulty分类
        our_difficulty_scores = defaultdict(list)
        for row in self.csv_data:
            difficulty = row.get('difficulty', row.get('question_type', 'Unknown'))
            score = row.get('llm_judge_score')
            if isinstance(score, (int, float)):
                our_difficulty_scores[difficulty].append(score)
        
        our_means = {d: sum(scores) / len(scores) for d, scores in our_difficulty_scores.items()}
        
        # 计算Baseline数据按difficulty分类
        baseline_difficulty_scores = defaultdict(list)
        if self.baseline_data:
            for row in self.baseline_data:
                difficulty = row.get('difficulty', row.get('question_type', 'Unknown'))
                score = row.get('Score') or row.get('score')
                if isinstance(score, (int, float)):
                    baseline_difficulty_scores[difficulty].append(score)
        
        baseline_means = {d: sum(scores) / len(scores) for d, scores in baseline_difficulty_scores.items()}
        
        # 生成图表HTML
        html_parts = ['<div class="chart-container">']
        html_parts.append('<h3>按Difficulty分类对比</h3>')
        html_parts.append('<div class="bar-chart">')
        
        all_difficulties = sorted(set(list(our_means.keys()) + list(baseline_means.keys())))
        
        for difficulty in all_difficulties:
            our_score = our_means.get(difficulty, 0)
            baseline_score = baseline_means.get(difficulty, 0)
            
            html_parts.append(f'<div class="chart-row">')
            html_parts.append(f'<div class="chart-label">{difficulty}</div>')
            html_parts.append(f'<div class="chart-bars">')
            html_parts.append(f'<div class="bar our-bar" style="width: {our_score * 100}%;"><span class="bar-label">我们: {our_score:.3f}</span></div>')
            html_parts.append(f'<div class="bar baseline-bar" style="width: {baseline_score * 100}%;"><span class="bar-label">Baseline: {baseline_score:.3f}</span></div>')
            html_parts.append(f'</div>')
            html_parts.append(f'</div>')
        
        html_parts.append('</div>')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _generate_processed_html(self, sample_ids: List[str]) -> str:
        """生成processed_json的HTML可视化"""
        html_parts = [self._get_html_header("Processed JSON 可视化")]
        html_parts.append(self._generate_statistics_html())
        html_parts.append('<hr>')
        html_parts.append('<h2>低分样本详情</h2>')
        
        for sample_id in sample_ids:
            json_file = self.processed_dir / f"processed_{sample_id}.json"
            if not json_file.exists():
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            row_data = self.scores.get(sample_id, {})
            html_parts.append(self._render_processed_sample(sample_id, row_data, data))
        
        html_parts.append(self._get_html_footer())
        return '\n'.join(html_parts)
    
    def _generate_samples_html(self, sample_ids: List[str]) -> str:
        """生成samples的HTML可视化"""
        html_parts = [self._get_html_header("Samples 可视化")]
        html_parts.append(self._generate_statistics_html())
        html_parts.append('<hr>')
        html_parts.append('<h2>低分样本详情</h2>')
        
        for sample_id in sample_ids:
            json_file = self.samples_dir / f"sample_{sample_id}.json"
            if not json_file.exists():
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            row_data = self.scores.get(sample_id, {})
            html_parts.append(self._render_sample(sample_id, row_data, data))
        
        html_parts.append(self._get_html_footer())
        return '\n'.join(html_parts)
    
    def _get_html_header(self, title: str) -> str:
        """生成HTML头部"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .statistics {{
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-section {{
            margin-bottom: 20px;
        }}
        .stat-table-container {{
            overflow-x: auto;
            margin-bottom: 15px;
        }}
        .stat-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .stat-table th, .stat-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .stat-table th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        .stat-table tr:hover {{
            background: #f5f5f5;
        }}
        .sample {{
            background: white;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .sample-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            user-select: none;
        }}
        .sample-header:hover {{
            opacity: 0.95;
        }}
        .sample-header h2 {{
            margin: 0 0 10px 0;
            display: inline-block;
        }}
        .expand-icon {{
            float: right;
            font-size: 20px;
            transition: transform 0.3s;
        }}
        .expand-icon.expanded {{
            transform: rotate(180deg);
        }}
        .score-preview {{
            font-size: 13px;
            margin-top: 5px;
        }}
        .score-badge {{
            display: inline-block;
            padding: 3px 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
            margin-right: 10px;
            font-size: 12px;
        }}
        .baseline-comparison {{
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .mean-comparison {{
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
            font-size: 16px;
        }}
        .mean-item {{
            padding: 10px 15px;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .score-value {{
            color: #667eea;
            font-size: 20px;
            font-weight: bold;
        }}
        .chart-container {{
            margin-top: 20px;
        }}
        .bar-chart {{
            margin-top: 15px;
        }}
        .chart-row {{
            margin-bottom: 15px;
        }}
        .chart-label {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #495057;
        }}
        .chart-bars {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .bar {{
            height: 30px;
            display: flex;
            align-items: center;
            padding-left: 10px;
            border-radius: 3px;
            transition: all 0.3s;
            min-width: 80px;
        }}
        .our-bar {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }}
        .baseline-bar {{
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        }}
        .bar-label {{
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        .sample-body {{
            padding: 20px;
            display: none;
        }}
        .sample-body.expanded {{
            display: block;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .content {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .task {{
            margin-bottom: 25px;
            padding: 15px;
            background: #fff9e6;
            border-radius: 5px;
            border: 1px solid #ffe066;
        }}
        .task-header {{
            cursor: pointer;
            user-select: none;
        }}
        .task-header:hover {{
            background: rgba(255,224,102,0.3);
            margin: -15px;
            padding: 15px;
            border-radius: 5px;
        }}
        .task-title {{
            font-weight: bold;
            color: #d4860f;
            margin-bottom: 10px;
            font-size: 15px;
        }}
        .task-body {{
            margin-top: 10px;
        }}
        .task-body.collapsed {{
            display: none;
        }}
        .message {{
            margin: 10px 0;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 3px solid #ddd;
        }}
        .message.tool-call {{
            background: #fff3cd;
            border-left-color: #ffc107;
        }}
        .message.tool-result {{
            background: #d1ecf1;
            border-left-color: #17a2b8;
        }}
        .message-role {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
            font-size: 13px;
        }}
        .tool-call .message-role {{
            color: #856404;
        }}
        .tool-result .message-role {{
            color: #0c5460;
        }}
        .tool-name {{
            display: inline-block;
            padding: 2px 8px;
            background: #fd7e14;
            color: white;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 5px;
        }}
        .grep-info {{
            background: #e7f3ff;
            padding: 10px;
            border-radius: 5px;
            margin-top: 8px;
            font-size: 13px;
        }}
        .grep-logic {{
            display: inline-block;
            padding: 2px 6px;
            background: #007bff;
            color: white;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .keyword {{
            padding: 1px 4px;
            border-radius: 2px;
            font-weight: bold;
        }}
        .keyword-group {{
            margin: 5px 0;
            padding-left: 15px;
        }}
        .match-count {{
            color: #28a745;
            font-weight: bold;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .info-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }}
        .info-label {{
            font-weight: bold;
            color: #6c757d;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        .info-value {{
            color: #212529;
            word-break: break-word;
            overflow-wrap: break-word;
        }}
        .info-value.long-text {{
            font-size: 10px;
            line-height: 1.3;
        }}
        .csv-section {{
            margin-bottom: 20px;
        }}
        .csv-header {{
            background: #e9ecef;
            padding: 10px 15px;
            cursor: pointer;
            user-select: none;
            border-radius: 5px;
            font-weight: bold;
            color: #495057;
        }}
        .csv-header:hover {{
            background: #dee2e6;
        }}
        .csv-body {{
            margin-top: 10px;
            display: none;
        }}
        .csv-body.expanded {{
            display: block;
        }}
        .toggle-btn {{
            cursor: pointer;
            color: #667eea;
            text-decoration: underline;
            font-size: 14px;
            margin-top: 10px;
            display: inline-block;
        }}
        .collapsible {{
            display: none;
        }}
        .collapsible.show {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>分数阈值: {self.threshold} ({self.score_column})</p>
"""
    
    def _get_html_footer(self) -> str:
        """生成HTML尾部"""
        return """
    </div>
    <script>
        function toggleSample(id) {
            const body = document.getElementById('body_' + id);
            const icon = document.getElementById('icon_' + id);
            body.classList.toggle('expanded');
            icon.classList.toggle('expanded');
        }
        
        function toggleTask(id) {
            const body = document.getElementById('task_body_' + id);
            body.classList.toggle('collapsed');
        }
        
        function toggleContent(id) {
            const elem = document.getElementById(id);
            elem.classList.toggle('show');
        }
        
        function toggleType(id) {
            const body = document.getElementById(id);
            body.classList.toggle('expanded');
        }
        
        function toggleCSV(id) {
            const body = document.getElementById(id);
            body.classList.toggle('expanded');
        }
    </script>
</body>
</html>
"""
    
    def _get_score_preview(self, row_data: Dict) -> str:
        """生成分数预览 - 只显示llm_judge_score和difficulty"""
        llm_score = row_data.get('llm_judge_score', 0.0)
        difficulty = row_data.get('difficulty', row_data.get('question_type', 'Unknown'))
        
        # 根据llm_judge_score设置颜色
        color = '#28a745' if llm_score == 1.0 else '#dc3545'
        
        # 查找对应的Baseline score
        baseline_score = 'N/A'
        if self.baseline_data:
            qid = row_data.get('question_id')
            for baseline_row in self.baseline_data:
                if baseline_row.get('question_id') == qid:
                    baseline_score = baseline_row.get('Score', baseline_row.get('score', 'N/A'))
                    if isinstance(baseline_score, float):
                        baseline_score = f'{baseline_score:.3f}'
                    break
        
        badges = [
            f'<span class="score-badge" style="background: {color};">llm_judge_score: {llm_score:.3f}</span>',
            f'<span class="score-badge">Baseline: {baseline_score}</span>',
            f'<span class="score-badge">Difficulty: {difficulty}</span>'
        ]
        return ''.join(badges)
    
    def _render_processed_sample(self, sample_id: str, row_data: Dict, data: Dict) -> str:
        """渲染processed样本"""
        score = row_data.get(self.score_column, 0.0)
        score_preview = self._get_score_preview(row_data)
        
        parts = [f"""
        <div class="sample">
            <div class="sample-header" onclick="toggleSample('{sample_id}')">
                <h2>{sample_id}</h2>
                <span class="expand-icon" id="icon_{sample_id}">▼</span>
                <div class="score-preview">{score_preview}</div>
            </div>
            <div class="sample-body" id="body_{sample_id}">
        """]
        
        # 不显示CSV数据
        
        # 渲染每个task
        for task_idx, (task_name, task_data) in enumerate(data.items()):
            if not isinstance(task_data, dict):
                continue
            
            task_id = f"{sample_id}_task_{task_idx}"
            parts.append(f'<div class="task">')
            parts.append(f'<div class="task-header" onclick="toggleTask(\'{task_id}\')">')
            parts.append(f'<div class="task-title">📋 {task_name} ▼</div>')
            parts.append(f'</div>')
            parts.append(f'<div class="task-body" id="task_body_{task_id}">')
            
            # Query
            if 'query' in task_data:
                parts.append(f'<div class="section">')
                parts.append(f'<div class="section-title">用户查询:</div>')
                parts.append(f'<div class="content">{self._escape_html(task_data["query"])}</div>')
                parts.append(f'</div>')
            
            # Response
            if 'response' in task_data:
                parts.append(f'<div class="section">')
                parts.append(f'<div class="section-title">搜索结果:</div>')
                parts.append(f'<div class="content">{self._format_text(task_data["response"])}</div>')
                parts.append(f'</div>')
            
            # Tools used
            if 'tools_used' in task_data:
                parts.append(self._render_tools_used(task_data['tools_used']))
            
            # Token info
            if any(k in task_data for k in ['input_tokens', 'output_tokens', 'tokens']):
                parts.append(f'<div class="info-grid">')
                if 'input_tokens' in task_data:
                    parts.append(f'<div class="info-item"><div class="info-label">Input Tokens</div><div class="info-value">{task_data["input_tokens"]}</div></div>')
                if 'output_tokens' in task_data:
                    parts.append(f'<div class="info-item"><div class="info-label">Output Tokens</div><div class="info-value">{task_data["output_tokens"]}</div></div>')
                if 'tokens' in task_data:
                    parts.append(f'<div class="info-item"><div class="info-label">Total Tokens</div><div class="info-value">{task_data["tokens"]}</div></div>')
                parts.append(f'</div>')
            
            parts.append(f'</div>')  # task-body
            parts.append(f'</div>')  # task
        
        parts.append('</div></div>')  # sample-body, sample
        return '\n'.join(parts)
    
    def _render_sample(self, sample_id: str, row_data: Dict, data: Dict) -> str:
        """渲染完整样本"""
        score = row_data.get(self.score_column, 0.0)
        score_preview = self._get_score_preview(row_data)
        
        parts = [f"""
        <div class="sample">
            <div class="sample-header" onclick="toggleSample('{sample_id}')">
                <h2>{sample_id}</h2>
                <span class="expand-icon" id="icon_{sample_id}">▼</span>
                <div class="score-preview">{score_preview}</div>
            </div>
            <div class="sample-body" id="body_{sample_id}">
        """]
        
        # 不显示CSV数据
        
        # Sample info
        if 'sample_info' in data:
            info = data['sample_info']
            parts.append(f'<div class="section">')
            parts.append(f'<div class="section-title">样本信息:</div>')
            parts.append(f'<div class="info-grid">')
            for key, value in info.items():
                if key not in ['expected_context_ids']:
                    parts.append(f'<div class="info-item">')
                    parts.append(f'<div class="info-label">{key}</div>')
                    parts.append(f'<div class="info-value">{self._escape_html(str(value))}</div>')
                    parts.append(f'</div>')
            parts.append(f'</div>')
            parts.append(f'</div>')
        
        # Response
        if 'response' in data:
            parts.append(f'<div class="section">')
            parts.append(f'<div class="section-title">最终回答:</div>')
            parts.append(f'<div class="content">{self._format_text(data["response"])}</div>')
            parts.append(f'</div>')
        
        # Messages
        if 'messages' in data:
            collapse_id = f"messages_{sample_id}"
            parts.append(f'<div class="section">')
            parts.append(f'<div class="section-title">消息历史 ({len(data["messages"])} 条消息):</div>')
            parts.append(f'<span class="toggle-btn" onclick="toggleContent(\'{collapse_id}\')">展开/收起</span>')
            parts.append(f'<div id="{collapse_id}" class="collapsible">')
            parts.append(self._render_messages(data['messages']))
            parts.append(f'</div>')
            parts.append(f'</div>')
        
        parts.append('</div></div>')  # sample-body, sample
        return '\n'.join(parts)
    
    def _render_messages(self, messages: List[Dict]) -> str:
        """渲染消息列表"""
        parts = []
        
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            msg_type = msg.get('type', 'text')
            
            css_class = 'message'
            if msg_type == 'tool_call':
                css_class = 'message tool-call'
            elif msg_type == 'tool_result':
                css_class = 'message tool-result'
            
            parts.append(f'<div class="{css_class}">')
            parts.append(f'<div class="message-role">{role} - {msg_type}')
            
            if msg_type == 'tool_call' and 'tool_call' in msg:
                tool_call = msg['tool_call']
                tool_name = tool_call.get('name', 'unknown')
                parts.append(f'<span class="tool-name">{tool_name}</span>')
            
            parts.append(f'</div>')
            
            if 'content' in msg:
                parts.append(f'<pre>{self._format_text(msg["content"])}</pre>')
            
            if msg_type == 'tool_call' and 'tool_call' in msg:
                tool_call = msg['tool_call']
                if 'arguments' in tool_call:
                    args = tool_call['arguments']
                    
                    if tool_call.get('name') == 'grep_files':
                        parts.append(self._render_grep_call(args))
                    else:
                        parts.append(f'<pre>{self._escape_html(json.dumps(args, ensure_ascii=False, indent=2))}</pre>')
            
            if msg_type == 'tool_result' and 'tool_result' in msg:
                result = msg['tool_result'].get('result', '')
                result_str = str(result)
                
                # 检查是否是grep_files的结果（包含FILE标记或SUMMARY标记）
                if '# SUMMARY' in result_str or '# FILE:' in result_str or '# File:' in result_str or '[KEYWORD:' in result_str:
                    # 查找对应的tool_call获取keywords
                    keywords = None
                    if i > 0:
                        for j in range(i-1, -1, -1):
                            if messages[j].get('type') == 'tool_call':
                                tool_call = messages[j].get('tool_call', {})
                                if tool_call.get('name') == 'grep_files':
                                    keywords = tool_call.get('arguments', {}).get('keywords')
                                    break
                    parts.append(self._render_grep_result_with_keywords(result, keywords))
                else:
                    parts.append(f'<pre>{self._format_text(result)}</pre>')
            
            parts.append(f'</div>')
        
        return '\n'.join(parts)
    
    def _render_tools_used(self, tools: Dict) -> str:
        """渲染工具使用信息"""
        parts = [f'<div class="section">']
        parts.append(f'<div class="section-title">使用的工具:</div>')
        
        for tool_key, tool_data in tools.items():
            if not isinstance(tool_data, dict):
                continue
            
            tool_name = tool_data.get('name', 'unknown')
            parts.append(f'<div style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px;">')
            parts.append(f'<strong>🔧 {tool_name}</strong>')
            
            if 'arguments' in tool_data:
                args = tool_data['arguments']
                
                if tool_name == 'grep_files':
                    parts.append(self._render_grep_call(args))
                else:
                    parts.append(f'<pre style="margin-top: 5px;">{self._escape_html(json.dumps(args, ensure_ascii=False, indent=2))}</pre>')
            
            if 'result' in tool_data:
                result = tool_data['result']
                result_str = str(result)
                
                # 检查是否是grep_files的结果（包含FILE标记或SUMMARY标记）
                if '# SUMMARY' in result_str or '# FILE:' in result_str or '# File:' in result_str or '[KEYWORD:' in result_str:
                    keywords = tool_data.get('arguments', {}).get('keywords') if tool_name == 'grep_files' else None
                    parts.append(self._render_grep_result_with_keywords(result, keywords))
                else:
                    collapse_id = f"result_{id(tool_data)}"
                    parts.append(f'<span class="toggle-btn" onclick="toggleContent(\'{collapse_id}\')">查看结果</span>')
                    parts.append(f'<div id="{collapse_id}" class="collapsible">')
                    parts.append(f'<pre style="margin-top: 5px;">{self._format_text(result)}</pre>')
                    parts.append(f'</div>')
            
            parts.append(f'</div>')
        
        parts.append(f'</div>')
        return '\n'.join(parts)
    
    def _render_grep_call(self, args: Dict) -> str:
        """渲染grep_files调用参数"""
        parts = ['<div class="grep-info">']
        
        if 'keywords' in args:
            keywords = args['keywords']
            # 使用统一的颜色分配函数
            keyword_colors = build_keyword_colors(keywords)
            
            if keywords and isinstance(keywords[0], list):
                parts.append('<div><span class="grep-logic">AND</span> 逻辑 (所有组都必须匹配):</div>')
                for group_idx, group in enumerate(keywords):
                    group_keywords = []
                    for kw in group:
                        color = keyword_colors.get(kw, GROUP1_COLORS[0])
                        group_keywords.append(f'<span class="keyword" style="background:{color};">{self._escape_html(kw)}</span>')
                    parts.append(f'<div class="keyword-group">组 {group_idx + 1} (OR): {" | ".join(group_keywords)}</div>')
            else:
                parts.append('<div><span class="grep-logic">OR</span> 逻辑:</div>')
                keywords_html = []
                for kw in keywords:
                    color = keyword_colors.get(kw, GROUP1_COLORS[0])
                    keywords_html.append(f'<span class="keyword" style="background:{color};">{self._escape_html(kw)}</span>')
                parts.append(f'<div class="keyword-group">{" | ".join(keywords_html)}</div>')
        
        if 'reason_refine' in args:
            parts.append(f'<div style="margin-top: 8px;"><strong>原因:</strong> {self._escape_html(args["reason_refine"])}</div>')
        
        parts.append('</div>')
        return '\n'.join(parts)
    
    def _render_grep_result_with_keywords(self, result: str, keywords: Optional[List] = None) -> str:
        """渲染grep_files结果，从KEYWORD标记或Preview中提取并高亮实际命中的关键词"""
        result_str = str(result)
        
        parts = ['<div class="grep-info" style="margin-top: 8px;">']
        
        if 'match ALL' in result_str:
            parts.append('<div><span class="grep-logic">AND</span> 所有关键词组都匹配</div>')
        elif 'union' in result_str.lower() or 'No intersection' in result_str:
            parts.append('<div><span class="grep-logic">UNION</span> 关键词组的并集（无交集）</div>')
        
        summary_match = re.search(r'# SUMMARY.*?(\d+) files?', result_str)
        if summary_match:
            parts.append(f'<div>找到 <span class="match-count">{summary_match.group(1)}</span> 个文件匹配</div>')
        
        parts.append('</div>')
        
        # 优先尝试从结果中提取[KEYWORD: xxx]标记（新格式）
        keyword_pattern = r'\[KEYWORD: (.+?)\]'
        keyword_markers = re.findall(keyword_pattern, result_str)
        
        if keyword_markers:
            # 有KEYWORD标记，使用它们
            # KEYWORD标记中的内容可能是逗号分隔的多个关键词，需要拆分
            all_keywords_from_markers = []
            for marker in keyword_markers:
                # 按中文逗号或英文逗号分隔
                kws = [k.strip() for k in marker.replace('，', ',').split(',')]
                all_keywords_from_markers.extend(kws)
            
            # 使用统一的颜色分配函数
            if keywords:
                keyword_colors = build_keyword_colors(keywords)
            else:
                # 没有keywords参数，使用第一组颜色
                keyword_colors = {}
                for i, kw in enumerate(set(all_keywords_from_markers)):
                    keyword_colors[kw] = GROUP1_COLORS[i % len(GROUP1_COLORS)]
            
            # 高亮KEYWORD标记和文本中的关键词
            # 策略：先按KEYWORD标记分段，分别处理每段，最后组合
            
            # 分段：将文本按KEYWORD标记分成多段
            parts_to_process = []
            last_pos = 0
            for match in re.finditer(r'\[KEYWORD: ([^\]]+)\]', result_str):
                # 添加KEYWORD标记之前的普通文本
                if match.start() > last_pos:
                    parts_to_process.append(('text', result_str[last_pos:match.start()]))
                # 添加KEYWORD标记本身
                parts_to_process.append(('keyword_marker', match.group(1)))  # 只保存括号内的内容
                last_pos = match.end()
            # 添加最后一段文本
            if last_pos < len(result_str):
                parts_to_process.append(('text', result_str[last_pos:]))
            
            # 处理每一段
            final_parts = []
            for part_type, part_content in parts_to_process:
                if part_type == 'keyword_marker':
                    # 处理KEYWORD标记：高亮其中的关键词
                    marker_parts = []
                    kw_list = [kw.strip() for kw in part_content.replace('，', ',').split(',')]
                    for kw in kw_list:
                        color = keyword_colors.get(kw, GROUP1_COLORS[0])
                        marker_parts.append(f'<span class="keyword" style="background:{color};">{self._escape_html(kw)}</span>')
                    # 使用原始分隔符
                    separator = '，' if '，' in part_content else ','
                    marker_html = '[KEYWORD: ' + separator.join(marker_parts) + ']'
                    final_parts.append(marker_html)
                else:
                    # 普通文本：先转义HTML，然后高亮关键词
                    escaped_text = self._escape_html(part_content)
                    # 在普通文本中高亮关键词
                    for kw, color in keyword_colors.items():
                        escaped_kw = self._escape_html(kw)
                        # 使用简单替换，但要避免替换已经在span标签中的内容
                        # 由于我们是按顺序处理，只需要避免替换已经被高亮的关键词
                        temp_text = escaped_text
                        escaped_text = ''
                        while temp_text:
                            # 查找下一个关键词位置
                            pos = temp_text.find(escaped_kw)
                            if pos == -1:
                                escaped_text += temp_text
                                break
                            # 检查这个位置是否在span标签中
                            before = temp_text[:pos]
                            if '<span class="keyword"' in before and '</span>' not in before.split('<span class="keyword"')[-1]:
                                # 在span标签中，跳过
                                escaped_text += temp_text[:pos + len(escaped_kw)]
                                temp_text = temp_text[pos + len(escaped_kw):]
                            else:
                                # 不在span标签中，高亮
                                escaped_text += before + f'<span class="keyword" style="background:{color};">{escaped_kw}</span>'
                                temp_text = temp_text[pos + len(escaped_kw):]
                    final_parts.append(escaped_text)
            
            highlighted = ''.join(final_parts)
            
            # 对FILE行进行特殊处理：加粗、放大、下划线
            highlighted = re.sub(
                r'(# FILE:[^\n]+)',
                r'<span style="font-weight:bold; font-size:1.1em; text-decoration:underline; color:#2c3e50;">\1</span>',
                highlighted
            )
            
            parts.append(f'<pre style="margin-top: 8px;">{highlighted}</pre>')
            return '\n'.join(parts)
        
        # 没有KEYWORD标记，从Preview中提取（旧格式兼容）
        if not keywords:
            # 没有关键词信息，直接返回
            parts.append(f'<pre style="margin-top: 8px;">{self._escape_html(result_str)}</pre>')
            return '\n'.join(parts)
        
        # 使用统一的颜色分配函数
        keyword_colors = build_keyword_colors(keywords)
        
        # 在转义之前，先从Preview中提取匹配的文本并高亮关键词
        preview_pattern = r'Preview: "([^"]*)"'
        
        def highlight_preview(match):
            original_preview = match.group(1)
            highlighted_preview = self._escape_html(original_preview)
            # 高亮所有匹配的关键词
            for kw, color in keyword_colors.items():
                escaped_kw = self._escape_html(kw)
                # 不区分大小写匹配
                pattern = re.compile(re.escape(escaped_kw), re.IGNORECASE)
                highlighted_preview = pattern.sub(
                    lambda m: f'<span class="keyword" style="background:{color};">{m.group(0)}</span>',
                    highlighted_preview
                )
            return f'Preview: &quot;{highlighted_preview}&quot;'
        
        # 分块处理：转义HTML，但Preview部分会被replace函数特殊处理
        final_parts_list = []
        last_end = 0
        
        for match in re.finditer(preview_pattern, result_str):
            # Preview之前的部分需要转义
            before_text = result_str[last_end:match.start()]
            final_parts_list.append(self._escape_html(before_text))
            # Preview部分用highlight_preview处理
            final_parts_list.append(highlight_preview(match))
            last_end = match.end()
        
        # 最后一段
        final_parts_list.append(self._escape_html(result_str[last_end:]))
        
        highlighted = ''.join(final_parts_list)
        
        parts.append(f'<pre style="margin-top: 8px;">{highlighted}</pre>')
        
        return '\n'.join(parts)
    
    def _format_text(self, text: str) -> str:
        """格式化文本，保留换行"""
        if not text:
            return ''
        return self._escape_html(str(text))
    
    def _escape_html(self, text: str) -> str:
        """转义HTML特殊字符"""
        if not text:
            return ''
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#39;')
        return text


def main():
    parser = argparse.ArgumentParser(
        description='RAG系统结果可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python read_results.py experiment_20251224_104240
    python read_results.py experiment_20251224_104240 --threshold 0.7
    python read_results.py experiment_20251224_104240 --threshold 0.8 --score-column perf_score
        """
    )
    
    parser.add_argument(
        'experiment_dir',
        type=str,
        help='实验结果目录名称（如 experiment_20251224_104240）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='分数阈值，只显示低于此分数的样本（默认: 0.8）'
    )
    
    parser.add_argument(
        '--score-column',
        type=str,
        default='overall_score',
        help='CSV中的分数列名（默认: overall_score）'
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help='Baseline CSV文件路径（可选）'
    )
    
    parser.add_argument(
        '--samples',
        type=str,
        nargs='+',
        default=None,
        help='手动指定要显示的样本ID列表（可选，如果不指定则自动选择每类4个样本）'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / args.experiment_dir
    
    if not results_dir.exists():
        print(f"错误: 实验目录不存在: {results_dir}")
        return
    
    print(f"正在分析实验: {args.experiment_dir}")
    print(f"分数阈值: {args.threshold}")
    print(f"分数列: {args.score_column}")
    print()
    
    baseline_path = Path(args.baseline) if args.baseline else None
    manual_samples = args.samples if args.samples else None
    
    visualizer = RAGResultsVisualizer(
        results_dir=results_dir,
        threshold=args.threshold,
        score_column=args.score_column,
        baseline_csv=baseline_path,
        manual_samples=manual_samples
    )
    
    visualizer.generate_visualizations()


if __name__ == '__main__':
    main()
