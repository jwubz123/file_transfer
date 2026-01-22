#!/usr/bin/env python3
"""
MCP工具过滤器 - 用于过滤filesystem MCP服务器的工具

此脚本作为MCP代理，只暴露指定的工具，过滤掉其他不需要的工具。
适用于减少工具数量，避免模型在大量工具中做出错误判断。

使用方法:
在配置文件中将此脚本作为MCP服务器使用，而不是直接使用filesystem服务器。

命令行参数:
  --tools TOOL1,TOOL2,...  指定允许的工具列表（逗号分隔）
  其他参数将传递给上游MCP服务器
"""

import argparse
import asyncio
import json
import sys
import re
import copy
import logging
import subprocess
import os
import yaml
import aiohttp
import traceback
import hashlib
import time
from typing import Any, Dict, List, Optional, Set
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import read_file_char utility
import sys
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_file_char import read_file_char_sync, format_read_file_char_output
READ_FILE_CHAR_AVAILABLE = True

# Import grep_char for character-level context search
from grep_char import GrepTools

# 配置logging以抑制mcp_filesystem的ripgrep警告
logging.basicConfig(level=logging.ERROR)
# 特别抑制mcp_filesystem.grep的警告
logging.getLogger('mcp_filesystem.grep').setLevel(logging.ERROR)


# ============================================================================
# Keyword Parser Functions (embedded to avoid external imports)
# ============================================================================

def normalize_quotes(text: str) -> str:
    """标准化所有引号为英文引号"""
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('，', ',')
    return text


def extract_all_strings_helper(obj: Any) -> List[str]:
    """递归提取对象中的所有字符串（回退机制使用）"""
    strings = []
    if isinstance(obj, str):
        obj = obj.strip().strip('"\'()[]{}')
        if obj:
            strings.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strings.extend(extract_all_strings_helper(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(extract_all_strings_helper(value))
    return strings


def extract_keywords_content(text: str) -> str:
    """提取keywords参数的内容"""
    match = re.search(r'keywords\s*=\s*(\[.+)', text, re.DOTALL)
    if match:
        content = match.group(1)
        return extract_until_matching_bracket(content)
    
    match = re.search(r'grep_files\s*\(\s*(\[.+)', text, re.DOTALL)
    if match:
        content = match.group(1)
        return extract_until_matching_bracket(content)
    
    return ""


def extract_until_matching_bracket(text: str) -> str:
    """从左括号开始，找到匹配的右括号"""
    if not text.startswith('['):
        return ""
    
    text_stripped = text.rstrip()
    if text_stripped.endswith('>)'):
        text = text_stripped[:-2] + '])'
    elif text_stripped.endswith('>'):
        text = text_stripped[:-1] + ']'
    
    depth = 0
    for i, char in enumerate(text):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0:
                return text[:i+1]
    
    # 括号不匹配时，尝试在参数分隔符处截断
    # 查找 ]),reason_ 或 ]),original_ 模式
    param_match = re.search(r'\]\s*\)\s*,\s*(reason_|original_)', text)
    if param_match:
        # 在这个位置截断，添加缺失的外层]
        return text[:param_match.start() + 1] + ']'
    
    # 如果没有找到参数分隔符，添加缺失的]
    return text.rstrip() + ']' * depth


def extract_string(text: str) -> str:
    """提取字符串内容（移除引号）"""
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def parse_tuple_items(text: str) -> List[str]:
    """解析元组内的多个项目"""
    items = []
    current = ""
    in_string = False
    string_char = None
    depth = 0
    
    for i, char in enumerate(text):
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                if i > 0 and text[i-1] != '\\':
                    in_string = False
                    string_char = None
            current += char
            continue
        
        if in_string:
            current += char
            continue
        
        if char in '[(':
            depth += 1
            current += char
        elif char in '])':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            if current.strip():
                item_text = current.strip()
                if item_text.startswith('"') or item_text.startswith("'"):
                    items.append(extract_string(item_text))
                else:
                    items.append(item_text)
            current = ""
        else:
            current += char
    
    if current.strip():
        item_text = current.strip()
        if item_text.startswith('"') or item_text.startswith("'"):
            items.append(extract_string(item_text))
        else:
            items.append(item_text)
    
    return items


def parse_item(text: str) -> Union[str, List[str], None]:
    """解析单个项目"""
    text = text.strip()
    if not text:
        return None
    
    # 处理元组：(item1, item2, ...)
    if text.startswith('(') and text.endswith(')'):
        inner = text[1:-1].strip()
        # 元组内容需要split成多个项
        return parse_tuple_items(inner)
    
    # 处理列表：[item1, item2, ...]
    if text.startswith('['):
        inner = text[1:-1] if text.endswith(']') else text[1:]
        return parse_tuple_items(inner)
    
    # 处理字符串
    elif text.startswith('"') or text.startswith("'"):
        return extract_string(text)
    
    # 处理带冒号的key:value格式
    elif ':' in text and not text.startswith('{'):
        parts = text.split(':', 1)
        value = parts[1].strip()
        if value.startswith('"') or value.startswith("'"):
            return extract_string(value)
        return value
    else:
        return text


def parse_list_or_tuple(text: str) -> List[Union[str, List[str]]]:
    """解析列表或元组"""
    text = text.strip()
    
    if text.startswith('['):
        text = text[1:]
    if text.endswith(']'):
        text = text[:-1]
    
    result = []
    depth = 0
    current_item = ""
    in_string = False
    string_char = None
    
    i = 0
    while i < len(text):
        char = text[i]
        
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                if i > 0 and text[i-1] != '\\':
                    in_string = False
                    string_char = None
            current_item += char
            i += 1
            continue
        
        if in_string:
            current_item += char
            i += 1
            continue
        
        if char in '[(':
            depth += 1
            current_item += char
        elif char in '])':
            depth -= 1
            current_item += char
        elif char == ',' and depth == 0:
            if current_item.strip():
                item = parse_item(current_item.strip())
                if item is not None:
                    result.append(item)
            current_item = ""
        else:
            current_item += char
        
        i += 1
    
    if current_item.strip():
        item = parse_item(current_item.strip())
        if item is not None:
            result.append(item)
    
    return result


def parse_group(text: str) -> Union[List[Union[str, List[str]]], None]:
    """解析单个组"""
    text = text.strip()
    if not text:
        return None
    
    if text.startswith('['):
        return parse_list_or_tuple(text)
    elif text.startswith('"') or text.startswith("'"):
        return [extract_string(text)]
    else:
        return [text]


def parse_keywords_list(content: str) -> List[List[Union[str, List[str]]]]:
    """解析keywords列表内容"""
    content = content.strip()
    if content.startswith('['):
        content = content[1:]
    if content.endswith(']'):
        content = content[:-1]
    
    result = []
    depth = 0
    current_item = ""
    in_string = False
    string_char = None
    
    i = 0
    while i < len(content):
        char = content[i]
        
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                if i > 0 and content[i-1] != '\\':
                    in_string = False
                    string_char = None
            current_item += char
            i += 1
            continue
        
        if in_string:
            current_item += char
            i += 1
            continue
        
        if char in '[(':
            depth += 1
            current_item += char
        elif char in '])':
            depth -= 1
            current_item += char
            
            if depth == 0:
                group = parse_group(current_item.strip())
                if group is not None:
                    result.append(group)
                current_item = ""
        elif char == ',' and depth == 0:
            if current_item.strip():
                group = parse_group(current_item.strip())
                if group is not None:
                    result.append(group)
                current_item = ""
        else:
            current_item += char
        
        i += 1
    
    if current_item.strip():
        group = parse_group(current_item.strip())
        if group is not None:
            result.append(group)
    
    return result


def normalize_keywords_string(keywords_str: str) -> str:
    """
    标准化keywords字符串：
    1. 将字典转换为值列表 {'k': 'v'} → ['v']
    2. 删除单词周围的小括号 ("word") → "word"
    3. 将加号替换为逗号 "a"+"b" → "a","b"
    4. 将加号连接的词分开 "word1+word2" → "word1","word2"
    """
    result = keywords_str
    
    # 步骤0: 将字典格式转换为列表
    # {'group1': 'keyword1', 'group2': 'keyword2'} → ['keyword1', 'keyword2']
    def convert_dict_to_list(match):
        dict_str = match.group(0)
        # 提取所有值（引号内的内容）
        values = re.findall(r":\s*['\"]([^'\"]+)['\"]", dict_str)
        if values:
            return '[' + ', '.join(f'"{v}"' for v in values) + ']'
        return dict_str
    
    # 匹配 {key: value, ...} 格式
    result = re.sub(r'\{[^{}]*:[^{}]*\}', convert_dict_to_list, result)
    
    # 步骤1: 删除单词周围的小括号
    # 处理 ("word") 或 ('word') 或 ("word1","word2")
    # 先处理元组内的多个元素 ("a","b") → "a","b"
    result = re.sub(r'\(\s*("(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*)\s*\)', r'\1', result)
    result = re.sub(r"\(\s*('(?:[^'\\]|\\.)*'(?:\s*,\s*'(?:[^'\\]|\\.)*')*)\s*\)", r'\1', result)
    
    # 步骤2: 将加号分隔的字符串替换为逗号分隔
    # "word1"+"word2" → "word1","word2"
    result = re.sub(r'"\s*\+\s*"', '","', result)
    result = re.sub(r"'\s*\+\s*'", "','", result)
    
    # 步骤3: 处理字符串内的加号 "word1+word2+word3" → "word1","word2","word3"
    # 找到所有带加号的字符串并拆分
    def split_plus_in_string(match):
        content = match.group(1)
        if '+' in content:
            # 拆分并重建为多个引号字符串
            parts = content.split('+')
            return '"' + '","'.join(parts) + '"'
        return match.group(0)
    
    result = re.sub(r'"([^"]*\+[^"]*)"', split_plus_in_string, result)
    result = re.sub(r"'([^']*\+[^']*)'", split_plus_in_string, result)
    
    return result


def extract_keywords_from_call(text: str) -> List[List[Union[str, List[str]]]]:
    """从grep_files调用文本中提取keywords参数"""
    text = normalize_quotes(text)
    # 先应用标准化（删除括号、加号变逗号）
    text = normalize_keywords_string(text)
    keywords_content = extract_keywords_content(text)
    if not keywords_content:
        return []
    return parse_keywords_list(keywords_content)


def flatten_to_two_levels(keywords: List) -> List[List[str]]:
    """将keywords展平为二层list"""
    result = []
    
    for group in keywords:
        if not group:
            continue
            
        if isinstance(group, str):
            result.append([group])
            continue
        
        if isinstance(group, (list, tuple)):
            flat_group = []
            
            for item in group:
                if isinstance(item, str):
                    flat_group.append(item)
                elif isinstance(item, (list, tuple)):
                    for sub_item in item:
                        if isinstance(sub_item, str):
                            flat_group.append(sub_item)
                        else:
                            flat_group.append(str(sub_item))
                else:
                    flat_group.append(str(item))
            
            if flat_group:
                result.append(flat_group)
    
    return result if result else [[]]


def parse_grep_files_call(text: str) -> dict:
    """完整解析grep_files调用"""
    try:
        keywords = extract_keywords_from_call(text)
        keywords = flatten_to_two_levels(keywords)
        
        # 智能合并：如果有连续的单元素list在多元素list之前，合并它们
        # 这是为了处理格式错误的输入（如缺少外层括号）
        if len(keywords) > 2:
            # 找到第一个多元素list的位置
            first_multi_idx = -1
            for i, group in enumerate(keywords):
                if len(group) > 1:
                    first_multi_idx = i
                    break
            
            # 如果有多元素list，且之前有单元素list，合并它们
            if first_multi_idx > 0:
                merged = []
                for item in keywords[:first_multi_idx]:
                    merged.extend(item)
                result = [merged] if merged else []  # 不要再包一层list
                result.extend(keywords[first_multi_idx:])
                keywords = result
        
        return {'keywords': keywords}
    except Exception as e:
        print(f"[WARNING] Parse failed, using fallback: {e}", file=sys.stderr)
        all_strings = extract_all_strings_helper(text)
        return {'keywords': [all_strings] if all_strings else [[]]}


def normalize_keywords_input(keywords: Any) -> List[List[str]]:
    """将不同格式的关键词输入规范化为二层list格式"""
    def preprocess_string(s: str) -> str:
        if not isinstance(s, str):
            return str(s)
        s = s.replace('"', '"').replace('"', '"')
        s = s.replace(''', "'").replace(''', "'")
        s = s.replace('（', '(').replace('）', ')')
        s = s.replace('【', '[').replace('】', ']')
        s = s.strip('()（）""\'\'[]【】')
        return s
    
    def extract_all_keywords(obj: Any) -> List[str]:
        results = []
        if isinstance(obj, str):
            cleaned = preprocess_string(obj)
            if cleaned:
                results.append(cleaned)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                results.extend(extract_all_keywords(item))
        elif isinstance(obj, dict):
            for value in obj.values():
                results.extend(extract_all_keywords(value))
        return results
    
    def flatten_item(item: Any) -> List[str]:
        if isinstance(item, str):
            return [preprocess_string(item)]
        elif isinstance(item, tuple):
            result = []
            for sub in item:
                result.extend(flatten_item(sub))
            return result
        elif isinstance(item, list):
            if all(isinstance(x, str) for x in item):
                return [preprocess_string(x) for x in item if x]
            result = []
            for sub in item:
                result.extend(flatten_item(sub))
            return result
        elif isinstance(item, dict):
            result = []
            for value in item.values():
                result.extend(flatten_item(value))
            return result
        else:
            return [str(item)] if item else []
    
    if not keywords:
        return [[]]
    
    if isinstance(keywords, str):
        keywords = preprocess_string(keywords)
        try:
            import ast
            keywords = ast.literal_eval(keywords)
        except:
            return [[keywords]] if keywords else [[]]
    
    if isinstance(keywords, list):
        if all(isinstance(x, str) for x in keywords):
            cleaned = [preprocess_string(x) for x in keywords if x]
            return [cleaned] if cleaned else [[]]
        
        result = []
        for group in keywords:
            flat_group = flatten_item(group)
            if flat_group:
                result.append(flat_group)
        
        if result:
            return result
        else:
            all_kws = extract_all_keywords(keywords)
            return [all_kws] if all_kws else [[]]
    
    if isinstance(keywords, dict):
        values = flatten_item(keywords)
        return [values] if values else [[]]
    
    if isinstance(keywords, tuple):
        flat = flatten_item(keywords)
        return [flat] if flat else [[]]
    
    all_kws = extract_all_keywords(keywords)
    return [all_kws] if all_kws else [[]]


# ============================================================================
# End of Keyword Parser Functions
# ============================================================================


def load_decision_agent_id(config_path: str) -> Optional[str]:
    """
    从YAML配置文件加载decision agent的ID
    
    Args:
        config_path: YAML配置文件路径（decision_agent.yaml）
        
    Returns:
        agent_id字符串，如果加载失败返回None
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从agents列表中找到decision agent的id
        if 'agents' in config:
            for agent_item in config['agents']:
                if agent_item.get('id') == 'decision':
                    agent_id = agent_item.get('id')
                    print(f"[CONFIG] Found decision agent: {agent_id}", file=sys.stderr)
                    return agent_id
        
        print("[WARNING] decision agent not found in config", file=sys.stderr)
        return None
        
    except Exception as e:
        print(f"[WARNING] Failed to load decision agent id from {config_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# Removed _get_default_decision_config - no longer needed with MCP subagent


def format_content_block_markdown(file_path: str, content_blocks: List[str], relative_path: str) -> str:
    """
    将文件内容格式化为markdown格式，参考format.md的格式
    正文部分使用缩进符（4个空格）
    
    Args:
        file_path: 文件的绝对路径
        content_blocks: 内容块列表，每个块格式为 "Lines {start}-{end}:\n{content}"
        relative_path: 文件的相对路径（文件名）
        
    Returns:
        格式化后的markdown文本
    """
    # 解析所有块以计算总行数和省略行数
    all_line_ranges = []
    formatted_blocks = []
    
    for block in content_blocks:
        # 解析行号范围和内容
        # block格式: "Lines {start}-{end}:\n{content}"
        if block.startswith("Lines "):
            lines_marker_end = block.find(":\n")
            if lines_marker_end != -1:
                line_range_str = block[6:lines_marker_end]  # 去掉"Lines "前缀
                content = block[lines_marker_end + 2:]  # 去掉":\n"
                
                # 解析行号范围
                if '-' in line_range_str:
                    start, end = map(int, line_range_str.split('-'))
                else:
                    start = end = int(line_range_str)
                
                all_line_ranges.append((start, end))
                
                # 格式化这个块
                block_lines = [f"## offset {line_range_str}"]
                
                # 为每一行内容添加缩进（4个空格）
                for line in content.split('\n'):
                    block_lines.append(f"    {line}")
                
                formatted_blocks.append('\n'.join(block_lines))
    
    # 计算总行数和省略行数
    if all_line_ranges:
        max_line = max(end for _, end in all_line_ranges)
        shown_lines = sum(end - start + 1 for start, end in all_line_ranges)
        omitted_lines = max(0, max_line - shown_lines)
    else:
        max_line = 0
        omitted_lines = 0
    
    # 构建最终输出
    result_lines = [f"# File: {relative_path}: Total length: {max_line} lines, {omitted_lines} lines omitted"]
    result_lines.extend(formatted_blocks)
    
    return '\n'.join(result_lines)


def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量
    
    对于中文，每个字符约等于1个token
    对于英文，每4个字符约等于1个token
    这是一个粗略估算
    
    Args:
        text: 要估算的文本
        
    Returns:
        估算的token数量
    """
    if not text:
        return 0
    
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text)
    english_chars = total_chars - chinese_chars
    
    # 中文: 1字符 ≈ 1 token
    # 英文: 4字符 ≈ 1 token
    return chinese_chars + (english_chars // 4)


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    将文本截断到指定的token数量
    
    Args:
        text: 要截断的文本
        max_tokens: 最大token数量
        
    Returns:
        截断后的文本
    """
    if not text or max_tokens <= 0:
        return ""
    
    current_tokens = 0
    result_chars = []
    
    for char in text:
        # 估算当前字符的token数
        if '\u4e00' <= char <= '\u9fff':
            char_tokens = 1
        else:
            char_tokens = 0.25  # 英文字符
        
        if current_tokens + char_tokens > max_tokens // 2:
            break
        
        result_chars.append(char)
        current_tokens += char_tokens
    
    truncated = ''.join(result_chars)
    if len(truncated) < len(text):
        truncated += "... [truncated]"
    
    return truncated



class TextContent:
    """文本内容类，用于构建MCP响应"""
    def __init__(self, text: str):
        self.type = 'text'
        self.text = text


def normalize_keywords_input(keywords: Any) -> List[List[str]]:
    """
    将不同格式的关键词输入规范化为二层list格式（list of list）
    
    核心原则：
    1. 输出必须是二层list，不允许三层或更多层
    2. 小括号被省略，不作为特殊分组标记
    3. 回退机制：无法识别结构时，提取所有关键词到一个组
    
    支持大模型可能输出的各种格式：
    - 标准格式: [['k1', 'k2'], ['k3']]
    - 元组格式: [[('k1',), ('k2',)], [('k3',)]] -> 小括号被省略
    - 中文引号/括号: [[（"AMT挂账"）], [（"核销"）]]
    - 缺少引号: [["k1", "k2'], ["k3"]]
    - 字典格式: [{'g1': 'k1'}, {'g2': 'k2'}]
    - 混合格式: [['k1'], ('k2',), {'k3': 'v3'}]
    
    Args:
        keywords: 可以是多种格式的关键词输入
        
    Returns:
        List[List[str]]: 规范化后的二层list格式
        
    Examples:
        >>> normalize_keywords_input(['k1', 'k2'])
        [['k1', 'k2']]
        
        >>> normalize_keywords_input([['k1', 'k2'], ['k3', 'k4']])
        [['k1', 'k2'], ['k3', 'k4']]
        
        >>> normalize_keywords_input([[('k1',), ('k2',)], [('k3',)]])
        [['k1', 'k2'], ['k3']]
        
        >>> normalize_keywords_input([[("AMT挂账"), ("研发样机")], [("核销")]])
        [['AMT挂账', '研发样机'], ['核销']]
    """
    
    def preprocess_string(s: str) -> str:
        """预处理字符串：替换中文标点，去除多余括号和引号"""
        if not isinstance(s, str):
            return str(s)
        # 替换中文标点
        s = s.replace('"', '"').replace('"', '"')
        s = s.replace(''', "'").replace(''', "'")
        s = s.replace('（', '(').replace('）', ')')
        s = s.replace('【', '[').replace('】', ']')
        # 去除字符串两端的括号和引号
        s = s.strip('()（）""\'\'[]【】')
        return s
    
    def extract_all_keywords(obj: Any) -> List[str]:
        """递归提取所有关键词（回退机制）"""
        results = []
        if isinstance(obj, str):
            cleaned = preprocess_string(obj)
            if cleaned:
                results.append(cleaned)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                results.extend(extract_all_keywords(item))
        elif isinstance(obj, dict):
            for value in obj.values():
                results.extend(extract_all_keywords(value))
        return results
    
    def flatten_item(item: Any) -> List[str]:
        """
        展平单个项目，确保返回字符串列表
        小括号被忽略
        """
        if isinstance(item, str):
            return [preprocess_string(item)]
        elif isinstance(item, tuple):
            # 小括号被省略，展平元组
            result = []
            for sub in item:
                result.extend(flatten_item(sub))
            return result
        elif isinstance(item, list):
            # 检查是否是字符串列表
            if all(isinstance(x, str) for x in item):
                return [preprocess_string(x) for x in item if x]
            # 否则递归展平
            result = []
            for sub in item:
                result.extend(flatten_item(sub))
            return result
        elif isinstance(item, dict):
            # 字典：提取所有值
            result = []
            for value in item.values():
                result.extend(flatten_item(value))
            return result
        else:
            return [str(item)] if item else []
    
    # 处理空输入
    if not keywords:
        return [[]]
    
    # 如果是字符串，尝试解析或使用回退
    if isinstance(keywords, str):
        keywords = preprocess_string(keywords)
        try:
            import ast
            keywords = ast.literal_eval(keywords)
        except:
            # 回退：单个关键词
            return [[keywords]] if keywords else [[]]
    
    # 如果是列表
    if isinstance(keywords, list):
        # 检查是否是字符串列表（一层）
        if all(isinstance(x, str) for x in keywords):
            # 单层列表，作为一个组
            cleaned = [preprocess_string(x) for x in keywords if x]
            return [cleaned] if cleaned else [[]]
        
        # 多层列表，展平到二层
        result = []
        for group in keywords:
            flat_group = flatten_item(group)
            if flat_group:
                result.append(flat_group)
        
        # 检查是否成功解析
        if result:
            return result
        else:
            # 回退：提取所有关键词
            all_kws = extract_all_keywords(keywords)
            return [all_kws] if all_kws else [[]]
    
    # 如果是字典
    if isinstance(keywords, dict):
        values = flatten_item(keywords)
        return [values] if values else [[]]
    
    # 如果是元组（小括号被省略）
    if isinstance(keywords, tuple):
        flat = flatten_item(keywords)
        return [flat] if flat else [[]]
    
    # 回退：提取所有关键词
    all_kws = extract_all_keywords(keywords)
    return [all_kws] if all_kws else [[]]



# def normalize_keywords_input(keywords: Any) -> List[List[str]]:
#     """
#     将不同格式的关键词输入规范化为list of list格式
    
#     支持大模型可能输出的各种格式：
#     - 标准格式: [['k1', 'k2'], ['k3']]
#     - 元组格式: [[('k1',), ('k2',)], [('k3',)]]
#     - 嵌套元组: [[('k1'), ('k2')], [('k3')]]
#     - 字典格式: [{'g1': 'k1'}, {'g2': 'k2'}]
#     - 混合格式: [['k1'], ('k2',), {'k3': 'v3'}]
#     - 各种括号组合
#     - 中文引号/括号: [[（"AMT挂账"）], [（"核销"）]]
#     - 缺少引号: [["k1", "k2'], ["k3"]]
    
#     Args:
#         keywords: 可以是以下格式之一:
#             - List[str]: 单个关键词列表，如 ['k1', 'k2']
#             - List[Dict]: 字典列表，如 [{'group1': 'k1', 'group2': 'k2'}]
#             - List[List[str]]: 已经是正确格式
#             - List[List[tuple]]: 包含元组的列表，如 [[('k1',), ('k2',)], [('k3',)]]
#             - Dict: 单个字典，如 {'group1': 'k1', 'group2': 'k2'}
#             - Mixed: 混合格式，如 [{'title': '材料'}, ['department']]
        
#     Returns:
#         List[List[str]]: 规范化后的list of list格式
        
#     Helper function:
#         preprocess_string: 预处理字符串，替换中文标点
        
#     Examples:
#         >>> normalize_keywords_input(['k1', 'k2'])
#         [['k1', 'k2']]
        
#         >>> normalize_keywords_input([{'g1': 'k1'}, {'g2': 'k2'}])
#         [['k1'], ['k2']]
        
#         >>> normalize_keywords_input([['k1', 'k2'], ['k3', 'k4']])
#         [['k1', 'k2'], ['k3', 'k4']]
        
#         >>> normalize_keywords_input({'group1': 'k1', 'group2': 'k2'})
#         [['k1', 'k2']]
        
#         >>> normalize_keywords_input([{'title': '材料'}, ['department']])
#         [['材料'], ['department']]
        
#         >>> normalize_keywords_input([[('项目化',), ('项目制',)], [('任职申报',), ('职位申请',)]])
#         [['项目化', '项目制'], ['任职申报', '职位申请']]
        
#         >>> normalize_keywords_input([[('项目化任职'), ('岗位聘任')], [('材料清单'), ('需提交')]])
#         [['项目化任职', '岗位聘任'], ['材料清单', '需提交']]
        
#         >>> normalize_keywords_input([[("AMT挂账"), ("研发样机")], [("核销"), ("冲销")]])
#         [['AMT挂账', '研发样机'], ['核销', '冲销']]
#     """
#     # 辅助函数：预处理字符串，替换中文标点
#     def preprocess_string(s: str) -> str:
#         """预处理字符串：替换中文标点为英文标点"""
#         if not isinstance(s, str):
#             return str(s)
#         s = s.replace('"', '"').replace('"', '"')  # 中文引号
#         s = s.replace(''', "'").replace(''', "'")  # 中文单引号  
#         s = s.replace('（', '(').replace('）', ')')  # 中文括号
#         s = s.replace('【', '[').replace('】', ']')  # 中文方括号
#         # 去除字符串两端可能的括号和引号
#         chars_to_strip = '()（）""' + "'" + '[]【】'
#         s = s.strip(chars_to_strip)
#         return s
    
#     # 处理空输入
#     if not keywords:
#         return [[]]
    
#     # 如果输入本身是字符串（格式错误的情况），尝试解析
#     if isinstance(keywords, str):
#         # 替换中文标点
#         keywords = keywords.replace('"', '"').replace('"', '"')
#         keywords = keywords.replace(''', "'").replace(''', "'")
#         keywords = keywords.replace('（', '(').replace('）', ')')
#         keywords = keywords.replace('【', '[').replace('】', ']')
        
#         try:
#             import ast
#             keywords = ast.literal_eval(keywords)
#         except:
#             print(f"[WARNING] Failed to parse keywords string: {keywords[:100]}", file=sys.stderr)
#             return [[]]
    
#     # 如果已经是list格式
#     if isinstance(keywords, list) and len(keywords) > 0:
#         result = []
        
#         # 遍历所有元素，处理混合格式
#         for elem in keywords:
#             # 如果是list (已经是正确格式的一组)
#             if isinstance(elem, list):
#                 # 递归转换内部元素，处理元组、字符串、嵌套结构
#                 converted = []
#                 for x in elem:
#                     if isinstance(x, tuple):
#                         # 元组转换：提取所有非空元素
#                         for item in x:
#                             if item:  # 过滤空值
#                                 converted.append(preprocess_string(str(item)))
#                     elif isinstance(x, list):
#                         # 嵌套列表：递归处理
#                         for item in x:
#                             if isinstance(item, tuple):
#                                 for sub_item in item:
#                                     if sub_item:
#                                         converted.append(preprocess_string(str(sub_item)))
#                             elif item:
#                                 converted.append(preprocess_string(str(item)))
#                     elif isinstance(x, str):
#                         converted.append(preprocess_string(x))
#                     elif isinstance(x, (int, float)):
#                         # 数字转字符串
#                         converted.append(str(x))
#                     elif isinstance(x, dict):
#                         # 嵌套字典：提取值
#                         for v in x.values():
#                             if v:
#                                 converted.append(preprocess_string(str(v)))
#                     elif x:
#                         # 其他类型转为字符串
#                         converted.append(str(x))
#                 if converted:
#                     result.append(converted)
            
#             # 如果是元组，递归处理所有元素
#             elif isinstance(elem, tuple):
#                 # 提取元组中所有非空元素
#                 tuple_items = []
#                 for item in elem:
#                     if item:
#                         tuple_items.append(preprocess_string(str(item)))
#                 if tuple_items:
#                     result.append(tuple_items)
            
#             # 如果是dict: {'g1': 'k1', 'g2': 'k2'}
#             # 提取所有值作为一个组（OR逻辑）
#             elif isinstance(elem, dict):
#                 values = [preprocess_string(str(v)) for v in elem.values() if v]
#                 if values:
#                     result.append(values)
            
#             # 如果是字符串，作为单独的一组
#             elif isinstance(elem, str) and elem:
#                 result.append([preprocess_string(elem)])
        
#         return result if result else [[]]
    
#     # 如果是单个字典: {'group1': 'k1', 'group2': 'k2'}
#     # 应该变为 [['k1', 'k2']] - 单次grep调用，OR逻辑
#     if isinstance(keywords, dict):
#         values = [str(v) for v in keywords.values() if v]
#         return [values] if values else [[]]
    
#     # 如果是字符串，作为单个关键词
#     if isinstance(keywords, str):
#         return [[keywords]]
    
#     # 默认返回空列表
#     return [[]]


def convert_keywords_to_pattern(keywords: List[str]) -> str:
    """
    将关键词列表转换为grep搜索模式
    
    对于包含中文字符且有空格的关键词，会将空格分隔的部分拆分为OR关系。
    英文短语保持完整。
    
    Args:
        keywords: 关键词列表
        
    Returns:
        用 '|' 连接的搜索模式
        
    Examples:
        >>> convert_keywords_to_pattern(['机器学习', '人工智能', '深度学习 神经网络'])
        '机器学习|人工智能|深度学习|神经网络'
        >>> convert_keywords_to_pattern(['machine learning', 'AI'])
        'machine learning|AI'
    """
    if not keywords:
        return ""
    if isinstance(keywords, dict):
        keywords = keywords.get('text', '')
    processed_keywords = []
    for keyword in keywords:
        
        if not keyword:
            continue
        
        keyword = keyword.strip()
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in keyword)
        
        if has_chinese and ' ' in keyword:
            # 中文关键词有空格时拆分为OR关系
            processed_keywords.extend(keyword.split())
        else:
            # 英文短语或无空格中文词保持完整
            processed_keywords.append(keyword)
    
    return '|'.join(processed_keywords)


def extract_file_paths_from_result(result: Any) -> set:
    """
    从grep_files结果中提取文件路径
    
    Args:
        result: grep_files返回的结果对象
        
    Returns:
        文件路径的集合
    """
    file_paths = set()
    
    if not hasattr(result, 'content'):
        return file_paths
    
    for content in result.content:
        if hasattr(content, 'text'):
            text = content.text
            
            # 主要模式：匹配以/开头，以:结尾的行（grep输出格式）
            matches = re.findall(r'^(/[^:]+):', text, re.MULTILINE)
            if matches:
                file_paths.update(matches)
                continue
            
            # 备用模式1: "Match in /path/to/file" 格式
            matches = re.findall(r'Match in (/[^\n:]+)', text)
            if matches:
                file_paths.update(matches)
                continue
            
            # 备用模式2: 匹配包含corpus的路径
            matches = re.findall(r'(/[^\n]*corpus/[^\n:]+)', text)
            if matches:
                file_paths.update(matches)
    
    return file_paths


def merge_overlapping_ranges(ranges: List[tuple]) -> List[tuple]:
    """
    合并重叠的行号范围
    
    Args:
        ranges: 行号范围列表，每个元素是 (start_line, end_line) 元组
        
    Returns:
        合并后的范围列表
        
    Examples:
        >>> merge_overlapping_ranges([(5, 5), (5, 6), (5, 8)])
        [(5, 8)]
        >>> merge_overlapping_ranges([(1, 3), (5, 7), (6, 9)])
        [(1, 3), (5, 9)]
    """
    if not ranges:
        return []
    
    # 按起始行排序
    sorted_ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
    
    merged = [sorted_ranges[0]]
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        
        # 如果当前范围与上一个范围重叠或相邻，合并它们
        if current_start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # 否则添加新范围
            merged.append((current_start, current_end))
    
    return merged


# grep_with_byte_offset moved to grep_char.py as grep_with_char_context


def rank_search_results(file_preview_map: Dict[str, List[Dict]], total_groups: int, 
                       group_file_counts: Optional[Dict[int, int]] = None,
                       group_keywords_map: Optional[Dict[int, List[str]]] = None) -> List[Dict]:
    """
    对搜索结果进行排序，使用增强的打分系统
    
    新的打分公式：
    总分 = preview分数 + 0.5 * file分数 + Σ(1/该组关键词出现的文件数) + 关键词完整性加分
    
    其中：
    - preview分数 = 该preview命中的不同组数
    - file分数 = 该文件命中的不同组数
    - 命中同组不同keywords分数一致（例如命中'AI'和'人工智能'都算1分，因为同组）
    - 关键词完整性加分 = α * 同组内命中的不同关键词数量 + β * 平均关键词完整性分数
        α=0.1, β=0.2 (可调整的权重参数)
    
    细化排序规则（当总分相同时）：
    1. 优先比较preview分数（命中不同组数）
    2. 如果preview分数也相同，比较同组内命中的不同关键词数量
    3. 如果还相同，比较关键词匹配的完整性
    
    Args:
        file_preview_map: 文件到preview列表的映射
            {file_path: [{'preview_start': ..., 'preview_end': ..., 
                          'keywords': [...], 'group_idx': ..., 'group_indices': [...], ...}, ...]}
        total_groups: 总的关键词组数量
        group_file_counts: 每个group命中的文件数量 {group_idx: file_count}
        group_keywords_map: 每个group包含的关键词列表 {group_idx: [keyword1, keyword2, ...]}
                           （新增参数，用于关键词完整性比较）
        
    Returns:
        排序后的结果列表，每个元素包含file_path、preview信息和score
    """
    results = []
    
    # 关键词完整性权重参数（可调整）
    KEYWORD_COUNT_WEIGHT = 0.1      # 同组关键词数量权重
    COMPLETENESS_WEIGHT = 0.2       # 关键词完整性权重
    
    # 为每个文件计算file分数（命中的不同组数）
    file_scores = {}
    for file_path, previews in file_preview_map.items():
        file_groups = set()
        for preview in previews:
            if 'group_indices' in preview:
                file_groups.update(preview['group_indices'])
            else:
                file_groups.add(preview['group_idx'])
        file_scores[file_path] = len(file_groups)
    
    # 为每个preview计算总分和细化指标
    for file_path, previews in file_preview_map.items():
        file_score = file_scores[file_path]
        
        for preview in previews:
            # 获取preview命中的不同组
            preview_groups = set()
            if 'group_indices' in preview:
                preview_groups = set(preview['group_indices'])
                preview_group_indices = list(preview['group_indices'])
            else:
                preview_groups.add(preview['group_idx'])
                preview_group_indices = [preview['group_idx']]
            
            # preview分数 = 命中的组数
            preview_score = len(preview_groups)
            
            # 计算频率倒数：1/(所有命中组的文件总数之和)
            frequency_inverse_sum = 0.0
            if group_file_counts:
                total_file_count = sum(group_file_counts.get(group_idx, 1) for group_idx in preview_groups)
                frequency_inverse_sum = 1.0 / total_file_count if total_file_count > 0 else 0.0
            
            # 计算关键词相关的细化指标（作为总分的一部分）
            same_group_keyword_count = 0
            keyword_completeness_score = 0.0
            keyword_completeness_bonus = 0.0  # 关键词完整性额外加分
            
            # 如果有keywords信息和group_keywords_map，计算关键词完整性
            if 'keywords' in preview and preview['keywords'] and group_keywords_map:
                # 按关键词组统计命中情况
                group_keyword_hits = {}
                
                # 处理每个命中的关键词
                for keyword_info in preview['keywords']:
                    if isinstance(keyword_info, dict) and 'group_idx' in keyword_info:
                        group_idx = keyword_info['group_idx']
                        keyword_text = keyword_info.get('text', '') if isinstance(keyword_info.get('text'), str) else str(keyword_info.get('text', ''))
                        
                        if group_idx not in group_keyword_hits:
                            group_keyword_hits[group_idx] = set()
                        group_keyword_hits[group_idx].add(keyword_text)
                
                # 计算同组内命中的不同关键词数量（取最大值）
                if group_keyword_hits:
                    same_group_keyword_count = max(len(keywords) for keywords in group_keyword_hits.values())
                    
                    # 计算关键词匹配完整性分数
                    completeness_scores = []
                    for group_idx, keywords in group_keyword_hits.items():
                        # 获取该组所有关键词
                        if group_idx in group_keywords_map:
                            all_group_keywords = group_keywords_map[group_idx]
                            if not all_group_keywords:
                                continue
                            
                            # 找到该组最长的关键词作为基准
                            longest_keyword = max(all_group_keywords, key=len)
                            
                            # 对于该组命中的每个关键词，计算完整性
                            for keyword in keywords:
                                # 查找包含这个关键词的最长关键词
                                containing_keywords = [k for k in all_group_keywords if keyword in k]
                                if containing_keywords:
                                    # 找到包含当前关键词的最长关键词
                                    longest_containing = max(containing_keywords, key=len)
                                    # 计算完整性比例
                                    completeness = len(keyword) / len(longest_containing) if longest_containing else 0
                                    completeness_scores.append(completeness)
                                else:
                                    # 如果没有包含关系，可能是完全独立的短关键词
                                    # 这种情况给一个基础分数，避免为0
                                    completeness_scores.append(0.3)  # 基础分数
                    
                    # 计算平均完整性分数
                    if completeness_scores:
                        keyword_completeness_score = sum(completeness_scores) / len(completeness_scores)
                    
                    # 计算关键词完整性额外加分（作为总分的一部分）
                    keyword_completeness_bonus = (
                        KEYWORD_COUNT_WEIGHT * same_group_keyword_count + 
                        COMPLETENESS_WEIGHT * keyword_completeness_score
                    )
            
            # 新的总分公式：包含关键词完整性加分
            total_score = (
                preview_score + 
                0.5 * file_score + 
                frequency_inverse_sum + 
                keyword_completeness_bonus
            )
            
            # 创建sort_key，用于最终排序
            # 注意：即使已经将完整性纳入总分，但仍然在sort_key中保留细化指标
            # 用于处理浮点数精度问题导致的微小差异
            sort_key = (
                -total_score,  # 主要按总分降序
                -preview_score,  # 其次按preview分数降序
                -same_group_keyword_count,  # 然后按同组关键词数量降序
                -keyword_completeness_score,  # 再按关键词完整性降序
                file_path,  # 文件路径（字符串排序）
                preview.get('preview_start', 0)  # 预览起始位置
            )
            
            results.append({
                'file_path': file_path,
                'preview': preview,
                'preview_score': preview_score,
                'file_score': file_score,
                'frequency_inverse_sum': frequency_inverse_sum,
                'same_group_keyword_count': same_group_keyword_count,
                'keyword_completeness_score': keyword_completeness_score,
                'keyword_completeness_bonus': keyword_completeness_bonus,  # 新增：完整性加分
                'total_score': total_score,
                'sort_key': sort_key
            })
    
    # 按sort_key排序
    results.sort(key=lambda x: x['sort_key'])
    
    return results

def select_topk_closest_line_pairs(
    keyword_group_lines: List[List[int]], 
    topk: int = 8
) -> List[tuple]:
    """
    从不同关键词组匹配的行号中，选择距离最近的top-k个行对
    
    Args:
        keyword_group_lines: 每个关键词组匹配的行号列表
            例如: [[1, 5], [5, 6, 8]] 表示第一组匹配1,5行，第二组匹配5,6,8行
        topk: 返回距离最近的前k个行对
        
    Returns:
        行号范围列表 [(start, end), ...] 已去重合并
        
    Examples:
        >>> select_topk_closest_line_pairs([[1, 5], [5, 6, 8]], topk=3)
        [(5, 8)]  # 距离: (5-5)=0, (6-5)=1, (8-5)=3, 合并后5-8
    """
    if len(keyword_group_lines) < 2:
        # 如果只有一个关键词组，返回所有匹配行
        if keyword_group_lines:
            lines = keyword_group_lines[0]
            return [(line, line) for line in lines]
        return []
    
    # 计算所有可能的行对及其距离
    # (distance, group_a_line, group_b_line, group_a_idx, group_b_idx)
    distances = []
    
    # 对于每对关键词组
    for i in range(len(keyword_group_lines)):
        for j in range(i + 1, len(keyword_group_lines)):
            group_a = keyword_group_lines[i]
            group_b = keyword_group_lines[j]
            
            # 计算这两组之间所有行的距离
            for line_a in group_a:
                for line_b in group_b:
                    distance = abs(line_b - line_a)
                    min_line = min(line_a, line_b)
                    max_line = max(line_a, line_b)
                    distances.append((distance, min_line, max_line, i, j))
    
    # 按距离排序
    distances.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # 选择前topk个
    selected_pairs = distances[:topk]
    
    # 提取行号范围
    ranges = [(min_line, max_line) for _, min_line, max_line, _, _ in selected_pairs]
    
    # 合并重叠范围
    merged_ranges = merge_overlapping_ranges(ranges)
    
    return merged_ranges


def merge_and_filter_results(valid_results: List[Dict], allowed_paths: set, file_frequencies: Optional[Dict[str, int]] = None, max_files: int = 5) -> Any:
    """
    合并多个grep结果，对于交集中的每个文件，输出所有keyword组找到的行的并集
    
    Args:
        valid_results: 包含多个grep结果的列表，每个元素是{'result': result, 'keywords': [...], ...}
        allowed_paths: 允许的文件路径集合（交集）
        file_frequencies: 文件路径到出现次数的映射，用于排序
        max_files: 最多返回的文件数量
        
    Returns:
        合并后的结果对象
    """
    if not valid_results:
        return None
    
    # 按频率排序allowed_paths
    if file_frequencies:
        sorted_paths = sorted(
            allowed_paths,
            key=lambda path: (-file_frequencies.get(path, 0), path)
        )
        allowed_paths = set(sorted_paths[:max_files])
    elif len(allowed_paths) > max_files:
        allowed_paths = set(list(allowed_paths)[:max_files])
    
    # 为每个文件收集所有结果中的行号（并集）
    # file_path -> {line_number -> line_content}
    file_line_map = {}
    
    for item in valid_results:
        result = item['result']
        if not hasattr(result, 'content'):
            continue
        
        for content in result.content:
            if not hasattr(content, 'text'):
                continue
            
            text = content.text
            current_file = None
            
            for line in text.split('\n'):
                # 检查是否是文件路径行
                path_match = re.match(r'^(/[^:]+):', line)
                if path_match:
                    current_file = path_match.group(1)
                    if current_file not in file_line_map:
                        file_line_map[current_file] = {}
                elif current_file and current_file in allowed_paths:
                    # 提取行号和内容
                    line_stripped = line.strip()
                    if line_stripped and ':' in line_stripped:
                        colon_pos = line_stripped.find(':')
                        potential_line_num = line_stripped[:colon_pos].strip()
                        if potential_line_num.isdigit():
                            line_num = int(potential_line_num)
                            # 合并逻辑：保留最长的版本（最完整）
                            if line_num not in file_line_map[current_file] or len(line) > len(file_line_map[current_file][line_num]):
                                file_line_map[current_file][line_num] = line
    
    # 构建合并后的输出
    file_blocks = []
    for file_path in sorted(allowed_paths):
        if file_path in file_line_map and file_line_map[file_path]:
            lines = [f"{file_path}:"]
            # 按行号排序输出
            for line_num in sorted(file_line_map[file_path].keys()):
                lines.append(file_line_map[file_path][line_num])
            file_blocks.append('\n'.join(lines))
    
    # 创建新的结果对象
    if file_blocks:
        merged_text = '\n\n'.join(file_blocks)
        merged_text += f"\n\nFound matches in {len(file_blocks)} files"
        
        # 使用第一个结果作为模板
        base_result = valid_results[0]['result']
        merged_result = copy.copy(base_result)
        
        # 创建新的content
        if base_result.content and len(base_result.content) > 0:
            new_content = copy.copy(base_result.content[0])
            new_content.text = merged_text
            merged_result.content = [new_content]
        else:
            merged_result.content = [TextContent(merged_text)]
        
        return merged_result
    
    # 没有匹配，返回空结果
    base_result = valid_results[0]['result']
    empty_result = copy.copy(base_result)
    empty_result.content = [TextContent("No matches found after filtering")]
    return empty_result



async def merge_and_filter_results_with_context(
    filter_instance: 'MCPToolFilter',
    valid_results: List[Dict], 
    allowed_paths: set, 
    file_frequencies: Optional[Dict[str, int]] = None, 
    max_files: int = 5,
    context_char: int = 100,
    max_tokens: Optional[int] = None,
    reason_refine: str = ""
) -> Any:
    """
    使用grep -ob重新实现的merge函数 - 直接获取字符级匹配
    
    对于每个文件和关键词组，使用grep -ob获取字符级匹配及上下文
    """
    if not valid_results:
        return None
    
    total_groups = len(valid_results)
    
    # 第一步：为每个文件的每个关键词组执行grep -ob搜索
    file_preview_map = {}  # file_path -> [preview_dict, ...]
    
    # 创建临时的GrepTools实例（不需要PathValidator）
    from grep_char import GrepTools
    
    # 创建一个简单的validator mock
    class SimpleValidator:
        async def validate_path(self, path):
            return (path, True)
    
    grep_tools = GrepTools(SimpleValidator())
    
    for item_idx, item in enumerate(valid_results):
        keywords_list = item.get('keywords', [])
        
        # 将keywords转换为pattern（OR逻辑）
        # 使用 convert_keywords_to_pattern 处理中文短语分词
        if not keywords_list:
            continue
        pattern = convert_keywords_to_pattern(keywords_list)
        
        # 对每个文件执行grep
        for file_path in allowed_paths:
            try:
                matches = await grep_tools.grep_with_char_context(
                    file_path, 
                    pattern, 
                    context_char,
                    is_case_sensitive=False
                )
                
                if not matches:
                    continue
                
                if file_path not in file_preview_map:
                    file_preview_map[file_path] = []
                
                # 将匹配结果添加到preview_map
                for match in matches:
                    # 找出实际匹配的关键词
                    matched_keywords = []
                    for kw in keywords_list:
                        if kw and kw in match['matched_text']:
                            matched_keywords.append(kw)
                    
                    preview_info = {
                        'preview_start': match['preview_start'],
                        'preview_end': match['preview_end'],
                        'preview': match['preview'],
                        'group_idx': item_idx,
                        'keywords': matched_keywords,
                        'group_indices': [item_idx]  # 当前preview命中的groups
                    }
                    
                    file_preview_map[file_path].append(preview_info)
                    
            except Exception as e:
                print(f"[ERROR] Failed to grep {file_path}: {e}", file=sys.stderr)
                continue
    
    # 第二步：先排序，再合并重叠的previews（只合并相邻且重叠的previews）
    merged_file_preview_map = {}
    for file_path, previews in file_preview_map.items():
        # 先按位置排序
        previews_sorted = sorted(previews, key=lambda x: x['preview_start'])
        
        merged_previews = []
        for preview in previews_sorted:
            # 检查是否与最后一个merged preview重叠或接近（只看相邻的）
            if merged_previews:
                last_merged = merged_previews[-1]
                # 只有当前preview与最后一个merged preview重叠时才合并
                if (preview['preview_start'] <= last_merged['preview_end'] and
                    preview['preview_end'] >= last_merged['preview_start']):
                    # 扩展范围
                    last_merged['preview_start'] = min(last_merged['preview_start'], preview['preview_start'])
                    last_merged['preview_end'] = max(last_merged['preview_end'], preview['preview_end'])
                    
                    # 合并关键词和group信息
                    existing_kws = set(last_merged['keywords'])
                    new_kws = set(preview['keywords'])
                    last_merged['keywords'] = list(existing_kws | new_kws)
                    
                    existing_groups = set(last_merged['group_indices'])
                    new_groups = set(preview['group_indices'])
                    last_merged['group_indices'] = list(existing_groups | new_groups)
                    
                    # 重新生成preview文本
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            preview_text = content[last_merged['preview_start']:last_merged['preview_end']]
                            if last_merged['preview_start'] > 0:
                                preview_text = '...' + preview_text
                            if last_merged['preview_end'] < len(content):
                                preview_text = preview_text + '...'
                            last_merged['preview'] = preview_text
                    except Exception as e:
                        print(f"[WARNING] Failed to regenerate preview: {e}", file=sys.stderr)
                else:
                    # 不重叠，添加新preview
                    merged_previews.append(preview.copy())
            else:
                # 第一个preview，直接添加
                merged_previews.append(preview.copy())
        
        merged_file_preview_map[file_path] = merged_previews
    
    # 计算每个group命中的文件数量（用于排序）
    group_file_counts = {}
    for file_path, previews in merged_file_preview_map.items():
        for preview in previews:
            # 使用group_indices而不是group_idx，因为preview可能命中多个组
            group_indices = preview.get('group_indices', [preview.get('group_idx')])
            for group_idx in group_indices:
                if group_idx not in group_file_counts:
                    group_file_counts[group_idx] = set()
                group_file_counts[group_idx].add(file_path)
    
    # 将set转换为count
    group_file_counts = {group_idx: len(files) for group_idx, files in group_file_counts.items()}
    
    # 第三步：排序（使用rank_search_results函数，传递group_file_counts）
    ranked_results = rank_search_results(merged_file_preview_map, total_groups, group_file_counts)
    
    # 第四步：生成输出文本（按ranking顺序，换文件时重新输出FILE头）
    output_lines = []
    last_file_path = None
    file_score_map = {}  # 缓存file_score
    
    for result_item in ranked_results:
        file_path = result_item['file_path']
        preview = result_item['preview']
        total_score = result_item['total_score']
        file_score = result_item['file_score']
        
        # 如果换了文件，输出新的文件头
        if file_path != last_file_path:
            relative_path = filter_instance.get_relative_path(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_length = len(f.read())
            except:
                file_length = "..."
            
            output_lines.append(f"# FILE: {relative_path}")
            last_file_path = file_path
        
        # 输出preview，去除前后省略号
        keywords_str = '，'.join(preview['keywords'])
        
        # 处理preview文本：移除grep添加的省略号，只在确实被截断的地方保留
        preview_text = preview['preview']
        # 移除前后的省略号
        preview_text = preview_text.strip()
        if preview_text.startswith('...'):
            preview_text = preview_text[3:]
        if preview_text.endswith('...'):
            preview_text = preview_text[:-3]
        preview_text = preview_text.strip()
        
        output_lines.append(f"## OFFSET {preview['preview_start']}-{preview['preview_end']} [KEYWORD: {keywords_str}] (score {total_score:.2f})")
        output_lines.append(f"    {preview_text}")
    
    merged_text = '\n'.join(output_lines)
    total_tokens = estimate_tokens(merged_text)
    
    # 检查并应用token截断
    if max_tokens is not None and total_tokens > max_tokens:
        # 需要截断，按ranking顺序逐个添加previews直到达到限制
        truncated_lines = []
        current_tokens = 0
        last_file_path = None
        included_count = 0
        total_count = len(ranked_results)
        
        for result_item in ranked_results:
            file_path = result_item['file_path']
            preview = result_item['preview']
            total_score = result_item['total_score']
            file_score = result_item['file_score']
            
            # 构建这个preview的文本
            preview_lines = []
            
            # 如果换了文件，需要添加文件头
            if file_path != last_file_path:
                relative_path = filter_instance.get_relative_path(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_length = len(f.read())
                except:
                    file_length = "..."
                preview_lines.append(f"# FILE: {relative_path}")
                last_file_path = file_path
            
            keywords_str = '，'.join(preview['keywords'])
            preview_text = preview['preview'].strip()
            if preview_text.startswith('...'):
                preview_text = preview_text[3:]
            if preview_text.endswith('...'):
                preview_text = preview_text[:-3]
            preview_text = preview_text.strip()
            
            preview_lines.append(f"## OFFSET {preview['preview_start']}-{preview['preview_end']} [KEYWORD: {keywords_str}] (score {total_score:.2f})")
            preview_lines.append(f"    {preview_text}")
            
            preview_block = '\n'.join(preview_lines)
            preview_tokens = estimate_tokens(preview_block)
            
            # 检查是否超过限制
            if current_tokens + preview_tokens > max_tokens:
                # 如果当前preview会导致超出限制，截断到max_tokens
                remaining_tokens = max_tokens - current_tokens
                truncated_preview = truncate_text_to_tokens(preview_block, remaining_tokens)
                truncated_lines.append(truncated_preview)
                current_tokens += estimate_tokens(truncated_preview)
                included_count += 1
                break
            
            truncated_lines.extend(preview_lines)
            current_tokens += preview_tokens
            included_count += 1
        
        merged_text = '\n'.join(truncated_lines)
        merged_text += f"\n\n⚠️ Output truncated: showing {included_count}/{total_count} previews (~{current_tokens} tokens, limit: {max_tokens})"
    
    # 创建结果对象
    base_result = valid_results[0]['result']
    merged_result = copy.copy(base_result)
    
    if base_result.content and len(base_result.content) > 0:
        new_content = copy.copy(base_result.content[0])
        new_content.text = merged_text
        merged_result.content = [new_content]
    else:
        merged_result.content = [TextContent(merged_text)]
    
    return merged_result



def filter_result_by_paths(result: Any, allowed_paths: set, file_frequencies: Optional[Dict[str, int]] = None, max_files: int = 5) -> Any:
    """
    根据允许的路径过滤grep_files结果，只保留intersection中的文件
    
    Args:
        result: 原始grep_files结果
        allowed_paths: 允许的文件路径集合
        file_frequencies: 文件路径到出现次数的映射，用于排序
        max_files: 最多返回的文件数量
        
    Returns:
        过滤后的结果对象
    """
    if not hasattr(result, 'content'):
        return result
    
    # 如果提供了频率信息，按频率排序allowed_paths
    if file_frequencies:
        # 按频率降序排序，频率相同时保持原顺序
        sorted_paths = sorted(
            allowed_paths,
            key=lambda path: (-file_frequencies.get(path, 0), path)
        )
        # 只保留前max_files个文件
        allowed_paths = set(sorted_paths[:max_files])
    elif len(allowed_paths) > max_files:
        # 如果没有频率信息，按顺序取前max_files个
        allowed_paths = set(list(allowed_paths)[:max_files])
    
    filtered_content = []
    seen_paths = set()  # 跟踪已处理的文件路径，避免重复
    
    for content in result.content:
        if hasattr(content, 'text'):
            text = content.text
            
            # 解析文本，提取每个文件的内容块
            # grep输出格式: /path/to/file:\n     line_num: content...
            file_blocks = []
            current_file = None
            current_lines = []
            seen_line_numbers = set()  # 跟踪当前文件块中已见的行号
            line_content_map = {}  # 行号 -> 完整行内容的映射，用于合并同一行的多个高亮
            
            for line in text.split('\n'):
                # 检查是否是文件路径行
                path_match = re.match(r'^(/[^:]+):', line)
                if path_match:
                    # 保存之前的文件块（处理并合并重复行）
                    if current_file and current_file in allowed_paths and current_file not in seen_paths:
                        # 合并同一行的多个匹配，按行号排序输出
                        merged_lines = [current_lines[0]]  # 保留文件路径行
                        for line_num in sorted(seen_line_numbers):
                            if line_num in line_content_map:
                                merged_lines.append(line_content_map[line_num])
                        
                        # 添加汇总信息（Found X matches...）和空行
                        for line_text in current_lines:
                            if line_text.strip().startswith('Found ') or not line_text.strip():
                                merged_lines.append(line_text)
                        
                        file_blocks.append('\n'.join(merged_lines))
                        seen_paths.add(current_file)
                    
                    # 开始新文件块
                    current_file = path_match.group(1)
                    current_lines = [line]
                    seen_line_numbers = set()  # 重置行号跟踪
                    line_content_map = {}  # 重置行内容映射
                else:
                    # 继续当前文件块
                    if current_file:
                        # 提取行号进行去重判断
                        # 行格式可能是：
                        # "     123: content >>> highlighted <<< more content"
                        # 或空行、分隔符等
                        line_stripped = line.strip()
                        
                        # 尝试提取行号
                        line_num = None
                        if line_stripped and ':' in line_stripped:
                            # 查找第一个冒号的位置
                            colon_pos = line_stripped.find(':')
                            potential_line_num = line_stripped[:colon_pos].strip()
                            if potential_line_num.isdigit():
                                line_num = int(potential_line_num)
                        
                        # 如果成功提取行号，进行去重和合并处理
                        if line_num is not None:
                            # 始终保存（会覆盖之前的版本）
                            # 这样最后一个版本会被保留，通常最后的版本包含最完整的信息
                            if line_num not in seen_line_numbers:
                                seen_line_numbers.add(line_num)
                            line_content_map[line_num] = line
                        else:
                            # 没有行号的行（如空行、分隔符、汇总信息等）
                            # 这些行不需要去重，直接添加到current_lines
                            current_lines.append(line)
            
            # 处理最后一个文件块
            if current_file and current_file in allowed_paths and current_file not in seen_paths:
                # 合并同一行的多个匹配，按行号排序输出
                merged_lines = [current_lines[0]]  # 保留文件路径行
                for line_num in sorted(seen_line_numbers):
                    if line_num in line_content_map:
                        merged_lines.append(line_content_map[line_num])
                
                # 添加汇总信息（Found X matches...）和空行
                for line_text in current_lines:
                    if line_text.strip().startswith('Found ') or not line_text.strip():
                        merged_lines.append(line_text)
                
                file_blocks.append('\n'.join(merged_lines))
                seen_paths.add(current_file)
            
            # 如果有有效的文件块，创建新的content
            if file_blocks:
                filtered_text = '\n'.join(file_blocks)
                new_content = copy.copy(content)
                new_content.text = filtered_text
                filtered_content.append(new_content)
        else:
            # 非文本内容，直接保留
            filtered_content.append(content)
    
    # 使用copy模块复制结果对象并替换content
    try:
        filtered_result = copy.copy(result)
        
        # 确保content不为空（MCP要求）
        if not filtered_content:
            # 从原结果复制content模板
            if result.content and len(result.content) > 0:
                template = result.content[0]
                if hasattr(template, 'text'):
                    no_match_content = type(template)()
                    no_match_content.text = "No matches found after filtering"
                    if hasattr(template, 'type'):
                        no_match_content.type = template.type
                    filtered_content = [no_match_content]
                else:
                    filtered_content = [TextContent("No matches found after filtering")]
            else:
                filtered_content = [TextContent("No matches found after filtering")]
        
        filtered_result.content = filtered_content
        return filtered_result
    except Exception:
        # 如果copy失败，确保content不为空并返回
        if not filtered_content:
            filtered_content = [TextContent("No matches found after filtering")]
        result.content = filtered_content
        return result


class MCPToolFilter:
    """MCP工具过滤器类"""
    
    # 类变量：存储工具调用历史 {query_hash: {"query": str, "history": [...], "timestamp": float}}
    _tool_call_history: Dict[str, Dict[str, Any]] = {}
    _history_ttl: int = 3600  # 历史记录TTL（秒），1小时后自动清理
    
    @classmethod
    def _get_query_hash(cls, query: str) -> str:
        """生成查询的hash作为session key"""
        import hashlib
        return hashlib.md5(query.encode('utf-8')).hexdigest()[:16]
    
    @classmethod
    def _cleanup_old_history(cls):
        """清理过期的历史记录"""
        import time
        current_time = time.time()
        expired_keys = [
            key for key, data in cls._tool_call_history.items()
            if current_time - data.get('timestamp', 0) > cls._history_ttl
        ]
        for key in expired_keys:
            del cls._tool_call_history[key]
        if expired_keys:
            print(f"[HISTORY] Cleaned {len(expired_keys)} expired sessions", file=sys.stderr)
    
    @classmethod
    def _get_or_create_history(cls, query: str) -> List[Dict[str, str]]:
        """获取或创建查询的历史记录"""
        import time
        cls._cleanup_old_history()
        
        query_hash = cls._get_query_hash(query)
        if query_hash not in cls._tool_call_history:
            cls._tool_call_history[query_hash] = {
                'query': query,
                'history': [],
                'timestamp': time.time()
            }
            print(f"[HISTORY] Created new session for query: {query[:50]}...", file=sys.stderr)
        else:
            # 更新时间戳
            cls._tool_call_history[query_hash]['timestamp'] = time.time()
        
        return cls._tool_call_history[query_hash]['history']
    
    @classmethod
    def _add_to_history(cls, query: str, tool_name: str, tool_result: str):
        """添加工具调用结果到历史"""
        history = cls._get_or_create_history(query)
        history.append({
            'tool': tool_name,
            'result': tool_result[:2000]  # 限制每条历史的长度，避免过长
        })
        print(f"[HISTORY] Added {tool_name} result to session (total: {len(history)} calls)", file=sys.stderr)
    
    def __init__(self, upstream_command: str, upstream_args: List[str], allowed_tools: Set[str], grep_results_limit: Optional[int] = None, max_output_files: int = 5, context_char: int = 100, max_tokens: Optional[int] = None, project_root: Optional[str] = None, read_file_char_context: int = 300, decision_prompt_path: Optional[str] = None):
        """
        初始化过滤器
        
        Args:
            upstream_command: 上游MCP服务器命令
            upstream_args: 上游MCP服务器参数
            allowed_tools: 允许的工具名称集合
            grep_results_limit: grep_files的结果数量限制
            max_output_files: 最终输出的文件及content的最大个数，默认5
            context_char: 上下文字符数（替代context_lines），默认100
            max_tokens: 输出的最大token数量，超过则截断并警告
            project_root: 项目根目录，用于转换绝对路径为相对路径
            read_file_char_context: read_file_char工具的默认上下文字符数，默认300
            decision_prompt_path: Decision agent YAML配置文件路径（包含prompt和model配置）
        """
        self.context_char = context_char
        self.upstream_command = upstream_command
        self.upstream_args = upstream_args
        self.allowed_tools = allowed_tools
        
        # 初始化日志目录
        self.grep_results_limit = grep_results_limit
        self.max_output_files = max_output_files
        self.max_tokens = max_tokens
        self.project_root = project_root
        self.read_file_char_context = read_file_char_context
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # 加载decision agent ID
        if decision_prompt_path:
            self.decision_agent_id = load_decision_agent_id(decision_prompt_path)
            if self.decision_agent_id:
                print(f"[CONFIG] Decision agent enabled: {self.decision_agent_id}", file=sys.stderr)
            else:
                print("[CONFIG] Decision agent disabled (failed to load)", file=sys.stderr)
        else:
            self.decision_agent_id = None
            print("[CONFIG] Decision agent disabled (no config path provided)", file=sys.stderr)
        
        # MCP manager将在需要时初始化
        self._mcp_manager = None
    
    def get_relative_path(self, absolute_path: str) -> str:
        """将绝对路径转换为相对于project_root的路径（仅文件名）"""
        if not self.project_root:
            return absolute_path
        import os
        return os.path.basename(absolute_path)
    
    
    async def _init_mcp_manager(self):
        """初始化MCP manager用于调用decision agent"""
        if self._mcp_manager is not None:
            return
        
        if not self.decision_agent_id:
            return
        
        try:
            # 导入必要的模块
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
            from aip.config_parser import load_config
            from aip.runtime_context import RuntimeContext, get_runtime_context
            
            # 加载decision agent的配置文件
            decision_config_path = Path(__file__).parent.parent / 'config' / 'rag' / 'decision_agent.yaml'
            
            if not decision_config_path.exists():
                print(f"[ERROR] Decision agent config not found: {decision_config_path}", file=sys.stderr)
                return
            
            # 加载配置（包括models, prompts, agents）
            print(f"[DEBUG] Loading config: {decision_config_path}", file=sys.stderr)
            config_path_resolved, config = load_config(str(decision_config_path))
            
            # 获取或创建runtime context
            try:
                runtime_ctx = get_runtime_context()
            except:
                # 如果runtime context不存在，创建一个新的
                runtime_ctx = RuntimeContext()
            
            # 加载配置到runtime context
            runtime_ctx.load_config(config_path_resolved)
            
            # 验证decision agent已加载
            if self.decision_agent_id not in runtime_ctx.active_agents:
                print(f"[ERROR] Decision agent '{self.decision_agent_id}' not found in active agents", file=sys.stderr)
                print(f"[ERROR] Available agents: {list(runtime_ctx.active_agents.keys())}", file=sys.stderr)
                return
            
            # 创建MCP manager（使用runtime context中的MCP servers配置）
            from aip.mcp.manager import MCPManager
            
            # 只包含decision agent的subagent server
            decision_servers = {}
            if self.decision_agent_id in runtime_ctx.active_mcp_servers:
                decision_servers[self.decision_agent_id] = runtime_ctx.active_mcp_servers[self.decision_agent_id]
            
            if not decision_servers:
                print(f"[ERROR] Decision agent MCP server config not found", file=sys.stderr)
                return
            
            self._mcp_manager = MCPManager(decision_servers)
            
            # 发现工具
            await self._mcp_manager.discover_all()
            
            print(f"[MCP] Initialized decision agent: {self.decision_agent_id}", file=sys.stderr)
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize MCP manager: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self._mcp_manager = None
    
    async def call_decision_agent(self, original_query: str, tool_result: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        调用decision agent分析工具结果（使用MCP subagent）
        
        Args:
            original_query: 用户原始查询
            history: 之前的工具调用历史 [{"tool": "grep_files", "result": "..."}, ...]
            tool_result: 工具执行结果文本
            
        Returns:
            Decision分析文本
        """
        
        # 如果没有配置decision agent，不调用
        if not self.decision_agent_id:
            return ""
        
        try:
            # 初始化MCP manager
            await self._init_mcp_manager()
            
            if not self._mcp_manager:
                return ""
            
            # 导入必要的类
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
            from aip.mcp.types import ToolCall
            
            # 构建指令 - 包含历史信息和工具调用次数
            instruction_parts = []
            
            # 计算工具调用次数
            num_tools_called = len(history) if history else 0
            
            # 如果有历史，先展示历史
            if history and len(history) > 0:
                instruction_parts.append("=== PREVIOUS TOOL CALLS ===")
                for i, hist_item in enumerate(history, 1):
                    tool_name = hist_item.get('tool', 'unknown')
                    result_preview = hist_item.get('result', '')[:500]
                    instruction_parts.append(f"\nCall {i}: {tool_name}")
                    instruction_parts.append(f"Result preview: {result_preview}...")
                instruction_parts.append("\n=== CURRENT TOOL CALL ===")
            
            # 当前查询和结果（严格限制长度）
            instruction_parts.append(f"Original Query: {original_query}")
            instruction_parts.append(f"Number of Tools Called: {num_tools_called}")
            
            # 严格限制tool_result长度
            max_tool_result_chars = 2000
            truncated_tool_result = tool_result[:max_tool_result_chars]
            if len(tool_result) > max_tool_result_chars:
                truncated_tool_result += "\n...[result truncated]"
            instruction_parts.append(f"\nCurrent Tool Result:\n{truncated_tool_result}")
            
            instruction = "\n".join(instruction_parts)
            
            # 再次截断过长的指令（最终保护）
            max_chars = 6000
            if len(instruction) > max_chars:
                instruction = instruction[:max_chars] + "\n...[message truncated]"
            
            print(f"[DECISION_AGENT] Calling subagent for query: {original_query[:50]}... (history: {len(history) if history else 0} calls, total tools: {num_tools_called})", file=sys.stderr)
            
            # 创建tool call
            tool_call = ToolCall(
                id="decision_" + str(hash(original_query))[:8],
                name=self.decision_agent_id,
                arguments={"instruction": instruction}
            )
            
            # 调用subagent
            result_text = ""
            async for result in self._mcp_manager.dispatch(tool_call):
                if result.error:
                    print(f"[DECISION_AGENT] Error: {result.error}", file=sys.stderr)
                    return ""
                result_text = result.result
            
            # 格式化输出
            if result_text:
                formatted = f"\n--- DECISION ANALYSIS ---\n{result_text.strip()}\n"
                print(f"[DECISION_AGENT] Response: {result_text[:100]}...", file=sys.stderr)
                return formatted
            
            return ""
            
        except Exception as e:
            print(f"[DECISION_AGENT] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return ""
        
    async def connect(self):
        """连接到上游MCP服务器"""
        server_params = StdioServerParameters(
            command=self.upstream_command,
            args=self.upstream_args
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        await self.session.initialize()
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出过滤后的工具"""
        if not self.session:
            raise RuntimeError("Not connected to upstream server")
            
        # 获取所有工具
        result = await self.session.list_tools()
        
        # 过滤工具
        filtered_tools = [
            tool for tool in result.tools
            if tool.name in self.allowed_tools
        ]
        
        # 修改grep_files工具的输入schema
        tools_list = []
        for tool in filtered_tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            
            # 如果是grep_files工具，修改其输入schema以接受关键词列表的列表
            if tool.name == "grep_files":
                # 创建新的schema，只暴露keywords和reason_refine参数
                # is_regex在代码中处理，path从PROJECT_ROOT读取
                
                # 修改描述
                tool_dict["description"] = (
                    "Search for files containing keywords with AND logic between keyword groups. "
                    "Accepts a list of keyword lists. Each inner list uses OR logic (any keyword matches), "
                    "and the outer lists use AND logic (intersection of results). "
                    "Only Chinese keywords with spaces will be split into separate terms. "
                    "English phrases remain intact. "
                    "Example: [['ai', 'artificial intelligence', '机器 学习'], ['hong kong', '香港']] "
                    "will search for (ai|artificial intelligence|机器|学习) AND (hong kong|香港) "
                    "and return files that match both patterns. "
                    "Path is automatically set to PROJECT_ROOT. "
                    "MUST provide reason_refine to explain your search strategy."
                )
                
                # 创建简化的输入schema，只包含keywords、reason_refine和original_query
                tool_dict["inputSchema"] = {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "array", "items": {"type": "string"}},
                                    {"type": "object"},
                                    {"type": "string"}
                                ]
                            },
                            "description": "List of keyword lists. Inner lists use OR logic, outer lists use AND logic (intersection). Supports mixed formats: lists, dicts, or strings. Only Chinese keywords with spaces will be split into separate terms; English phrases remain intact."
                        },
                        "reason_refine": {
                            "type": "string",
                            "description": "REQUIRED. Explain your search strategy and refinement reasoning. For initial search, explain keyword selection. For refined search, explain what went wrong in previous attempt and how you're adjusting the strategy (e.g., 'No matches found, adding synonyms', 'Too many results, adding specific keywords')."
                        },
                        "original_query": {
                            "type": "string",
                            "description": "REQUIRED. The original user query/question. This is used for relevance analysis and decision making."
                        }
                    },
                    "required": ["keywords", "reason_refine", "original_query"]
                }
            
            tools_list.append(tool_dict)
        
        # 添加read_file_char工具（如果在allowed_tools中）
        if "read_file_char" in self.allowed_tools and READ_FILE_CHAR_AVAILABLE:
            read_file_char_tool = {
                "name": "read_file_char",
                "description": (
                    "Read content from a file starting at a specific character position (anchor point). "
                    "The tool automatically reads context before and after the anchor position. "
                    "Use the OFFSET start value from grep_files output as the anchor (e.g., '7' from 'OFFSET 7-16'). "
                    f"The tool will read {self.read_file_char_context} characters before AND after the anchor. "
                    "Example: read_file_char(path='file.txt', start_char=100, original_query='your question')"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file (use the path from grep_files output)"
                        },
                        "start_char": {
                            "type": "integer",
                            "description": "Anchor character position - use the OFFSET start from grep_files (e.g., '7' from 'OFFSET 7-16')"
                        },
                        "original_query": {
                            "type": "string",
                            "description": "REQUIRED. The original user query/question for relevance analysis."
                        }
                    },
                    "required": ["path", "start_char", "original_query"]
                }
            }
            tools_list.append(read_file_char_tool)
        
        return tools_list
    def _log_tool_text(self, text: str, tool_name: str):
        """Save tool call text to tool_calls.txt (one line per call)"""
        try:
            # 保存到项目根目录的tool_calls.txt
            workspace_root = os.environ.get('WORKSPACE_ROOT', '.')
            log_file = Path(workspace_root) / f'tool_calls_{tool_name}.txt'
            
            # 追加模式写入，每行一个工具调用
            with open(log_file, 'a', encoding='utf-8') as f:
                # 清理文本：移除多余的空白和换行
                clean_text = ' '.join(text.split())
                f.write(clean_text + '\n')
            
            print(f"[LOG] Appended to {log_file}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to log tool call: {e}", flush=True)
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], text_input: Optional[str] = None) -> Any:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数（JSON格式的正常tool call）
            text_input: 大模型生成的原始文本（可选）
            
        Returns:
            工具执行结果
            
        支持两种模式：
        1. text_input模式：传递text_input参数，从文本中解析工具调用
        2. JSON模式：传递arguments参数，直接使用结构化参数
        """
        if not self.session:
            raise RuntimeError("Not connected to upstream server")
            
        if tool_name not in self.allowed_tools:
            raise ValueError(f"Tool '{tool_name}' is not allowed")
        
        # 处理text_input模式
        if text_input:
            print(f"[TOOL_CALL] Received text_input for {tool_name}: {text_input[:100]}...", file=sys.stderr)
            
            if tool_name == "grep_files":
                try:
                    parsed = parse_grep_files_call(text_input)
                    keywords = parsed.get('keywords', [])
                    self._log_tool_text(str(keywords), tool_name)
                    # 更新arguments，不要覆盖原有参数（如path）
                    arguments['keywords'] = keywords
                    
                    # 尝试提取可选参数
                    reason_match = re.search(r'reason_refine\s*=\s*["\']([^"\']+)["\']', text_input)
                    query_match = re.search(r'original_query\s*=\s*["\']([^"\']+)["\']', text_input)
                    if reason_match:
                        arguments['reason_refine'] = reason_match.group(1)
                    if query_match:
                        arguments['original_query'] = query_match.group(1)
                except Exception as e:
                    print(f"[ERROR] Failed to parse grep_files call: {e}", file=sys.stderr)
                    traceback.print_exc()
                    # 回退：空关键词
                    arguments['keywords'] = [[]]
            elif tool_name == "read_file_char":
                try:
                    path_match = re.search(r'path\s*=\s*["\']([^"\']+)["\']', text_input)
                    if path_match:
                        arguments['path'] = path_match.group(1)
                    start_match = re.search(r'start_char\s*=\s*(\d+)', text_input)
                    if start_match:
                        arguments['start_char'] = int(start_match.group(1))
                    query_match = re.search(r'original_query\s*=\s*["\']([^"\']+)["\']', text_input)
                    if query_match:
                        arguments['original_query'] = query_match.group(1)
                except Exception as e:
                    print(f"[ERROR] Failed to parse read_file_char call: {e}", file=sys.stderr)
                    traceback.print_exc()
            
        
        # 处理raw_text参数（兼容旧模式）
        if "raw_text" in arguments:
            raw_text = arguments.pop("raw_text")
            print(f"[TOOL_CALL] Received raw_text for {tool_name}: {raw_text[:100]}...", file=sys.stderr)
            
            
            if tool_name == "grep_files":
                try:
                    parsed = parse_grep_files_call(raw_text)
                    keywords = parsed.get('keywords', [])
                    arguments['keywords'] = keywords
                    self._log_tool_text(str(keywords), tool_name)
                    reason_match = re.search(r'reason_refine\s*=\s*["\']([^"\']+)["\']', raw_text)
                    query_match = re.search(r'original_query\s*=\s*["\']([^"\']+)["\']', raw_text)
                    if reason_match:
                        arguments['reason_refine'] = reason_match.group(1)
                    if query_match:
                        arguments['original_query'] = query_match.group(1)
                except Exception as e:
                    print(f"[ERROR] Failed to parse grep_files call: {e}", file=sys.stderr)
                    traceback.print_exc()
                    arguments['keywords'] = [[]]
            elif tool_name == "read_file_char":
                try:
                    path_match = re.search(r'path\s*=\s*["\']([^"\']+)["\']', raw_text)
                    if path_match:
                        arguments['path'] = path_match.group(1)
                    start_match = re.search(r'start_char\s*=\s*(\d+)', raw_text)
                    if start_match:
                        arguments['start_char'] = int(start_match.group(1))
                    query_match = re.search(r'original_query\s*=\s*["\']([^"\']+)["\']', raw_text)
                    if query_match:
                        arguments['original_query'] = query_match.group(1)
                except Exception as e:
                    print(f"[ERROR] Failed to parse read_file_char call: {e}", file=sys.stderr)
                    traceback.print_exc()
            
        # 处理grep_files的keywords参数
        if tool_name == "grep_files" and "keywords" in arguments:
            try:
                keywords = arguments.pop("keywords")
                reason_refine = arguments.pop("reason_refine", "No reason provided")
                original_query = arguments.pop("original_query", "")
                # 忽略所有其他参数（如is_regex、offset、path等）
                # 清空arguments，只保留我们需要的参数
                arguments.clear()
                
                # 记录refinement原因到stderr（用于调试）
                print(f"[GREP_FILES] Query: {original_query}", file=sys.stderr)
                print(f"[GREP_FILES] Reason: {reason_refine}", file=sys.stderr)
                print(f"[GREP_FILES] Original Keywords: {keywords}", file=sys.stderr)
                
                # 规范化输入为list of list格式
                normalized_keywords = normalize_keywords_input(keywords)
                print(f"[GREP_FILES] Normalized Keywords: {normalized_keywords}", file=sys.stderr)
                
                
                
                # List of list格式 - 使用AND逻辑（交集）
                # 每个内部list使用OR逻辑，外部lists之间使用AND逻辑（交集）
                results = []
                
                # 为每个keyword组执行grep_files
                
                for keyword_group in normalized_keywords:
                    pattern = convert_keywords_to_pattern(keyword_group)
                    group_arguments = {}
                    group_arguments["pattern"] = pattern
                    
                    # is_regex在代码中设置，不作为参数暴露
                    # 如果pattern包含|（OR逻辑），必须使用regex模式
                    if '|' in pattern:
                        group_arguments["is_regex"] = True
                    else:
                        group_arguments["is_regex"] = False
                    
                    # path从PROJECT_ROOT读取，不作为参数暴露
                    if self.project_root:
                        group_arguments["path"] = self.project_root
                    
                    # 添加results_limit配置
                    if self.grep_results_limit is not None:
                        group_arguments["results_limit"] = self.grep_results_limit
                    
                    # 如果配置了context_char，grep_files会使用字符级上下文
                    # 移除行级context参数（避免冲突）
                    if self.context_char > 0:
                        pass
                    
                    # 执行grep
                    try:
                        result = await self.session.call_tool(tool_name, group_arguments)
                    except Exception as e:
                        # 如果执行出错，返回错误信息而不是空输出
                        error_result = type('Result', (), {})() 
                        error_result.content = [TextContent(f"Error executing grep for keywords {keyword_group}: {str(e)}")]
                        return error_result

                    # 检查是否有匹配
                    has_matches = True
                    if hasattr(result, 'content'):
                        for content in result.content:
                            if hasattr(content, 'text') and 'No matches found' in content.text:
                                has_matches = False
                                break
                    
                    results.append({
                        'result': result,
                        'has_matches': has_matches,
                        'keywords': keyword_group,
                        'pattern': pattern
                    })
                
                # 如果所有组都没有匹配，返回第一个结果（确保有输出）
                if all(not item['has_matches'] for item in results):
                    if results:
                        return results[0]['result']
                    else:
                        # 如果results为空，返回错误信息
                        error_result = type('Result', (), {})() 
                        error_result.content = [TextContent("Error: No grep results generated. Please check your keywords.")]
                        return error_result
                
                # 提取有匹配的结果的文件路径
                file_path_sets = []
                valid_results = []
                invalid_results = []
                
                for item in [r for r in results if r['has_matches']]:
                    paths = extract_file_paths_from_result(item['result'])
                    if paths:
                        file_path_sets.append(paths)
                        valid_results.append(item)
                for item in [r for r in results if not r['has_matches']]:
                    invalid_results.append(item['pattern'])
                
                # 如果没有提取到路径，返回所有匹配结果的合并
                if not file_path_sets:
                    # 如果valid_results也为空，说明没有任何匹配
                    if not valid_results:
                        # 返回第一个结果（可能是无匹配的消息）
                        if results:
                            return results[0]['result']
                        # 如果连results都为空，创建一个空结果
                        from mcp.types import TextContent as MCPTextContent
                        empty_result = type('Result', (), {})()
                        empty_result.content = [TextContent("No results found")]
                        return empty_result
                    
                    all_content = [TextContent("Note: Unable to extract file paths. Showing all results:\n\n")]
                    for i, item in enumerate(valid_results):
                        all_content.append(TextContent(f"--- Group {i+1}: {item['keywords']} ---\n"))
                        if hasattr(item['result'], 'content'):
                            all_content.extend(item['result'].content)
                        if i < len(valid_results) - 1:
                            all_content.append(TextContent("\n"))
                    
                    combined_result = valid_results[0]['result']
                    combined_result = copy.copy(combined_result)
                    combined_result.content = all_content
                    return combined_result
                
                # 计算并集（匹配任意关键词组的文件）
                union_paths = set()
                for path_set in file_path_sets:
                    union_paths.update(path_set)
                
                # 计算每个文件在所有grep结果中出现的次数（匹配的group数量，用于排序）
                file_frequencies = {}
                for path in union_paths:
                    frequency = sum(1 for path_set in file_path_sets if path in path_set)
                    file_frequencies[path] = frequency
                
                # 如果并集结果超过max_output_files，返回top max_output_files个文档（按匹配group数量排序）
                                
                if len(union_paths) > self.max_output_files:
                    # 按频率排序，选择top max_output_files个文档（优先返回匹配更多groups的文件）
                    sorted_paths = sorted(
                        union_paths,
                        key=lambda path: (-file_frequencies.get(path, 0), path)
                    )
                    top_paths = set(sorted_paths[:self.max_output_files])
                    
                    # 收集每个组的结果数量和报告信息
                    report_lines = [
                        f"Search Reason: {reason_refine}",
                        "",
                        f"Too many results: {len(union_paths)} files match (limit: {self.max_output_files}).",
                        f"Returning top {self.max_output_files} files.",
                        "",
                        "Results per keyword group:",
                        ""
                    ]
                    
                    for i, item in enumerate(valid_results, 1):
                        paths = extract_file_paths_from_result(item['result'])
                        count = len(paths)
                        report_lines.append(f"  {item['keywords']}: {count} results")
                    
                    report_lines.append("")
                    report_lines.append(f"Showing top {self.max_output_files} files with highest keyword frequency:")
                    report_lines.append("")
                    
                    # 使用top_paths继续处理
                    result = await merge_and_filter_results_with_context(
                        self,
                        valid_results,
                        top_paths,
                        file_frequencies,
                        self.max_output_files,
                        self.context_char,
                        self.max_tokens,
                        "\n".join(report_lines)
                    )
                    
                    # 添加decision分析（too many results分支）
                    if original_query and hasattr(result, 'content') and result.content:
                        result_text = ""
                        for content in result.content:
                            if hasattr(content, 'text'):
                                result_text += content.text
                        
                        # 截断过长的result_text避免token超限
                        max_result_chars = 5000
                        if len(result_text) > max_result_chars:
                            result_text = result_text[:max_result_chars] + "\n...[result truncated due to length]"
                        
                        # 获取历史并调用decision
                        history = self._get_or_create_history(original_query)
                        decision_output = await self.call_decision_agent(original_query, result_text, history)
                        if decision_output:
                            result.content.append(TextContent(decision_output))
                        
                        # 记录到历史（只保存前2000字符）
                        self._add_to_history(original_query, "grep_files", result_text[:2000])
                    
                    return result
                
                # 返回并集结果，按频率排序并限制数量
                # 为每个关键词组返回匹配行的上下文
                # 始终使用 merge_and_filter_results_with_context
                                
                result = await merge_and_filter_results_with_context(
                    self,
                    valid_results, 
                    union_paths, 
                    file_frequencies,
                    self.max_output_files,
                    self.context_char,
                    self.max_tokens,
                    reason_refine
                )
                
                # 添加decision分析
                if original_query and hasattr(result, 'content') and result.content:
                    # 提取结果文本
                    result_text = ""
                    for content in result.content:
                        if hasattr(content, 'text'):
                            result_text += content.text
                    
                    # 截断过长的result_text避免token超限
                    max_result_chars = 5000
                    if len(result_text) > max_result_chars:
                        result_text = result_text[:max_result_chars] + "\n...[result truncated due to length]"
                    
                    # 获取历史并调用decision agent
                    history = self._get_or_create_history(original_query)
                    decision_output = await self.call_decision_agent(original_query, result_text, history)
                    if decision_output:
                        result.content.append(TextContent(decision_output))
                    
                    # 记录到历史（只保存前2000字符）
                    self._add_to_history(original_query, "grep_files", result_text[:2000])
                
                return result
            
            except Exception as e:
                # 捕获所有错误，返回错误信息而不是空输出
                error_result = type('Result', (), {})() 
                error_result.content = [TextContent(f"Error in grep_files processing: {str(e)}\n\nKeywords: {keywords}\nReason: {reason_refine}")]
                return error_result
        
        # 添加results_limit配置
        if tool_name == "grep_files" and self.grep_results_limit is not None and "results_limit" not in arguments:
            arguments["results_limit"] = self.grep_results_limit
        
        # 处理read_file_char的特殊逻辑
        if tool_name == "read_file_char":
            if not READ_FILE_CHAR_AVAILABLE:
                error_result = type('Result', (), {})()
                error_result.content = [TextContent("Error: read_file_char module not available")]
                return error_result
            
            path = arguments.get("path")
            start_char = arguments.get("start_char")
            original_query = arguments.get("original_query", "")
            
            if not path or start_char is None:
                error_result = type('Result', (), {})()
                error_result.content = [TextContent("Error: path and start_char are required")]
                return error_result
            
            # 如果是相对路径，尝试基于project_root解析
            if self.project_root and not os.path.isabs(path):
                candidate_path = os.path.join(self.project_root, path)
                if os.path.exists(candidate_path):
                    path = candidate_path
            
            try:
                # 使用read_file_char_sync读取文件
                content, metadata = read_file_char_sync(
                    path=path,
                    start_char=start_char,
                    context_chars=self.read_file_char_context
                )
                
                # 格式化输出
                formatted_output = format_read_file_char_output(path, content, metadata)
                
                result = type('Result', (), {})()
                result.content = [TextContent(formatted_output)]
                
                # 添加decision分析
                if original_query:
                    # 截断过长的formatted_output避免token超限
                    max_result_chars = 5000
                    truncated_output = formatted_output
                    if len(formatted_output) > max_result_chars:
                        truncated_output = formatted_output[:max_result_chars] + "\n...[result truncated due to length]"
                    
                    # 获取历史并调用decision
                    history = self._get_or_create_history(original_query)
                    decision_output = await self.call_decision_agent(original_query, truncated_output, history)
                    if decision_output:
                        result.content.append(TextContent(decision_output))
                    
                    # 记录到历史（只保存前2000字符）
                    self._add_to_history(original_query, "read_file_char", formatted_output[:2000])
                
                return result
                
            except FileNotFoundError:
                error_result = type('Result', (), {})()
                error_result.content = [TextContent(f"Error: File not found: {path}")]
                return error_result
            except Exception as e:
                error_result = type('Result', (), {})()
                error_result.content = [TextContent(f"Error reading file: {str(e)}")]
                return error_result
        
        return await self.session.call_tool(tool_name, arguments)
    
    async def close(self):
        """关闭连接"""
        await self.exit_stack.aclose()


async def run_stdio_server(filter_instance: MCPToolFilter):
    """运行标准输入/输出MCP服务器"""
    
    while True:
        try:
            # 读取JSON-RPC请求
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            request = json.loads(line.strip())
            
            # 处理请求
            response = await handle_request(filter_instance, request)
            
            # 发送响应
            print(json.dumps(response), flush=True)
            
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                },
                "id": 0
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "id": 0
            }
            print(json.dumps(error_response), flush=True)


async def handle_request(filter_instance: MCPToolFilter, request: Dict[str, Any]) -> Dict[str, Any]:
    """处理JSON-RPC请求"""
    
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "filesystem-filter",
                    "version": "1.0.0"
                }
            }
            
        elif method == "tools/list":
            tools = await filter_instance.list_tools()
            result = {"tools": tools}
            
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            tool_result = await filter_instance.call_tool(tool_name, arguments)
            
            # 转换结果格式
            if hasattr(tool_result, 'content'):
                content_list = []
                for content in tool_result.content:
                    if isinstance(content, TextContent):
                        # 自定义的TextContent对象
                        content_list.append({
                            "type": content.type,
                            "text": content.text
                        })
                    elif hasattr(content, 'text'):
                        # MCP库返回的对象
                        content_list.append({
                            "type": getattr(content, 'type', 'text'),
                            "text": content.text
                        })
                    elif isinstance(content, str):
                        # 字符串
                        content_list.append({
                            "type": "text",
                            "text": content
                        })
                    else:
                        # 其他对象，转为字符串
                        content_list.append({
                            "type": "text",
                            "text": str(content)
                        })
                result = {"content": content_list}
            else:
                # 没有content属性，直接转换
                result = {
                    "content": [{
                        "type": "text",
                        "text": str(tool_result)
                    }]
                }
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": str(e)
            },
            "id": request_id
        }


async def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='MCP工具过滤器 - 过滤filesystem MCP服务器的工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--tools',
        type=str,
        default='read_file,read_multiple_files,grep_files',
        help='允许的工具列表，逗号分隔 (默认: read_file,read_multiple_files,grep_files)'
    )
    parser.add_argument(
        '--grep_results_limit',
        type=int,
        default=None,
        help='grep_files工具的结果数量限制 (默认: 无限制)'
    )
    parser.add_argument(
        '--max_output_files',
        type=int,
        default=5,
        help='最终输出的文件及content的最大个数 (默认: 5)'
    )

    parser.add_argument(
        '--context_char',
        type=int,
        default=100,
        help='上下文字符数（匹配文本前后各多少字符） (默认: 100)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=None,
        help='输出的最大token数量，超过则截断并警告 (默认: 无限制)'
    )
    
    parser.add_argument(
        '--project_root',
        type=str,
        default=None,
        help='项目根目录，用于转换绝对路径为相对路径 (默认: None，显示绝对路径)'
    )
    
    parser.add_argument(
        '--read_file_char_context',
        type=int,
        default=300,
        help='read_file_char工具读取的上下文字符数 (默认: 300)'
    )
    
    parser.add_argument(
        '--decision_prompt_path',
        type=str,
        default=None,
        help='Decision agent YAML配置文件路径（包含prompt和model配置） (默认: None，不启用decision agent)'
    )
    
    # 分离已知参数和上游服务器参数
    args, upstream_args = parser.parse_known_args()
    
    # 解析允许的工具列表
    allowed_tools = set(tool.strip() for tool in args.tools.split(',') if tool.strip())
    
    # 清理上游参数，移除 '--' 分隔符
    if upstream_args and upstream_args[0] == '--':
        upstream_args = upstream_args[1:]
    
    print(f"Allowed tools: {allowed_tools}", file=sys.stderr)
    print(f"Grep results limit: {args.grep_results_limit}", file=sys.stderr)
    print(f"Max output files: {args.max_output_files}", file=sys.stderr)
    print(f"Context chars: {args.context_char}", file=sys.stderr)
    print(f"Max tokens: {args.max_tokens}", file=sys.stderr)
    print(f"Project root: {args.project_root}", file=sys.stderr)
    print(f"Read file char context: {args.read_file_char_context}", file=sys.stderr)
    print(f"Decision config path: {args.decision_prompt_path}", file=sys.stderr)
    print(f"Upstream args: {upstream_args}", file=sys.stderr)
    
    # 配置上游filesystem服务器
    # 使用 python3 -m mcp_filesystem 而不是 npx
    upstream_command = "python3"
    
    # 创建过滤器实例
    filter_instance = MCPToolFilter(
        upstream_command=upstream_command,
        upstream_args=upstream_args,
        allowed_tools=allowed_tools,
        grep_results_limit=args.grep_results_limit,
        max_output_files=args.max_output_files,
        context_char=args.context_char,
        max_tokens=args.max_tokens,
        project_root=args.project_root,
        read_file_char_context=args.read_file_char_context,
        decision_prompt_path=args.decision_prompt_path
    )
    
    try:
        # 连接到上游服务器
        await filter_instance.connect()
        
        # 运行stdio服务器
        await run_stdio_server(filter_instance)
        
    finally:
        await filter_instance.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
        sys.exit(0)
