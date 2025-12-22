#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理目录中所有.txt文件：
- 如果某行以'抄送'开始，用分号分行
- 将所有抄送内容移动到文档最后
"""

import os
import sys
from pathlib import Path


def process_file(file_path):
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return False
    
    # 存储普通内容和抄送内容
    normal_lines = []
    cc_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 检查是否以"抄送"开头
        if stripped.startswith('抄送'):
            # 用分号分割
            parts = stripped.split(';')
            # 将所有分割后的部分添加到抄送行列表
            for part in parts:
                part = part.strip()
                if part:
                    cc_lines.append(part + '\n')
        else:
            # 普通行保持原样
            normal_lines.append(line)
    
    # 合并：普通内容 + 抄送内容
    if cc_lines:
        # 如果有抄送内容，先确保普通内容末尾有换行
        if normal_lines and not normal_lines[-1].endswith('\n'):
            normal_lines[-1] += '\n'
        
        # 添加一个空行作为分隔（可选）
        if normal_lines:
            normal_lines.append('\n')
        
        # 添加抄送内容
        final_content = normal_lines + cc_lines
    else:
        final_content = normal_lines
    
    # 写回文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(final_content)
        return True
    except Exception as e:
        print(f"写入文件失败 {file_path}: {e}")
        return False


def process_directory(directory):
    """处理目录中所有.txt文件"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"目录不存在: {directory}")
        return
    
    if not directory.is_dir():
        print(f"不是目录: {directory}")
        return
    
    # 查找所有.txt文件
    txt_files = list(directory.glob('**/*.txt'))
    
    if not txt_files:
        print(f"在 {directory} 中未找到.txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个.txt文件")
    
    success_count = 0
    for txt_file in txt_files:
        print(f"处理: {txt_file.relative_to(directory)}", end=' ... ')
        if process_file(txt_file):
            print("✓")
            success_count += 1
        else:
            print("✗")
    
    print(f"\n完成! 成功处理 {success_count}/{len(txt_files)} 个文件")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # 默认为当前目录
        target_dir = '.'
    
    print(f"开始处理目录: {os.path.abspath(target_dir)}")
    process_directory(target_dir)
