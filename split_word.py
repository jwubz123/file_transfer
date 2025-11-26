import os
from docx import Document
import re

def word_to_txt_subsections(word_file_path, output_folder):
    """
    将Word文档按目录的子章节(subsection)分割成多个txt文件
    每个subsection的所有段落保存到一个txt文件中
    
    参数:
    word_file_path: Word文档路径
    output_folder: 输出文件夹路径
    """
    try:
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 读取Word文档
        doc = Document(word_file_path)
        
        # 获取Word文档名称（不含扩展名）
        file_name = os.path.splitext(os.path.basename(word_file_path))[0]
        
        # 存储所有subsection及其内容
        subsections = []
        current_subsection = None
        current_paragraphs = []
        
        # 遍历所有段落
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            
            # 跳过空段落
            if not text:
                continue
            
            # 检查是否是标题（通过样式判断）
            style_name = paragraph.style.name
            is_heading = (style_name.startswith('Heading') or 
                         style_name.startswith('标题') or
                         '标题' in style_name)
            
            # 也可以通过编号格式判断（如 1.1, 1.2, 2.1 等）
            is_numbered_heading = bool(re.match(r'^\d+(\.\d+)+', text))
            
            if is_heading or is_numbered_heading:
                # 如果已经有当前subsection，保存它
                if current_subsection and current_paragraphs:
                    subsections.append({
                        'title': current_subsection,
                        'content': current_paragraphs.copy()
                    })
                
                # 开始新的subsection
                current_subsection = text
                current_paragraphs = []
            else:
                # 普通段落，添加到当前subsection
                if current_subsection is not None:
                    current_paragraphs.append(text)
        
        # 保存最后一个subsection
        if current_subsection and current_paragraphs:
            subsections.append({
                'title': current_subsection,
                'content': current_paragraphs.copy()
            })
        
        # 为每个subsection创建txt文件
        print(f"文档共识别出 {len(subsections)} 个子章节")
        
        for i, subsection in enumerate(subsections):
            title = subsection['title']
            content = subsection['content']
            
            # 清理标题，用作文件名
            # 移除特殊字符，保留中英文、数字、空格、点号
            safe_title = re.sub(r'[^\w\s\.\-\u4e00-\u9fff]', '', title)
            safe_title = safe_title.replace(' ', '_')[:50]  # 限制长度
            
            # 创建txt文件名
            txt_filename = f"{file_name}_{i+1:02d}_{safe_title}.txt"
            txt_filepath = os.path.join(output_folder, txt_filename)
            
            # 写入内容：标题 + 内容
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(f"{title}\n\n")
                f.write('\n\n'.join(content))
            
            print(f"已创建: {txt_filename} (包含 {len(content)} 个段落)")
        
        print(f"\n处理完成！")
        print(f"总子章节数: {len(subsections)}")
        print(f"输出文件夹: {output_folder}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 请修改为你的Word文档路径
    word_file = "your_document.docx"  # 替换为你的Word文件路径
    
    # 输出文件夹
    output_dir = "subsections_output"
    
    # 执行转换
    word_to_txt_subsections(word_file, output_dir)