import os
from docx import Document

def word_to_txt_paragraphs(word_file_path, output_folder):
    """
    将Word文档按段落分割成多个txt文件
    
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
        
        # 处理每个段落
        paragraph_count = 0
        valid_paragraph_count = 0
        
        for i, paragraph in enumerate(doc.paragraphs):
            paragraph_count += 1
            text = paragraph.text.strip()
            
            # 跳过空段落
            if not text:
                continue
            
            valid_paragraph_count += 1
            
            # 创建txt文件名
            txt_filename = f"{file_name}_段落{valid_paragraph_count:03d}.txt"
            txt_filepath = os.path.join(output_folder, txt_filename)
            
            # 写入段落内容到txt文件
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"已创建: {txt_filename}")
        
        print(f"\n处理完成！")
        print(f"总段落数: {paragraph_count}")
        print(f"有效段落数: {valid_paragraph_count}")
        print(f"输出文件夹: {output_folder}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 请修改为你的Word文档路径
    word_file = "your_document.docx"  # 替换为你的Word文件路径
    
    # 输出文件夹
    output_dir = "paragraphs_output"
    
    # 执行转换
    word_to_txt_paragraphs(word_file, output_dir)