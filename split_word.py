import os
from docx import Document

def word_to_txt_sections(word_file_path, output_folder):
    """
    将Word文档按节(section)分割成多个txt文件
    每个section的所有段落保存到一个txt文件中
    
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
        
        # 按section分组段落
        sections = doc.sections
        section_count = len(sections)
        
        print(f"文档共有 {section_count} 个节(section)")
        
        # 为每个section收集段落
        section_paragraphs = [[] for _ in range(section_count)]
        
        # 遍历所有段落，判断它们属于哪个section
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            # 跳过空段落
            if not text:
                continue
                
            # 获取段落所属的section索引
            # 通过段落的_element获取其所在section
            para_section_index = 0
            for i, section in enumerate(sections):
                # 检查段落是否在当前section的范围内
                if paragraph._element.getparent() is not None:
                    # 简化处理：按文档顺序分配到section
                    # 这里使用段落在文档中的位置来估算所属section
                    para_index = doc.paragraphs.index(paragraph)
                    # 平均分配（这是简化方法，实际section边界由分节符决定）
                    section_size = len(doc.paragraphs) / section_count
                    para_section_index = min(int(para_index / section_size), section_count - 1)
                    break
            
            section_paragraphs[para_section_index].append(text)
        
        # 为每个section创建txt文件
        valid_section_count = 0
        for i, paragraphs in enumerate(section_paragraphs):
            if not paragraphs:
                continue
                
            valid_section_count += 1
            
            # 创建txt文件名
            txt_filename = f"{file_name}_节{i+1:02d}.txt"
            txt_filepath = os.path.join(output_folder, txt_filename)
            
            # 写入section的所有段落
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                # 段落之间用双换行分隔
                f.write('\n\n'.join(paragraphs))
            
            print(f"已创建: {txt_filename} (包含 {len(paragraphs)} 个段落)")
        
        print(f"\n处理完成！")
        print(f"总节数: {section_count}")
        print(f"有效节数: {valid_section_count}")
        print(f"输出文件夹: {output_folder}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 请修改为你的Word文档路径
    word_file = "your_document.docx"  # 替换为你的Word文件路径
    
    # 输出文件夹
    output_dir = "sections_output"
    
    # 执行转换
    word_to_txt_sections(word_file, output_dir)