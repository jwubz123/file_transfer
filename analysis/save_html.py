import os
import nbformat
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter
from traitlets.config import Config


def create_html_report_stable(experiment_folder, notebook_path):
    # 读取 notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
    
    # 配置转换器，隐藏代码输入
    c = Config()
    c.HTMLExporter.exclude_input = True
    c.HTMLExporter.exclude_output_prompt = True
    c.HTMLExporter.exclude_input_prompt = True
    
    # 创建转换器
    html_exporter = HTMLExporter(config=c)
    
    # 转换 notebook
    (body, resources) = html_exporter.from_notebook_node(notebook_content)
    
    # 写入文件
    html_filename = f"report.html"
    html_path = os.path.join(experiment_folder, html_filename)
    
    writer = FilesWriter()
    writer.write(body, resources, notebook_name=html_path.replace('.html', ''))
    
    print(f"✅ 已创建无代码 HTML 报告: {html_path}")
    print("📊 报告中只包含 markdown 和输出，代码已自动隐藏")


