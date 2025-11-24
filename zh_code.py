import http.server
import socketserver
import argparse
import os

class UTF8HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        if directory is None:
            directory = os.getcwd()
        super().__init__(*args, directory=directory, **kwargs)
    
    def send_head(self):
        path = self.translate_path(self.path)
        # 为不同文件类型设置正确的Content-Type和编码
        if self.path.endswith('.html') or self.path.endswith('.htm'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            return open(path, 'rb')
        elif self.path.endswith('.txt'):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            return open(path, 'rb')
        elif self.path.endswith('.css'):
            self.send_response(200)
            self.send_header('Content-type', 'text/css; charset=utf-8')
            self.end_headers()
            return open(path, 'rb')
        elif self.path.endswith('.js'):
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript; charset=utf-8')
            self.end_headers()
            return open(path, 'rb')
        else:
            return super().send_head()

def main():
    parser = argparse.ArgumentParser(description='启动支持中文的HTTP服务器')
    parser.add_argument('--port', '-p', type=int, default=8000, help='端口号 (默认: 8000)')
    parser.add_argument('--directory', '-d', type=str, default=os.getcwd(), help='要共享的目录路径 (默认: 当前目录)')
    parser.add_argument('--bind', '-b', type=str, default='0.0.0.0', help='绑定地址 (默认: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在")
        return
    
    if not os.path.isdir(args.directory):
        print(f"错误: '{args.directory}' 不是一个目录")
        return
    
    print(f"启动服务器在: http://{args.bind}:{args.port}")
    print(f"共享目录: {os.path.abspath(args.directory)}")
    print("按 Ctrl+C 停止服务器")
    
    os.chdir(args.directory)  # 切换到目标目录
    
    with socketserver.TCPServer((args.bind, args.port), UTF8HTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == '__main__':
    main()