import http.server
import socketserver

class UTF8HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='./', **kwargs)
    
    def send_head(self):
        path = self.translate_path(self.path)
        if path.endswith('.txt') or path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            return open(path, 'rb')
        return super().send_head()

with socketserver.TCPServer(('', 8020), UTF8HTTPRequestHandler) as httpd:
    print('中文UTF-8服务器运行在端口 8020')
    httpd.serve_forever()