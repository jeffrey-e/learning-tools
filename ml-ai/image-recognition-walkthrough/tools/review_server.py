"""A small local server for checking the photo labels before training.

Serves tools/review.html with the candidate photos, and saves the labels you set to data/labels.json.
It only listens on localhost.

Usage:  python3 -P tools/review_server.py      then open http://localhost:8001/
"""
import http.server
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "labels.json"
ALLOWED = {"husky", "malamute", "drop"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):   # the review page (keeping ?ids=… for the page's own filter)
            self.path = "/tools/review.html" + ("?" + query if query else "")
        if self.path == "/assets/site.css":   # the site's shared styles, two folders up
            css = (ROOT.parent.parent / "assets" / "site.css").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/css")
            self.send_header("Content-Length", str(len(css)))
            self.end_headers()
            return self.wfile.write(css)
        if self.path == "/labels.json":
            body = LABELS.read_bytes() if LABELS.exists() else b"{}"
            return self.reply(200, body)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/labels.json":
            return self.reply(404, b'{"error":"not found"}')
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ok = isinstance(data, dict) and all(
                isinstance(k, str) and isinstance(v, dict) and v.get("label") in ALLOWED for k, v in data.items())
        except (ValueError, TypeError, KeyError):
            ok = False
        if not ok:
            return self.reply(400, b'{"error":"labels must map ids to {label: husky | malamute | drop}"}')
        LABELS.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        return self.reply(200, json.dumps({"saved": len(data)}).encode())

    def reply(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("Review the labels at http://localhost:8001/  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
