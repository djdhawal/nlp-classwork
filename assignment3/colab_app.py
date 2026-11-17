"""Start the Streamlit chatbot (app.py) in the background and give a URL to open it.

On Colab the app is shown through the Colab port proxy; if that fails, a
cloudflared quick tunnel gives a public https URL. Locally it is localhost.
Close the notebook's Qdrant client first: embedded Qdrant allows one process.
"""
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from assignment3 import config

APP = config.PKG_DIR / "app.py"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_procs = {}


def _healthy(port, timeout=60):
    """Wait until Streamlit answers its health check."""
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def _cloudflared(port):
    """Open a cloudflared quick tunnel and return its public URL (Linux only)."""
    exe = config.CACHE_DIR / "cloudflared"
    if not exe.exists():
        if platform.system() != "Linux":
            return None
        exe.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL, exe)
        exe.chmod(0o755)
    log = open(config.CACHE_DIR / "cloudflared.log", "w")
    _procs["tunnel"] = subprocess.Popen([str(exe), "tunnel", "--url", f"http://localhost:{port}"],
                                        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(1)
        m = re.search(r"https://[\w-]+\.trycloudflare\.com", Path(log.name).read_text())
        if m:
            return m.group(0)
    return None


def launch(port=8501, tunnel=False, fake=False):
    """Start app.py on `port` and return the URL. tunnel=True forces cloudflared on Colab."""
    stop()
    env = {**os.environ, "A3_OUT_DIR": str(config.OUT_DIR), "OLLAMA_URL": config.OLLAMA_URL,
           "PYTHONPATH": str(config.ROOT)}
    if fake:
        env["A3_FAKE_RAG"] = "1"
    config.ensure_dirs()
    log = open(config.CACHE_DIR / "streamlit.log", "w")
    _procs["app"] = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port),
         "--server.headless", "true", "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false", "--browser.gatherUsageStats", "false"],
        cwd=config.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not _healthy(port):
        print(f"Streamlit did not start; see {log.name}")
        return None

    url = f"http://localhost:{port}"
    if config.IN_COLAB:
        if not tunnel:
            try:
                from google.colab import output
                from google.colab.output import eval_js
                url = eval_js(f"google.colab.kernel.proxyPort({port})")
                output.serve_kernel_port_as_window(port, anchor_text="Open the chat app")
                print(f"App running: {url}  (works in this browser session only)")
                return url
            except Exception as e:
                print(f"Colab proxy failed ({e}); trying cloudflared")
        public = _cloudflared(port)
        if public:
            url = public
    print(f"App running: {url}")
    return url


def stop():
    """Stop the app and the tunnel, if running."""
    if not _procs:
        return
    for name in list(_procs):
        p = _procs.pop(name)
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    print("App stopped.")
