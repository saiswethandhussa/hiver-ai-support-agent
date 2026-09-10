"""
Interactive Demo & Web Server Launcher.
Usage:
    python run_demo.py
"""

import uvicorn
import os
import sys

def main():
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    print("=" * 60)
    print(f"Starting @AmazonHelp AI Support Agent Web Console on http://{host}:{port}")
    print("=" * 60)
    uvicorn.run("web.server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
