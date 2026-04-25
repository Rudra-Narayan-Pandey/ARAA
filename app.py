"""
ASGI entrypoint for local OpenEnv / FastAPI serving.

Examples:
    uvicorn app:app --host 0.0.0.0 --port 8000
    python serve_openenv.py
"""

from serve_openenv import app
