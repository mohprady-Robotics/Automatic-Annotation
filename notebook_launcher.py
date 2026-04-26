from __future__ import annotations

import os
import time
from pathlib import Path
from subprocess import Popen

from dotenv import load_dotenv


def launch(
    host: str = "0.0.0.0",
    port: int = 8000,
    image_dir: str | None = None,
    enable_ngrok: bool = False,
    ngrok_auth_token: str | None = None,
) -> dict:
    """
    Launch the FastAPI annotation server from a notebook environment.

    Args:
        host: Bind address for uvicorn.
        port: Port for uvicorn.
        image_dir: Optional hint printed in notebook for the UI input field.
        enable_ngrok: If True, creates an ngrok tunnel and returns the public URL.
        ngrok_auth_token: Optional ngrok token (or set NGROK_AUTH_TOKEN env var).

    Returns:
        Dict with local_url and optionally public_url.
    """

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    load_dotenv(project_root / ".env")
    load_dotenv(project_root / ".env.colab")
    env_file_override = os.getenv("ANNOTATION_APP_ENV_FILE")
    if env_file_override:
        load_dotenv(env_file_override)

    command = [
        "python3",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--reload",
    ]
    process = Popen(command)  # noqa: S603,S607
    time.sleep(2)

    result = {
        "pid": process.pid,
        "local_url": f"http://127.0.0.1:{port}",
        "image_dir_hint": image_dir or "",
    }

    if enable_ngrok:
        from pyngrok import ngrok

        auth_token = ngrok_auth_token or os.getenv("NGROK_AUTH_TOKEN")
        if auth_token:
            ngrok.set_auth_token(auth_token)
        tunnel = ngrok.connect(port, "http")
        result["public_url"] = tunnel.public_url

    return result
