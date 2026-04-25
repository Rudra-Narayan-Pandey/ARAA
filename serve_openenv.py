"""
OpenEnv / FastAPI server entrypoint for ARAA.

Local run:
    python serve_openenv.py

Then connect with an OpenEnv client or run:
    uvicorn serve_openenv:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uvicorn
from openenv.core import create_fastapi_app

from env import ARAAAction, ARAAEnv, ARAAObservation


def build_env() -> ARAAEnv:
    return ARAAEnv.from_preset("adversarial", seed=7)


app = create_fastapi_app(
    env=build_env,
    action_cls=ARAAAction,
    observation_cls=ARAAObservation,
)


def main() -> None:
    uvicorn.run("serve_openenv:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
