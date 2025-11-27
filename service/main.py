import shlex
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, UploadFile

from inference_cli import main as inference

api = FastAPI()


@api.post("/upscale")
async def send(
    args: str,
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
):
    temp_dir = Path(tempfile.gettempdir())
    file_suffix = Path(file.filename).suffix
    stored_name = f"seedvr_{uuid4().hex}{file_suffix}"
    stored_path = temp_dir / stored_name
    contents = await file.read()
    stored_path.write_bytes(contents)

    background_tasks.add_task(upscale_task, str(stored_path), args)
    return {"status": "queued", "file": stored_name}


def upscale_task(filename: str, args: str) -> None:
    if "--cache_dit" not in args:
        args += "--cache_dit"

    if "--cache_vae" not in args:
        args += "--cache_vae"

    extra_args = shlex.split(args) if args else []
    inference([filename, *extra_args])


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0")
