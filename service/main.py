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
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    args: str = "--batch_size 5 --dit_offload_device cpu --vae_offload_device cpu --tensor_offload_device cpu --vae_encode_tiled --swap_io_components --compile_dit --compile_vae --blocks_to_swap 16 --debug",
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
        args += " --cache_dit"

    if "--cache_vae" not in args:
        args += " --cache_vae"

    extra_args = shlex.split(args) if args else []
    inference([filename, *extra_args])


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0")
