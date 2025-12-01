import shlex
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

from inference_cli import main as inference

# Use the system temporary directory for storing uploads and results
TEMP_DIR = Path(tempfile.gettempdir())

api = FastAPI()

# Expose the temporary directory via HTTP so generated results can be accessed
api.mount("/tmp", StaticFiles(directory=TEMP_DIR), name="tmp")


CMD = "--temporal_overlap 3 --dit_model seedvr2_ema_7b_fp16.safetensors --uniform_batch_size --vae_encode_tiled --vae_decode_tiled --batch_size 53 --dit_offload_device cpu --vae_offload_device cpu --tensor_offload_device cpu --swap_io_components --blocks_to_swap 36"


@api.post("/upscale")
async def send(
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    args: str = CMD,
):
    file_suffix = Path(file.filename).suffix
    stored_name = f"seedvr_{uuid4().hex}{file_suffix}"
    stored_path = TEMP_DIR / stored_name
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
