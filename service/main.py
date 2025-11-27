import atexit
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, UploadFile

api = FastAPI()


@api.post("/upscale")
async def send(
    args: str,
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
):
    """
    Accept an uploaded file and arguments for the CLI, persist the file
    to a temporary location, and enqueue an asynchronous upscale task.
    """
    # Persist the uploaded file to disk in the system temp directory so it is
    # available to the background task. We generate a unique name to avoid
    # collisions.
    temp_dir = Path(tempfile.gettempdir())
    file_suffix = Path(file.filename).suffix
    stored_name = f"seedvr_{uuid4().hex}{file_suffix}"
    stored_path = temp_dir / stored_name

    contents = await file.read()
    stored_path.write_bytes(contents)

    # Schedule the upscale task with the stored file path.
    background_tasks.add_task(upscale_task, str(stored_path), args)
    return {"status": "queued", "file": stored_name}


_active_processes: set[subprocess.Popen] = set()
_active_lock = threading.Lock()


def _terminate_all_children() -> None:
    """
    Best-effort termination of any child processes that are still running.

    This is invoked on FastAPI shutdown and at interpreter exit so that
    upscale jobs do not outlive the API process.
    """
    with _active_lock:
        processes = list(_active_processes)
        _active_processes.clear()

    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.terminate()
        except Exception:
            # Ignore failures; process may already be gone or in a bad state.
            continue

    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


@api.on_event("shutdown")
async def _on_shutdown() -> None:
    """FastAPI shutdown hook to clean up any running upscale subprocesses."""
    _terminate_all_children()


atexit.register(_terminate_all_children)


def upscale_task(filename: str, args: str) -> None:
    """
    Run the inference CLI as a child process, streaming all output to this
    process's stdout/stderr so it shows up in the service logs.
    """
    # Use shlex.split to respect quoting in args.
    extra_args = shlex.split(args) if args else []

    cmd = [
        sys.executable,
        "-u",  # unbuffered stdio for timely log streaming
        "inference_cli.py",
        filename,
        *extra_args,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with _active_lock:
        _active_processes.add(process)

    try:
        assert process.stdout is not None  # for type checkers

        for line in process.stdout:
            # Forward child output to our stdout so it is visible in logs.
            print(line, end="", flush=True)

        returncode = process.wait()

        if returncode != 0:
            print(
                f"[upscale_task] Child process exited with non-zero status {returncode}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        with _active_lock:
            _active_processes.discard(process)

    return


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0")
