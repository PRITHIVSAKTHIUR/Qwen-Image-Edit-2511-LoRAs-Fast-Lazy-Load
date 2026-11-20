"""The NCII prompt guard, in its own process.

[`hfmlsoc/ncii-guard-v02`](https://huggingface.co/hfmlsoc/ncii-guard-v02) is a 270M CPU text
classifier scoring the NCII risk of an edit prompt. It cannot live in the main process: with it loaded there,
every subsequent `@spaces.GPU` worker dies at `worker_init` with `RuntimeError: No CUDA GPUs are available` —
the fork inherits whatever CUDA driver state the classifier's torch activity left behind, and a factory reboot
does not clear it. It cannot be a `multiprocessing.spawn` child either: spawn re-imports the parent's main
module, and on a Space that main module is `app.py` — the child would re-run the whole startup, `start()`
included. So the classifier runs this file as a plain subprocess — a fresh interpreter that never sees `spaces`
— and answers over stdin/stdout, one JSON object per line.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading

GUARD_REPO = "hfmlsoc/ncii-guard-v02" #@v01 - https://huggingface.co/hfmlsoc/ncii-light-guard-v01

_lock = threading.Lock()
_process: subprocess.Popen | None = None


def _read(timeout: float) -> dict:
    readable, _, _ = select.select([_process.stdout], [], [], timeout)
    if not readable:
        raise TimeoutError(f"the guard did not answer within {timeout}s")
    line = _process.stdout.readline()
    if not line:
        raise EOFError("the guard process died")
    return json.loads(line)


def _spawn() -> None:
    global _process
    _process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # Generous: a cold cache downloads the checkpoint first.
    assert _read(300.0) == {"status": "ready"}


def start() -> None:
    """Launch the worker and block until its model is up.

    Called once at startup; `classify` revives it if it dies.
    """
    with _lock:
        _spawn()


def classify(prompt: str, timeout: float = 60.0) -> dict:
    """`{'label': 'safe' | 'ncii', 'score': ...}` for one prompt.

    Replaces a dead or wedged worker once before giving up.
    """
    with _lock:
        for attempt in (0, 1):
            try:
                if _process is None or _process.poll() is not None:
                    _spawn()
                _process.stdin.write(json.dumps({"prompt": prompt}) + "\n")
                _process.stdin.flush()
                return _read(timeout)
            except Exception:
                if attempt:
                    raise
                if _process is not None and _process.poll() is None:
                    _process.kill()


def _serve() -> None:
    """The child: plain torch on CPU.

    The protocol keeps the real stdout to itself — everything else
    (download progress, warnings) is pushed over to stderr so it
    cannot corrupt a reply.
    """
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    from transformers import pipeline

    classifier = pipeline("text-classification", model=GUARD_REPO, device="cpu")
    protocol.write(json.dumps({"status": "ready"}) + "\n")
    for line in sys.stdin:
        result = classifier(json.loads(line)["prompt"], truncation=True)[0]
        protocol.write(json.dumps({"label": result["label"], "score": float(result["score"])}) + "\n")


if __name__ == "__main__":
    _serve()