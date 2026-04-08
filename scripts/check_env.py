#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import shutil
import sys


def main() -> int:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    required_modules = (
        "PIL",
        "numpy",
        "cv2",
        "onnxruntime",
        "edge_tts",
    )
    optional_modules = (
        "panda3d",
        "direct",
    )
    checks: list[tuple[str, str]] = []

    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            checks.append((module_name, "ok"))
        except ModuleNotFoundError:
            checks.append((module_name, "missing"))

    for module_name in optional_modules:
        try:
            importlib.import_module(module_name)
            checks.append((module_name, "ok (optional single-chain renderer support)"))
        except ModuleNotFoundError:
            checks.append((module_name, "missing (optional single-chain renderer support)"))

    checks.append(("ffmpeg", "ok" if shutil.which("ffmpeg") else "missing"))

    for name, status in checks:
        print(f"{name}: {status}")

    missing = [name for name, status in checks if status == "missing"]
    if missing:
        print("environment check failed")
        return 1

    print("environment check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
