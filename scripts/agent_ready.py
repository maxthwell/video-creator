#!/usr/bin/env python3
from __future__ import annotations

import os
import py_compile
import sys
from pathlib import Path

from common.io import ROOT_DIR, asset_catalog


MAIN_SCRIPT = Path("scripts/generate_cangyun_escort_story.py")
POSE_SCRIPT = Path("scripts/generate_actions_pose_reconstruction.py")
EXTRACT_SCRIPT = Path("scripts/extract_action_poses.py")
POSE_DIR = Path("assets/actions/.cache/poses")


def _compile(path: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return False, str(exc)
    return True, "ok"


def main() -> int:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    print(f"root={ROOT_DIR}")
    print(f"ready_script={Path(__file__).resolve()}")

    catalog = asset_catalog()
    print(
        "assets="
        + " ".join(
            [
                f"backgrounds:{len(catalog.get('backgrounds', []))}",
                f"characters:{len(catalog.get('characters', []))}",
                f"effects:{len(catalog.get('effects', []))}",
                f"bgm:{len(catalog.get('bgm', []))}",
                f"audio:{len(catalog.get('audio', []))}",
                f"motions:{len(catalog.get('motions', []))}",
            ]
        )
    )

    checks = []
    for rel_path in (MAIN_SCRIPT, POSE_SCRIPT, EXTRACT_SCRIPT):
        abs_path = ROOT_DIR / rel_path
        ok, detail = _compile(abs_path)
        checks.append((rel_path, ok, detail))

    pose_cache_count = len(list((ROOT_DIR / POSE_DIR).glob("*.pose.json"))) if (ROOT_DIR / POSE_DIR).exists() else 0
    print(f"pose_cache_dir={ROOT_DIR / POSE_DIR}")
    print(f"pose_cache_count={pose_cache_count}")

    failures = [item for item in checks if not item[1]]
    for rel_path, ok, detail in checks:
        print(f"check={rel_path} status={'ok' if ok else 'failed'} detail={detail}")

    if failures:
        print("status=not-ready")
        return 1

    print(f"mainline={MAIN_SCRIPT}")
    print("status=ready")
    print("next=python3 scripts/list_assets.py --pretty")
    print("next=python3 scripts/extract_action_poses.py")
    print("next=python3 scripts/generate_cangyun_escort_story.py --fast3 --force --output outputs/preview.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
