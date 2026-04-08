---
name: pygame-agent-video
description: >-
  Maintain and render exactly one local video pipeline: the cangyun mainline
  under scripts/generate_cangyun_escort_story.py, backed by DNN pose detection
  and the panda3/DNN stickman rendering chain. This skill is optimized for weak
  models: never invent a second pipeline, never use deleted legacy generators,
  always inspect local assets first, update pose caches when actions change, and
  render by running the cangyun generator directly.
---

# Single-Chain Video Skill

This skill supports one maintained pipeline only:

- Main story generator: `scripts/generate_cangyun_escort_story.py`
- DNN pose extractor: `scripts/extract_action_poses.py`
- Shared pose renderer/model code: `scripts/generate_actions_pose_reconstruction.py`
- Runtime helpers: `scripts/check_env.py`, `scripts/agent_ready.py`, `scripts/list_assets.py`

Everything else is legacy and must not be recreated.

## Non-Negotiable Rules

- Only use the cangyun mainline. Do not create or maintain a second story pipeline.
- Do not resurrect deleted `generate_*.py`, `storyboard`, `pygame_renderer`, or legacy panda renderer code.
- If the user wants a new story, implement it by editing `scripts/generate_cangyun_escort_story.py`.
- If `assets/actions` changed, run `scripts/extract_action_poses.py` before rendering.
- Only use local assets from this repo. Never reference remote assets or external URLs.
- Prefer `--fast3` for previews and iteration. Use normal mode only for final quality when requested.
- Keep commands relative to the skill root.

## Exact Weak-Model Workflow

Use this order. Do not skip steps unless the user explicitly says to.

1. Check runtime:

```bash
python3 scripts/check_env.py
python3 scripts/agent_ready.py
```

2. Inspect local assets:

```bash
python3 scripts/list_assets.py --pretty
```

3. If the user changed action GIF/WebP resources, refresh pose caches:

```bash
python3 scripts/extract_action_poses.py
```

4. Edit only the cangyun generator:

```bash
scripts/generate_cangyun_escort_story.py
```

Typical edits:

- rewrite `TITLE`
- rewrite `SCENES`
- adjust actors, lines, expressions, BGM, SFX, effects
- keep using `generate_actions_pose_reconstruction.py` as the shared DNN stickman backend

5. Render:

Preview:

```bash
python3 scripts/generate_cangyun_escort_story.py --fast3 --force --output outputs/preview.mp4
```

Higher quality:

```bash
python3 scripts/generate_cangyun_escort_story.py --force --output outputs/final.mp4
```

## What To Touch

- `scripts/generate_cangyun_escort_story.py`
- `scripts/generate_actions_pose_reconstruction.py`
- `scripts/extract_action_poses.py`
- `scripts/check_env.py`
- `scripts/agent_ready.py`
- `scripts/list_assets.py`
- `SKILL.md`

## What Not To Touch

- Do not add a new renderer stack.
- Do not add a new `generate_*story*.py` script unless the user explicitly asks to fork the mainline.
- Do not reintroduce `storyboard` workflows.
- Do not use `scripts/run_pipeline.py` or `scripts/render_video.py`.
- Do not route story generation through `common/panda_renderer.py` or `common/pygame_renderer.py`.

## Asset Selection Rules

- Characters: use `assets/characters`
- Actions: use `assets/actions` and `assets/actions/.cache/poses`
- Backgrounds: use `assets/backgrounds`
- Effects: use `assets/effects`
- SFX: use `assets/audio`
- BGM: use `assets/bgm`

Always prefer actual filenames or ids returned by `scripts/list_assets.py`.

## Fast Decision Rules For Weak Models

- User asks for a new story video:
  edit `scripts/generate_cangyun_escort_story.py`

- User says actions changed:
  run `python3 scripts/extract_action_poses.py`

- User says expressions/pose/head/body rendering is wrong:
  patch `scripts/generate_actions_pose_reconstruction.py`

- User says assets are missing or command fails:
  run `python3 scripts/check_env.py` and `python3 scripts/agent_ready.py`

- User wants output now:
  render with `--fast3` first

## Minimal Command Set

```bash
python3 scripts/check_env.py
python3 scripts/agent_ready.py
python3 scripts/list_assets.py --pretty
python3 scripts/extract_action_poses.py
python3 scripts/generate_cangyun_escort_story.py --fast3 --force --output outputs/preview.mp4
python3 scripts/generate_cangyun_escort_story.py --force --output outputs/final.mp4
```
