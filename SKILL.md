---
name: pygame-agent-video
description: >-
  Create or render local multi-scene videos with this skill. Use when the user
  wants a story video, action showcase, dialogue video, TTS video,
  BGM/effects/audio mixing, or needs an existing `scripts/generate_*.py` video
  script edited or executed. This skill is optimized for weak models: always
  start from `scripts/agent_ready.py` or `scripts/list_assets.py`, use only
  local asset ids, prefer editing an existing generator script over inventing a
  new format, and run the generator script directly to produce the mp4.
---

# Pygame Agent Video

Use this skill from the skill root. Do not hard-code `/mnt/data/...` paths. It must still work after being copied to `~/.myskill/...`.

## Default flow

1. Run the readiness check first:

```bash
python3 scripts/agent_ready.py
```

2. Inspect local assets:

```bash
python3 scripts/list_assets.py --pretty
```

3. Prefer reusing or editing an existing generator under `scripts/generate_*.py`.

4. Validate before rendering:

```bash
python3 scripts/check_story_input.py --input scripts/generate_my_story.py
```

5. Render by running the generator directly:

```bash
python3 scripts/generate_my_story.py --cpu --output outputs/story.mp4
```

## Hard rules

- Only use local assets returned by `scripts/list_assets.py`.
- Prefer Python story scripts. Do not invent a new input format.
- Prefer editing an existing `scripts/generate_*.py` file instead of building a brand-new pipeline.
- Keep commands relative to the skill root so the skill also works from `~/.myskill`.
- If rendering fails, run `python3 scripts/check_env.py` and `python3 scripts/agent_ready.py` before changing code.
- If the user only wants a new video quickly, duplicate the closest existing generator and edit that.

## Stable commands

Environment check:

```bash
python3 scripts/check_env.py
```

Asset discovery:

```bash
python3 scripts/list_assets.py --pretty
```

Validate a script:

```bash
python3 scripts/check_story_input.py --input scripts/generate_my_story.py
```

Full pipeline from a script:

```bash
python3 scripts/run_pipeline.py --input scripts/generate_my_story.py --cpu --output outputs/story.mp4
```

Direct render from a generator:

```bash
python3 scripts/generate_my_story.py --cpu --output outputs/story.mp4
```

## When to read references

- Read `references/trigger-and-workflow.md` if you need the exact operational order.
- Read `references/architecture.md` if you need to patch runtime behavior.
- Read `references/scene-schema.md` only when editing the normalized story shape.
