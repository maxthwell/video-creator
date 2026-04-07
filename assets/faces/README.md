# Face Extraction Input

Drop source photos anywhere under `assets/faces/`.

Recommended layout:

- `assets/faces/<character_id>/<expression>/*.jpg`
- `assets/faces/<character_id>/<expression>/*.png`
- `assets/faces/raw/**/*.jpg`

Notes:

- One source image may contain multiple faces. The extraction script detects all of them.
- Each detected face is cropped, labeled, cached, and written into `assets/faces/.cache/`.
- By default, each source image is treated as one person and generates one character under `assets/characters/face-<source-name>/`.
- The script promotes the best matching face crops into that generated character's `skins/` directory and writes a `character.json` with estimated metadata for later filtering.
- Folder names such as `neutral`, `happy`, `smile`, `angry`, `skeptical`, `thinking`, `excited`, `sad`, `talk_neutral_open`, and `talk_angry_closed` are treated as requested expression slots.

Run:

```bash
python3 scripts/extract_face_expressions.py
```
