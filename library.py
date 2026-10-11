"""Game library + result cache. Stdlib only so it is easy to test.

library/library.json            index of clips (no results inside)
library/videos/<id>.mp4|.jpg    clip + thumbnail
library/results/<id>.json       precomputed analysis result
cache/<sha256>.json             results for user uploads (identical re-uploads are free)
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

BASE = Path(__file__).parent
LIB = BASE / "library"
VIDEOS = LIB / "videos"
RESULTS = LIB / "results"
INDEX = LIB / "library.json"
CACHE = BASE / "cache"

for d in (VIDEOS, RESULTS, CACHE):
    d.mkdir(parents=True, exist_ok=True)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "clip"


def _read_index():
    if not INDEX.exists():
        return []
    try:
        return json.loads(INDEX.read_text())
    except json.JSONDecodeError:
        return []


def entries(q="", sport=""):
    """List library entries, optionally filtered by sport and by a search string (all words must match)."""
    words = [w for w in re.split(r"\s+", q.lower().strip()) if w]
    out = []
    for e in _read_index():
        if sport and e.get("sport", "soccer") != sport:
            continue
        hay = " ".join([e.get("title", ""), e.get("sport", ""), e.get("description", ""),
                        " ".join(e.get("tags", []))]).lower()
        if all(w in hay for w in words):
            out.append(e)
    return out


def get(entry_id):
    for e in _read_index():
        if e["id"] == entry_id:
            res_path = RESULTS / f"{entry_id}.json"
            if not res_path.exists():
                return None
            return {**e, "result": json.loads(res_path.read_text())}
    return None


def save(title, video_path, result, sport="soccer", tags=None, description="", thumb_path=None):
    """Add (or replace) a clip in the library. Copies the video into library/videos."""
    entry_id = slug(title)
    video_path = Path(video_path)
    dest = VIDEOS / f"{entry_id}{video_path.suffix or '.mp4'}"
    if video_path.resolve() != dest.resolve():
        shutil.copyfile(video_path, dest)
    thumb = None
    if thumb_path and Path(thumb_path).exists():
        t = VIDEOS / f"{entry_id}.jpg"
        if Path(thumb_path).resolve() != t.resolve():
            shutil.copyfile(thumb_path, t)
        thumb = f"/library/videos/{t.name}"
    (RESULTS / f"{entry_id}.json").write_text(json.dumps(result))
    entry = {
        "id": entry_id, "title": title, "sport": sport, "tags": tags or [], "description": description,
        "video": f"/library/videos/{dest.name}", "thumb": thumb,
        "formations": result.get("formations", {}), "mode": result.get("mode"),
    }
    idx = [e for e in _read_index() if e["id"] != entry_id] + [entry]
    INDEX.write_text(json.dumps(idx, indent=2))
    return entry


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def cache_get(digest):
    p = CACHE / f"{digest}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cache_put(digest, result):
    (CACHE / f"{digest}.json").write_text(json.dumps(result))
