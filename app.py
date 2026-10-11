import os
import shutil
import threading
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import library
import pipeline
import sportdetect
import sports

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

UPLOAD_TTL_HOURS = int(os.environ.get("UPLOAD_TTL_HOURS", 24))      # visitors' videos are deleted after this
CACHE_TTL_DAYS = int(os.environ.get("CACHE_TTL_DAYS", 7))


def purge_old_files():
    """Delete uploaded videos and cached results past their time limit. Library clips are never touched.
    Visitors keep what they care about in their own browser (My clips), so the server holds nothing long-term."""
    import time
    now = time.time()
    for folder, ttl in ((UPLOADS, UPLOAD_TTL_HOURS * 3600), (library.CACHE, CACHE_TTL_DAYS * 86400)):
        for p in folder.iterdir():
            try:
                if p.is_file() and now - p.stat().st_mtime > ttl:
                    p.unlink()
            except OSError:
                pass


purge_old_files()
app = FastAPI(title="PitchSight")
# Allow the page to call the API even when it is opened from another origin
# (file://, VS Code Live Server on :5500, a separate front-end host). No cookies are used.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RequestLog:
    """Print a line the moment an API POST starts arriving. The normal access log only prints once a request
    has finished, so an upload that gets cut off halfway would otherwise leave no trace at all."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") == "POST" and scope["path"].startswith("/api/"):
            h = {k.decode(): v.decode("latin-1") for k, v in scope["headers"]}
            who = scope["client"][0] if scope.get("client") else "?"
            print(f"[request] POST {scope['path']} from {who} origin={h.get('origin', '-')} "
                  f"size={h.get('content-length', '?')} bytes", flush=True)
        await self.app(scope, receive, send)


app.add_middleware(RequestLog)
jobs: dict = {}


def to_browser_mp4(src: Path, dst: Path, seconds: int = None) -> bool:
    """Re-encode an upload to H.264 MP4 so every browser can play it. Phone clips (HEVC .mov) and
    .avi/.mkv files otherwise analyse fine but show a black player. The video is also capped to the
    length we analyse and to 1280 px wide, so the overlay and the footage always match."""
    try:
        import subprocess

        import imageio_ffmpeg
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
               "-t", str(seconds or pipeline.MAX_SECONDS), "-vf", "scale='min(1280,iw)':-2",
               "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-an", "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        return dst.exists() and dst.stat().st_size > 0
    except Exception as e:
        print("Video conversion failed, using the original file:", e)
        return False


def run_job(job_id: str, raw: Path, mp4: Path, digest: str, sport: str, deep: bool = True):
    def progress(p, m):
        jobs[job_id].update(progress=int(p), message=m)

    try:
        progress(1, "Preparing video")
        # deep scan analyses a shorter window, so the stored video is cut to the same length
        seconds = min(pipeline.MAX_SECONDS, pipeline.DEEP_MAX_SECONDS) if deep else pipeline.MAX_SECONDS
        if to_browser_mp4(raw, mp4, seconds):
            raw.unlink(missing_ok=True)
            work = mp4
        else:                                   # keep going with the original, it just may not play
            work = raw
            jobs[job_id]["video"] = f"/uploads/{raw.name}"
        cached = library.cache_get(digest)
        if cached:                              # identical clip and sport analysed before: instant
            jobs[job_id]["result"] = cached
            jobs[job_id].update(status="done", progress=100, message="Loaded from cache")
            return
        confidence = None
        if sport == "auto":
            progress(3, "Detecting the sport")
            sport, confidence = sportdetect.resolve_sport(str(work), "auto")
        result = pipeline.analyse(str(work), progress, sport, deep)
        if confidence is not None:   # tell the page this was detected, and how sure we were
            result["detected"] = {"confidence": round(confidence, 2),
                                  "low": confidence < sportdetect.LOW_CONFIDENCE}
        library.cache_put(digest, result)
        jobs[job_id]["result"] = result
        jobs[job_id].update(status="done", progress=100, message="Done")
    except Exception as e:
        jobs[job_id].update(status="error", message=str(e))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # the page carries its own inline icon; answer browsers' automatic request quietly instead of with a 404
    return Response(status_code=204)


@app.get("/api/health")
def health():
    # Lets the page confirm the server is up and report any missing libraries.
    import importlib.util
    import os

    needed = ["ultralytics", "cv2", "numpy", "sklearn", "anthropic"]
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    return {"ok": not missing, "missing": missing, "ai_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "auto_detect": importlib.util.find_spec("transformers") is not None}


@app.get("/api/sports")
def list_sports():
    return {k: {"label": v["label"], "unit": v["unit"], "surface": v["surface"]} for k, v in sports.SPORTS.items()}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), sport: str = Form("auto"), deep: str = Form("1")):
    if sport != "auto":   # "auto" means: work it out from the video
        try:
            sport, _ = sports.get(sport)
        except ValueError as e:
            raise HTTPException(400, str(e))
    purge_old_files()   # cheap, and keeps a long-running server from filling its disk
    job_id = uuid.uuid4().hex[:10]
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    raw = UPLOADS / f"{job_id}_raw{suffix}"
    with open(raw, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if raw.stat().st_size == 0:
        raw.unlink(missing_ok=True)
        raise HTTPException(400, "That file is empty. Choose a video file and try again.")
    # the same clip analysed as a different sport is a different result, so the sport is part of the key
    is_deep = deep not in ("0", "false", "no", "")
    digest = f"{library.file_hash(raw)}-{sport}-{'deep' if is_deep else 'quick'}-v{pipeline.VERSION}"
    mp4 = UPLOADS / f"{job_id}.mp4"
    jobs[job_id] = {"status": "running", "progress": 1, "message": "Queued", "video": f"/uploads/{mp4.name}"}
    threading.Thread(target=run_job, args=(job_id, raw, mp4, digest, sport, is_deep), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return JSONResponse(j)


@app.get("/api/library")
def list_library(q: str = "", sport: str = ""):
    return library.entries(q, sport)


class ShapeRequest(BaseModel):
    sport: str = "soccer"
    A: list = []          # [[x, y], ...] average positions of side A
    B: list = []


@app.post("/api/shape")
def recompute_shape(req: ShapeRequest):
    """Re-describe each side's shape after the user moves a player between sides in the page."""
    try:
        _, profile = sports.get(req.sport)
    except ValueError as e:
        raise HTTPException(400, str(e))
    def shape(pts):
        pts = [(float(p[0]), float(p[1])) for p in pts[:40] if len(p) >= 2]
        return pipeline.estimate_formation(pts, profile) if pts else "unclear"
    return {"A": shape(req.A), "B": shape(req.B)}


class SaveRequest(BaseModel):
    job_id: str
    title: str
    tags: str = ""
    description: str = ""


@app.post("/api/library/save")
def save_to_library(req: SaveRequest, request: Request):
    """Add an analysed upload to the game library. Only allowed from the machine running the server
    (or when ALLOW_LIBRARY_SAVE=1), so visitors to a public demo cannot add clips."""
    local = request.client and request.client.host in ("127.0.0.1", "::1", "localhost")
    if not local and os.environ.get("ALLOW_LIBRARY_SAVE") != "1":
        raise HTTPException(403, "Saving to the library is only allowed from the computer running the server.")
    j = jobs.get(req.job_id)
    if not j or j.get("status") != "done" or not j.get("result"):
        raise HTTPException(404, "That analysis is no longer available. Upload the clip again, then save it.")
    title = req.title.strip()[:80]
    if not title:
        raise HTTPException(400, "Give the clip a title first.")
    video = UPLOADS / Path(j["video"]).name
    if not video.exists():
        raise HTTPException(404, "The uploaded video file is missing. Upload the clip again, then save it.")
    existing = {e["id"] for e in library.entries()}
    base, n = title, 2
    while library.slug(title) in existing:        # never silently overwrite another clip
        title, n = f"{base} {n}", n + 1
    thumb = None
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1) // 3)
    ok, frame = cap.read()
    cap.release()
    if ok:
        h, w = frame.shape[:2]
        thumb = str(library.CACHE / "_thumb.jpg")
        cv2.imwrite(thumb, cv2.resize(frame, (480, int(480 * h / w))))
    result = j["result"]
    entry = library.save(title, video, result, sport=result.get("sport", "soccer"),
                         tags=[t.strip() for t in req.tags.split(",") if t.strip()],
                         description=req.description.strip(), thumb_path=thumb)
    return {"id": entry["id"], "title": entry["title"]}


@app.get("/api/library/{entry_id}")
def get_library_entry(entry_id: str):
    e = library.get(entry_id)
    if not e:
        raise HTTPException(404, "unknown clip")
    return e


app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
app.mount("/library/videos", StaticFiles(directory=library.VIDEOS), name="library-videos")


@app.get("/")
def index():
    # no-cache: browsers re-check the page every visit, so an updated site is never shown stale
    return FileResponse(BASE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
