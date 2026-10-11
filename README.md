# PitchSight

**AI match analyst for eight sports: soccer, basketball, American football, ice hockey, tennis, volleyball,
baseball and cricket.** Pick a game from the library (or upload a clip and let PitchSight recognise the sport) and it
tracks every player and the ball, reads each side's shape, and writes a coaching report: a score out of 10 for every
player, what they did right and wrong, where they should have been, the move they should have made, and nitpicks.
Everything is shown on an annotated video plus a tactical board drawn on that sport's playing surface.

### Privacy and personal clips

There are no accounts and no sign-in. The `library/` folder holds the **featured** clips everyone sees.
A visitor who analyses their own clip can press **Save to my clips**, which stores the clip and its report in
*their own browser* (IndexedDB), so only they can see it and it is not on the server. Uploaded videos are deleted
from the server after 24 hours (`UPLOAD_TTL_HOURS`) and cached results after 7 days (`CACHE_TTL_DAYS`).
When the written AI report is enabled, a few still frames from the clip are sent to the Claude API to write it.
The "Add to shared library" button only appears, and only works, on the machine running the server.

### Sport detection

Leave the sport on **Auto-detect** and a small CLIP model (`openai/clip-vit-base-patch32`) compares a few frames with
a text description of each sport. If it is not confident, the page says so and lets you pick the sport yourself.
Choosing a sport by hand always overrides it.

### Detection and tracking

Each clip is probed on a few frames first and gets the detector that suits it:

1. **YOLOv8n** (fast, side-on or broadcast footage),
2. **YOLOv8m at 1280 px** if too few people are found (small or distant players),
3. **Motion tracking** for steady top-down drone footage, where people are a few-pixel blobs a photo-trained
   model cannot see. The empty-scene background is the median of many frames; whatever moves against it is a player.

The **Deep scan** option (on by default) uses a higher-resolution pass, a dedicated ball pass, camera-drift
compensation, track stitching, and a player-count rule per sport (a tennis clip keeps two players, not the crowd).
Deep scan analyses the first 60 s of a clip; turn it off for a quick look at up to 120 s.

Motion mode needs a steady camera. Players who stand perfectly still all clip merge into the background.

### Only the people on the field count

A detector finds *everyone*: spectators, photographers, coaches, ball boys, the bench. `fieldmask.py` keeps only the
people who are actually playing, with no hand-drawn field and nothing sport-specific:

1. The camera's pan and zoom is tracked (`camera.py`, optical flow + a similarity fit), so a phone on the sideline
   does not make everyone look like they are running.
2. People who really move are almost certainly players. The colour of the ground beside *their* feet is the playing
   surface (grass, wood, ice, clay).
3. The largest connected patch of that colour, with lines, logos and players bridged, is the playing area. Anyone
   clearly outside it (the stands, the track behind the touchline) is ignored.
4. The movers form a group. Staff and the bench stand outside it, and in continuous-motion sports someone who is
   in view for seconds and never moves is not playing. (Baseball and cricket are exempt: fielders stand still.)

The results page states how many people were ignored. The filter only removes people, and steps aside when there
is not enough movement to find the area (a frozen clip, a two-player tennis clip).

Tracking uses a small distance-based tracker (`boxtracker.py`) instead of the detector's built-in one, which drops
small, bunched-up players because it needs two frames of box overlap before reporting anyone.

Markers stay put: a detector misses a player for a frame or two, so short gaps in a player's track are filled in and
the boxes lightly smoothed, and the video overlay blends between analysed frames so rings and numbers glide at the
video's own frame rate instead of stepping and blinking.

### Teams, ratings and honesty

- Sides are split by kit colour (or by side of the net in tennis and volleyball). Every player carries a confidence,
  and a **Wrong team? Move to…** button lets the user correct a player; the shape is recomputed.
- Ratings are relative to the other players in the same clip (distance covered, bursts, staying connected to the
  group, time near the ball when the ball was actually seen). They are a rating of the movement visible in this clip,
  not of the whole match, and the report says when a player was barely seen.
- The ball is only used when it was found often enough to mean something; the page shows how often it was seen.

### Sports

Detection and tracking are sport-agnostic (YOLOv8 finds people and balls). What changes per sport lives in
`sports.py` and `vocab.py` as data: how many players to keep, how to split the sides, how to describe the shape, and
the wording of the report (a tennis report never talks about a back four).

| Sport | Sides split by | Shape reported as |
|---|---|---|
| Soccer | kit colour | formation lines, e.g. 4-3-3 |
| Basketball | kit colour | depth bands, e.g. 3-2 |
| American football | kit colour | line / second level / secondary, e.g. 5-3-3 |
| Ice hockey | kit colour | defence / middle / forwards |
| Tennis | side of the net | singles, one up one back, level |
| Volleyball | side of the net | front row / back row, e.g. 3-3 |
| Baseball | kit colour | fielders in the infield vs outfield, e.g. 5 infield / 4 outfield |
| Cricket | kit colour | fielders close in vs deep, e.g. 5 close / 6 deep |

Built for Hacklanta.

## How it works

```
clip ──► sport detection (CLIP) ──► YOLOv8 + ByteTrack / motion tracker (players, ball)
     ──► kit-colour clustering (teams) ──► depth clustering (shape)
     ──► movement ratings + coaching text (Claude vision when a key is set) ──► JSON report ──► web UI
```

| File | Role |
|---|---|
| `static/index.html` | Whole front end: library, upload, video overlay, tactical board, player cards, My clips |
| `app.py` | FastAPI server: upload, job progress, library endpoints, result cache |
| `pipeline.py` | Tracking, team split, shape estimate, ratings, Claude analysis (tracking-only fallback without a key) |
| `fieldmask.py` | Keeps only the people on the playing area (surface colour + the group of movers) |
| `boxtracker.py` | Distance-based multi-person tracker for small, bunched-up players |
| `camera.py` | Camera pan and zoom between frames |
| `sports.py` | Per-sport profiles: player counts, team-split rule, shape estimator |
| `vocab.py` | Per-sport wording for the report |
| `sportdetect.py` | Zero-shot sport recognition |
| `library.py` | Searchable game library + upload cache (stdlib only) |
| `precompute.py` | Analyse a clip once and add it to the library |

## Run it

```bash
git clone https://github.com/AppaStoppa/pitchsight.git && cd pitchsight
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # optional, enables the written AI report
uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

On Windows, put the key in a `.env` file (`ANTHROPIC_API_KEY=...`, already git-ignored) and start with
`uvicorn app:app --port 8000 --env-file .env`.

### Just want to explore the sample results? (lightweight, no GPU)

The four library clips are already analysed, so you can open them with a small install (no PyTorch, no model downloads):

```bash
git clone https://github.com/AppaStoppa/pitchsight.git && cd pitchsight
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-demo.txt
python -m uvicorn app:app --port 8000
# open http://localhost:8000/?game=volleyball-clip-1
```

Analysing a new clip needs the full `requirements.txt` above.

Or with Docker: `docker build -t pitchsight . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... pitchsight`

### Add games to the library (the "search a game" feature)

```bash
python precompute.py my_clip.mp4 --title "Park League Final – 2nd half" --tags youth,4-3-3,counter
python precompute.py rally.mp4 --title "Club doubles" --sport tennis   # or: auto, soccer, basketball, football, hockey, volleyball, baseball, cricket
```

Each clip is analysed once and stored in `library/`, so the demo opens it instantly with no GPU and no API key.
Only add footage you own or are licensed to use.

## Sample footage

`library/` holds four short sample clips (a volleyball rally, a tennis rally, a basketball court seen from above and
an indoor youth match seen from a drone) so the demo opens instantly with no GPU. They are there only to demonstrate
the analysis. If you are the rights holder of a clip and want it removed, open an issue. Use your own footage with
`precompute.py`, and only add footage you own or are licensed to use.

## For judges

- No login or credentials needed.
- **Fastest way to see it:** open the site and click any clip in the **Game library**. Results are precomputed, so no API key is needed.
- To test live analysis, upload a 10–60 s clip (Deep scan reads the first 60 s; with it off, up to 120 s. Change the caps with `DEEP_MAX_SECONDS` / `MAX_SECONDS`). Without `ANTHROPIC_API_KEY` the report is built from the tracking data alone and the header says "Tracking analysis"; with a key it says "AI coaching report".
- Click a player card or a dot on the tactical board to see their ideal position and suggested run.
- Direct links open a result at once: `/?game=volleyball-clip-1`, `/?game=tennis-1-on-1-clip`,
  `/?game=aerial-basketball-court`, `/?game=indoor-youth-match-drone-view`.
- Tests (no GPU, no video needed): `python tests/test_sports.py` and `python tests/test_playing_area.py`.

## Tech used

YOLOv8 (Ultralytics) + ByteTrack, OpenCV, CLIP (Hugging Face Transformers), scikit-learn (KMeans), NumPy,
FastAPI/Uvicorn, Anthropic Claude API (vision, `claude-sonnet-5-5`), vanilla JS/Canvas front end.

## Disclosure

All application code in this repo was written during Hacklanta. Third-party pieces: open-source libraries above,
pretrained YOLOv8 COCO weights and the CLIP model (downloaded at first run), and the Anthropic Claude API.
The project was built with AI coding assistance (Claude).

## Known limits

- Positions are in video-frame space, not true pitch coordinates (no homography/pitch-keypoint model yet).
- Teams are split by two-colour clustering; similar kits, referees and goalkeepers can be mislabelled
  (each player shows a confidence, and the team can be corrected by hand).
- Top-down drone clips only detect players who move; someone standing still is invisible to the motion tracker.
- Coaches and officials who walk along the touchline, or stand just inside it, can still be counted as players.
- Very small players in a wide shot (a whole football field from the stands) are the hardest case: some of a
  tightly bunched line of scrimmage is missed, and the results page says how many players were tracked.
- The ball is small and often hidden; ball-based remarks are only made when it was seen often enough.
- Shape is estimated from the players visible in the clip.
- Sport-specific wording and counts are tested end to end on soccer, basketball and tennis footage; the other
  sports use the same tracking with their own profile and vocabulary.
- The written AI report is a model's judgement from a few sampled frames plus tracking stats, not ground truth.
