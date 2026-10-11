"""Offline checks for the tracker and the playing-area filter, on synthetic scenes.
Run: python tests/test_playing_area.py"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import boxtracker
import fieldmask
import sports

ok = True


def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(("PASS " if good else "FAIL ") + f"{name}: got {got!r}" + ("" if good else f", wanted {want!r}"))


# ---------------------------------------------------------------- tracker
tr = boxtracker.BoxTracker(fps=12)
ids_seen = set()
for f in range(10):                                   # a small runner moving 14 px a frame (no box overlap)
    x = 100 + 14 * f
    ids = tr.update([(x, 200, x + 20, 250, 0.5), (400, 300, 430, 360, 0.6)])
    ids_seen.add(ids[0])
check("fast small runner keeps one id", len(ids_seen), 1)
check("two people get two ids", len(set(tr.update([(240, 200, 260, 250, 0.5), (400, 300, 430, 360, 0.6)]))), 2)
tr2 = boxtracker.BoxTracker(fps=12)
a = tr2.update([(100, 100, 130, 170, 0.5)])[0]
for _ in range(3):
    tr2.update([])                                    # lost for 3 frames
b = tr2.update([(104, 102, 134, 172, 0.5)])[0]
check("a short loss keeps the same id", a, b)
check("a weak detection never starts a track", tr2.update([(600, 50, 620, 90, 0.15)])[0], None)

# ---------------------------------------------------------------- playing area
W, H, N = 1280, 720, 40
rng = np.random.default_rng(1)


def scene(with_staff=True, with_crowd=True):
    """Grass in the lower 60%, stands above. 8 players run around the middle; 6 staff stand still low down."""
    noise = np.random.default_rng(7)
    bgr = np.zeros((90, 160, 3), np.uint8)
    bgr[:] = (70, 60, 160)                            # stands: reddish, and busy like a crowd
    bgr[:36] = np.clip(bgr[:36].astype(int) + noise.integers(-60, 60, (36, 160, 3)), 0, 255).astype(np.uint8)
    bgr[36:] = (60, 140, 60)                          # grass: smooth
    bgr[36:] = np.clip(bgr[36:].astype(int) + noise.integers(-2, 3, (54, 160, 3)), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    frames, thumbs = [], []
    for i in range(N):
        p = []
        for k in range(8):                            # players: foot y ~ 0.55-0.65, running along x
            x = 0.15 + 0.08 * k + 0.05 * np.sin(i / 5 + k)
            y = 0.58 + 0.04 * np.cos(i / 7 + k)
            p.append([k + 1, x - 0.012, y - 0.09, x + 0.012, y])
        if with_staff:
            for k in range(6):                        # staff: still, on grass but far from the play
                x = 0.1 + 0.15 * k
                p.append([100 + k, x - 0.02, 0.84 - 0.2, x + 0.02, 0.97])
        if with_crowd:
            for k in range(6):                        # crowd: on the stands, still
                x = 0.1 + 0.15 * k
                p.append([200 + k, x - 0.01, 0.12, x + 0.01, 0.2])
        frames.append({"t": i / 12, "p": p, "cam": [0.0, 0.0]})
        thumbs.append(lab)
    tracks = {}
    for rec in frames:
        for q in rec["p"]:
            tracks.setdefault(q[0], {"pts": [], "colors": []})["pts"].append((rec["t"], (q[1] + q[3]) / 2, q[4]))
    return {"W": W, "H": H, "frames": frames, "tracks": tracks, "thumbs": thumbs}


prof = sports.get("football")[1]
d = scene()
res = fieldmask.apply(d, prof)
kept = sorted(d["tracks"])
check("crowd and staff are dropped, players stay", kept, list(range(1, 9)))
check("filter reports it applied", res["applied"], True)
check("it counts the people it ignored", res["ignored_people"], 12)

d = scene(with_staff=False, with_crowd=False)
res = fieldmask.apply(d, prof)
check("nothing is removed from a clean scene", sorted(d["tracks"]), list(range(1, 9)))

still = scene()                                       # nobody moves at all: no evidence, so no filtering
first = [list(q) for q in still["frames"][0]["p"]]
for rec in still["frames"]:
    rec["p"] = [list(q) for q in first]
res = fieldmask.apply(still, prof)
check("a frozen clip is left alone", len(still["tracks"]), 20)

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
