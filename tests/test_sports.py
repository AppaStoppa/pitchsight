"""Offline checks: shape estimation, team split, wording per sport. Run: python tests/test_sports.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pipeline, sports

rng = np.random.default_rng(0)
ok = True
def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(("PASS " if good else "FAIL ") + f"{name}: got {got!r}" + ("" if good else f", wanted {want!r}"))

def jitter(pts, s=0.01):
    return [(x + rng.normal(0, s), y + rng.normal(0, s)) for x, y in pts]

P = sports.SPORTS
# soccer 4-4-2 (x = depth), keeper at 0.05
soccer = [(0.05, .5)] + [(0.2, y) for y in (.15, .38, .62, .85)] + [(0.45, y) for y in (.15, .38, .62, .85)] + [(0.7, .4), (0.7, .6)]
check("soccer 4-4-2", pipeline.estimate_formation(jitter(soccer), P["soccer"]), "4-4-2")

# basketball half-court: 3 up top, 2 low
bball = [(0.3, .2), (0.3, .5), (0.3, .8), (0.75, .3), (0.75, .7)]
check("basketball 2 bands", pipeline.estimate_formation(jitter(bball), P["basketball"]), "3-2")

# volleyball: 3 front, 3 back
vb = [(0.4, .2), (0.4, .5), (0.4, .8), (0.1, .2), (0.1, .5), (0.1, .8)]
check("volleyball 3-3", pipeline.estimate_formation(jitter(vb), P["volleyball"]), "3-3")

# hockey: 2 D, 1 centre, 2 wings up front, 1 goalie => 3 bands
hockey = [(0.05, .5), (0.2, .3), (0.2, .7), (0.5, .5), (0.7, .25), (0.7, .75)]
bands = pipeline.estimate_formation(jitter(hockey), P["hockey"])
check("hockey sums to 6", sum(int(b) for b in bands.split("-")), 6)

# american football: 5 linemen, 3 LB, 3 DBs
fb = [(0.4, y) for y in (.2, .35, .5, .65, .8)] + [(0.55, y) for y in (.3, .5, .7)] + [(0.8, y) for y in (.2, .5, .8)]
check("football 5-3-3", pipeline.estimate_formation(jitter(fb), P["football"]), "5-3-3")

# cricket: 5 close, 6 deep ring around the centre
ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
cricket = [(0.5 + 0.08 * np.cos(a), 0.5 + 0.08 * np.sin(a)) for a in ang[:5]] + \
          [(0.5 + 0.4 * np.cos(a), 0.5 + 0.4 * np.sin(a)) for a in ang]
check("cricket ring", pipeline.estimate_formation(jitter(cricket, .005), P["cricket"]), "5 close / 6 deep")

# baseball: 4 infielders near the centre, 3 outfielders deep, plus pitcher/catcher
base = [(0.5, 0.62), (0.5, 0.45)] + [(0.38, 0.5), (0.62, 0.5), (0.45, 0.4), (0.55, 0.4)] + \
       [(0.15, 0.2), (0.5, 0.08), (0.85, 0.2)]
bb = pipeline.estimate_formation(jitter(base, .004), P["baseball"])
check("baseball uses infield/outfield words", "infield" in bb and "outfield" in bb, True)
check("baseball counts add up", sum(int(s.split()[0]) for s in bb.split(" / ")), 9)
check("baseball prompt mentions Baseball", "Baseball" in pipeline.build_system_prompt(P["baseball"]), True)

# tennis: doubles pair, one up one back (y is depth)
check("tennis one up one back", pipeline.estimate_formation([(.3, .30), (.7, .45)], P["tennis"]), "One up, one back")
check("tennis level", pipeline.estimate_formation([(.3, .30), (.7, .32)], P["tennis"]), "Level")
check("tennis singles", pipeline.estimate_formation([(.5, .30)], P["tennis"]), "Singles")

# team split by side: singles tennis, both players wearing white
def track(x, y, col=(240, 128, 128), n=10):
    return {"colors": [col] * n, "pts": [(i * .2, x, y) for i in range(n)]}
tracks = {7: track(.5, .25), 31: track(.5, .78)}   # far player, near player; identical colours
teams, cols, _ = pipeline.assign_teams(tracks, P["tennis"])
check("tennis sides differ", teams[7] != teams[31], True)
check("tennis far player is A", teams[7], "A")

# volleyball split by side with x axis
tracks = {i: track(.2, .3 + .1 * i) for i in range(3)} | {10 + i: track(.8, .3 + .1 * i) for i in range(3)}
teams, _, _ = pipeline.assign_teams(tracks, P["volleyball"])
check("volleyball left side = A", {teams[i] for i in range(3)}, {"A"})
check("volleyball right side = B", {teams[10 + i] for i in range(3)}, {"B"})

# colour split still works (basketball: red vs blue in LAB-ish space)
red, blue = (136, 208, 195), (60, 140, 90)
tracks = {i: track(.3, .1 * i, red) for i in range(4)} | {10 + i: track(.6, .1 * i, blue) for i in range(4)}
teams, cols, tinfo = pipeline.assign_teams(tracks, P["basketball"])
check("colour split separates", len({teams[i] for i in range(4)}) == 1 and teams[0] != teams[10], True)

check("clear colour split has high quality", tinfo["quality"] >= 0.7, True)

# identical kits: the system must say it is NOT sure rather than pretend
same = {i: track(.3, .1 * i, (130, 128, 128)) for i in range(8)}
for i in same:                                   # tiny random wobble so KMeans has something to split
    same[i]["colors"] = [tuple(np.array((130, 128, 128)) + rng.normal(0, 1.5, 3)) for _ in range(10)]
_, _, same_info = pipeline.assign_teams(same, P["basketball"])
check("identical kits give low quality", same_info["quality"] < 0.45, True)

# stitching: one player whose ID changes mid-clip must become one track; two real players must not merge
def seg(x, y, t0, n=8):
    return {"colors": [(130, 128, 128)] * n, "colors2": [], "crops": [],
            "pts": [(t0 + k * 0.1, x + k * 0.002, y) for k in range(n)]}
d = {"tracks": {1: seg(.30, .50, 0.0), 2: seg(.33, .50, 1.0),       # same person: ends at .8s, restarts 0.2s later nearby
                3: seg(.80, .20, 0.0), 4: seg(.80, .22, 0.0)},       # two different people, both present at once
     "frames": [{"t": 0.0, "p": [[1, .3, .4, .32, .5], [3, .8, .1, .82, .2], [4, .8, .12, .82, .22]]},
                {"t": 1.0, "p": [[2, .33, .4, .35, .5]]}], "keys": []}
merged = pipeline.stitch_tracks(d)
check("stitch merges the split player", sorted(d["tracks"]) == [1, 3, 4] and merged == 1, True)
check("stitch relabels frames", d["frames"][1]["p"][0][0], 1)

# ball: a lone outlier is dropped and a short gap is filled
bf = [{"t": k * .1, "p": [], "ball": [0.2 + 0.02 * k, 0.5]} for k in range(12)]
bf[5]["ball"] = [0.9, 0.1]            # glitch
bf[8]["ball"] = None; bf[9]["ball"] = None   # gap of two
res = pipeline.finalize_ball({"frames": bf, "step_fps": 10})
check("ball glitch removed and gap filled", bf[5]["ball"] is not None and abs(bf[5]["ball"][0] - 0.3) < 0.05 and bf[8]["ball"] is not None, True)
check("ball coverage reports seen < filled", res["seen"] < res["filled"], True)

# ball: a cone detected all clip long must NOT be reported as the ball, and the moving ball must be followed
N = 80
fr = []
for i in range(N):
    cands = [[0.56, 0.385, 0.627]]                          # the cone: same place every frame, fairly confident
    if i % 4 != 3:                                          # the real ball: visible in 75% of frames, moving
        cands.append([0.30, 0.2 + 0.008 * i, 0.5 + 0.1 * np.sin(i / 9)])
    if i % 10 == 0:
        cands.append([0.25, 0.9, 0.1])                      # an occasional random false positive far away
    fr.append({"t": i * 0.1, "p": [], "ball_cands": cands})
res2 = pipeline.finalize_ball({"frames": fr, "step_fps": 10})
on_cone = sum(1 for f in fr if f.get("ball") and abs(f["ball"][0] - 0.385) < 0.02 and abs(f["ball"][1] - 0.627) < 0.02)
check("cone is not reported as the ball", on_cone, 0)
check("static detections were dropped", res2["static_dropped"] >= N, True)
follow = [f for f in fr if f.get("ball") and not f.get("ball_est")]
good = sum(1 for k, f in enumerate(fr) if f.get("ball") and abs(f["ball"][0] - (0.2 + 0.008 * k)) < 0.03)
check("moving ball is followed", good >= 0.7 * N, True)
check("ball coverage is honest (about 75%, not 100%)", 0.6 <= res2["seen"] <= 0.85, True)

# a clip with ONLY the cone must report no ball at all
fr2 = [{"t": i * 0.1, "p": [], "ball_cands": [[0.5, 0.385, 0.627]]} for i in range(60)]
res3 = pipeline.finalize_ball({"frames": fr2, "step_fps": 10})
check("cone-only clip reports no ball", (res3["seen"], res3["filled"]), (0.0, 0.0))

# ---- player selection: the failures reported on real clips
def world(spec, n=100, keys_at=(10, 50)):
    """spec: {tid: (first_frame, last_frame, x_fn(i), y_fn(i))} -> data dict like track_video returns."""
    frames, tracks = [], {}
    for i in range(n):
        p = []
        for tid, (a, b, xf, yf) in spec.items():
            if a <= i <= b:
                x, y = xf(i), yf(i)
                p.append([tid, x - 0.01, y - 0.12, x + 0.01, y])
                tr = tracks.setdefault(tid, {"colors": [], "colors2": [], "crops": [], "pts": []})
                tr["pts"].append((i * 0.1, x, y)); tr["colors"].append((130, 128, 128))
        frames.append({"t": i * 0.1, "p": p, "ball": None})
    keys = [(frames[i]["t"], None, frames[i]) for i in keys_at]        # key frames share the frame's record, as in the app
    return {"frames": frames, "tracks": tracks, "keys": keys}

osc = lambda c, a, per: (lambda i: c + a * math.sin(i / per))
still = lambda c: (lambda i: c)
import math
# tennis singles: a far player, a near player, and the near player's track re-issued under a new id after a gap
d = world({1: (0, 99, osc(.5, .08, 7), still(.28)), 2: (0, 65, osc(.45, .15, 5), osc(.85, .03, 9)),
           11: (80, 99, osc(.43, .10, 5), osc(.84, .03, 9))})
sel = pipeline.select_players(d, P["tennis"])
check("tennis singles keeps exactly 2 players", sel["kept"], 2)
ids_early = {p[0] for p in d["frames"][10]["p"]}; ids_late = {p[0] for p in d["frames"][90]["p"]}
check("re-issued near-player track joins the same player", len(ids_early | ids_late), 2)

# tennis doubles: four real players plus a ball kid who barely moves
dd = world({1: (0, 99, osc(.3, .05, 6), osc(.30, .02, 5)), 2: (0, 99, osc(.7, .05, 7), osc(.30, .02, 6)),
            3: (0, 99, osc(.3, .08, 5), osc(.80, .03, 5)), 4: (0, 99, osc(.7, .08, 6), osc(.82, .03, 7)),
            9: (0, 99, still(.05), still(.5))})
check("tennis doubles keeps 4 and drops the ball kid", pipeline.select_players(dd, P["tennis"])["kept"], 4)

# basketball: 10 players, 3 spectators at the edges, and a second game far away
spec = {i: (0, 99, osc(.3 + .04 * i, .06, 5 + i % 4), osc(.35 + .03 * (i % 5), .05, 6 + i % 3)) for i in range(1, 11)}
spec.update({20: (0, 99, still(.03), still(.5)), 21: (0, 99, still(.97), still(.45)), 22: (0, 99, still(.5), still(.02))})
spec.update({30 + k: (0, 99, osc(.9, .04, 5 + k), osc(.1, .04, 6 + k)) for k in range(4)})   # far corner game
db = world(spec)
sel2 = pipeline.select_players(db, P["basketball"])
check("basketball keeps 10 and ignores spectators and the other game", sel2["kept"], 10)
check("no spectator or far-game box survives", all(p[0] <= 10 for f in db["frames"] for p in f["p"]), True)

# key frames must be remapped exactly once (they are the same objects as the frames)
kf = db["keys"][0][2]
check("key frame ids match the frame ids", [p[0] for p in kf["p"]] == [p[0] for p in db["frames"][10]["p"]], True)

# ---- every sport speaks its own language: no pitch talk in a tennis report, no puck in basketball, and so on
def report_text(sport_key, n_players):
    """Run the real report builder on synthetic movement for one sport and return every sentence it produced."""
    prof = P[sport_key]
    rnd = np.random.default_rng(7)
    spec = {}
    for i in range(1, n_players + 1):
        side = i % 2
        spec[i] = (0, 99, osc(.25 + .5 * side + .03 * (i % 5), .06 + .01 * (i % 4), 5 + i % 5),
                   osc(.3 + .4 * ((i // 2) % 3) / 2 + .02 * i, .05 + .02 * (i % 3), 6 + i % 4))
    w = world(spec, n=100)
    for k, f in enumerate(w["frames"]):                                   # a ball that crosses the picture with one fast burst
        f["ball"] = [0.1 + 0.8 * (k / 99) ** 1.5, 0.5 + 0.1 * math.sin(k / 6)] if k % 2 == 0 else None
    stats = []
    for tid, tr in w["tracks"].items():
        arr = np.array(tr["pts"])
        stats.append({"track_id": tid, "team": "A" if tid % 2 else "B", "team_conf": 1.0,
                      "avg_x": float(arr[:, 1].mean()), "avg_y": float(arr[:, 2].mean()),
                      "start": [float(arr[0, 1]), float(arr[0, 2])], "end": [float(arr[-1, 1]), float(arr[-1, 2])]})
    rep = pipeline.movement_report(w["frames"], stats, {"A": "FORM-A", "B": "FORM-B"}, prof)
    assert rep is not None, sport_key
    out = [rep["summary"]]
    for p in rep["players"]:
        out += [p["role"], p["ideal_position"]["why"], p["suggested_run"]["description"]] + p["doing_right"] + p["doing_wrong"] + p["nitpicks"]
    for t in rep["teams"].values():
        out += [t["style"]] + t["strengths"] + t["weaknesses"]
    for mo in rep["key_moments"]:
        out += [mo["title"], mo["description"]]
    return out, rep

BANNED = {   # words that would prove the report is talking about the wrong sport
    "soccer":     ["racket", "puck", "skater", "fielder", "batter", "dribble", "crossover", "rally"],
    "basketball": ["pitch", "racket", "puck", "skater", "fielder", "batter", "rally", "set piece", "offside", "wicket"],
    "football":   ["pitch", "racket", "puck", "skater", "fielder", "dribble", "rally", "wicket", "set piece", "offside"],
    "hockey":     ["pitch", "racket", "rally", "serve", "fielder", "batter", "dribble", "crossover", "court", "set piece"],
    "tennis":     ["pitch", "teammate", "pass", "press", "clearance", "set piece", "puck", "skater", "fielder", "batter",
                   "dribble", "floor", "diamond", "snap", "shape", "spectator", "the team"],
    "volleyball": ["pitch", "puck", "skater", "fielder", "batter", "dribble", "racket", "ice", "set piece", "offside"],
    "baseball":   ["puck", "skater", "racket", "rally", "court", "dribble", "set piece", "offside", "crossover"],
    "cricket":    ["puck", "skater", "racket", "rally", "court", "dribble", "set piece", "offside", "crossover", "pitch"],
}
COUNTS = {"soccer": 12, "basketball": 10, "football": 12, "hockey": 10, "tennis": 2, "volleyball": 12, "baseball": 9, "cricket": 11}
seen_titles = {}
for key in P:
    text, rep = report_text(key, COUNTS[key])
    low = " | ".join(text).lower()
    hits = [w for w in BANNED[key] if w in low]
    check(f"{key}: no wrong-sport words ({hits or 'none'})", hits, [])
    check(f"{key}: has key moments and a role on every card", len(rep["key_moments"]) >= 3 and all(p["role"] for p in rep["players"]), True)
    check(f"{key}: no empty strength/fault/nitpick lists", all(p["doing_right"] and p["doing_wrong"] and p["nitpicks"] for p in rep["players"]), True)
    seen_titles[key] = {m["title"] for m in rep["key_moments"]}
check("tennis uses court language in its roles", all("court" in p["role"].lower() for p in report_text("tennis", 2)[1]["players"]), True)
check("different sports produce different moment wording", len({frozenset(v) for v in seen_titles.values()}) >= 6, True)

# prompt carries the sport
check("prompt mentions label", "Ice hockey" in pipeline.build_system_prompt(P["hockey"]), True)
check("prompt has no template markers", "<<" in pipeline.build_system_prompt(P["tennis"]), False)

# unknown sport is rejected
try:
    sports.get("curling"); check("unknown sport rejected", False, True)
except ValueError:
    check("unknown sport rejected", True, True)

print("\nALL PASS" if ok else "\nSOME FAILED")
