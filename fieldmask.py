"""Who is actually *in the game*? Everyone a detector finds is a "person": the crowd, the photographers, the
coaches and ball boys, the bench. This module keeps only the people on the playing area.

How, with no hand-drawn field and nothing sport-specific:

1. People who really move (several body-lengths, camera drift removed) are almost certainly players, and they
   stand on the playing surface. The colour just under *their* feet is the surface (grass, wood, ice, clay...).
2. A few pictures from the clip are checked against that colour. Someone whose feet are not on the surface (the
   stands, the advertising boards, the track behind the far touchline) is not in the game.
3. Sideline staff DO stand on the grass, so colour is not enough. The moving players form a tight group; staff,
   ball boys and the bench stand well outside it. Anyone far from that group is ignored, unless they move like
   a player themselves.

Everything is conservative: when there is not enough evidence (a still clip, a clip with no movers) the filters
step aside and the old behaviour applies, so this can only remove people, never invent them.
"""
import cv2
import numpy as np

THUMB_W = 320            # a handful of small Lab pictures are kept while tracking: enough to see grass vs stands
MAX_THUMBS = 24
MOVER_EXTENT = 2.2       # body-lengths a track must travel (camera removed) to count as "moving like a player"
MIN_PRESENCE = 0.06


def thumb(frame):
    """A small Lab copy of a frame, kept while tracking so the surface can be checked afterwards."""
    h = max(8, int(round(THUMB_W * frame.shape[0] / frame.shape[1])))
    small = cv2.resize(frame, (THUMB_W, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2LAB)


def keep_every(expected_frames):
    """Keep a thumbnail every N analysed frames so memory stays small on long clips."""
    return max(1, int(expected_frames) // MAX_THUMBS)


def thumbs_from_video(path, frames):
    """Rebuild the thumbnails for already-tracked frames (used by tests and the precompute tool)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    every = keep_every(len(frames))
    out = []
    for i, rec in enumerate(frames):
        img = None
        if i % every == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(rec["t"] * fps)))
            ok, img = cap.read()
            img = img if ok else None
        out.append(thumb(img) if img is not None else None)
    cap.release()
    return out


def frame_matrix(rec):
    """3x3 (normalised coords) taking this frame to the first frame: pan + zoom if known, else a plain shift."""
    m = rec.get("M")
    if m and len(m) == 6:
        return np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0.0, 0.0, 1.0]])
    cx, cy = rec.get("cam", (0.0, 0.0))
    return np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])


def _observations(data):
    """track id -> [(frame_index, stab_x_px, stab_y_px, height_px, x1, x2, y2 (normalised, raw), height_norm)]
    Stabilised = where the feet are in the FIRST frame's picture (pan and zoom removed); the height is scaled
    the same way so distances can be counted in body-lengths."""
    W, H = data["W"], data["H"]
    obs = {}
    for i, rec in enumerate(data["frames"]):
        Mi = frame_matrix(rec)
        s = float(np.sqrt(abs(np.linalg.det(Mi[:2, :2]))))
        for p in rec["p"]:
            tid, x1, y1, x2, y2 = p[:5]
            q = Mi @ np.array([(x1 + x2) / 2, y2, 1.0])
            obs.setdefault(tid, []).append((i, float(q[0]) * W, float(q[1]) * H, (y2 - y1) * H * s,
                                            x1, x2, y2, y2 - y1))
    return obs


def _track_stats(obs, n_frames, times=None):
    """Per-track presence, height, how far it travels (ext, in body-lengths, robust to a swapped id or two) and
    how fast it moves at its quickest (spd, body-lengths per second over half-second windows)."""
    stats = {}
    for tid, r in obs.items():
        if len(r) < 4:
            continue
        pts = np.array([[a[1], a[2]] for a in r])
        h = float(np.median([a[3] for a in r])) or 1.0
        lo, hi = np.percentile(pts, 10, axis=0), np.percentile(pts, 90, axis=0)
        spd = 0.0
        if times is not None:
            t = np.array([times[a[0]] for a in r])
            speeds = []
            j = 0
            for i in range(len(r)):                       # pair each sighting with one about half a second later
                while j < len(r) - 1 and t[j] - t[i] < 0.5:
                    j += 1
                if t[j] - t[i] >= 0.45:
                    speeds.append(float(np.hypot(*(pts[j] - pts[i]))) / h / (t[j] - t[i]))
            if speeds:
                spd = float(np.percentile(speeds, 90))
        stats[tid] = {
            "n": len(r), "pres": len(r) / max(1, n_frames), "h": h,
            "ext": float(np.hypot(*(hi - lo))) / h * 1.25, "spd": spd,
            "x": float(np.median(pts[:, 0])), "y": float(np.median(pts[:, 1])),
        }
    return stats


def track_stats(data):
    """Per-track presence, body height and how far (in body-lengths, camera drift removed) each one travels."""
    return _track_stats(_observations(data), len(data["frames"]), [r["t"] for r in data["frames"]])


class _Pictures:
    """The few kept thumbnails, with a lookup that carries a position from one frame to another through the
    camera's pan and zoom."""

    def __init__(self, thumbs, frames):
        self.idx = [i for i, t in enumerate(thumbs) if t is not None]
        self.thumbs = thumbs
        self.mats = [frame_matrix(r) for r in frames]
        self.inv = {}
        self.masks = {}

    def nearest(self, i):
        return min(self.idx, key=lambda j: abs(j - i)) if self.idx else None

    def at(self, a):
        """(thumbnail index j, the observation's left foot point, right foot point, both in picture j)."""
        i = a[0]
        j = self.nearest(i)
        if j is None:
            return None, None, None
        if j not in self.inv:
            self.inv[j] = np.linalg.inv(self.mats[j])
        T = self.inv[j] @ self.mats[i]
        l = T @ np.array([a[4], a[6], 1.0])
        r = T @ np.array([a[5], a[6], 1.0])
        return j, l, r


def _ground_samples(pics, obs, seeds, max_obs=10):
    """Lab pixels of the ground right beside the movers' feet (same rows as the feet, left and right of the
    body): that is the playing surface, minus the players' own legs and shadows."""
    out = []
    for tid in seeds:
        r = obs[tid]
        for a in r[:: max(1, len(r) // max_obs)][:max_obs]:
            j, l, rr = pics.at(a)
            if j is None:
                continue
            t = pics.thumbs[j]
            th, tw = t.shape[:2]
            ty = int(round(l[1] * th))
            r0, r1 = max(0, ty - 2), min(th, ty + 1)
            if r1 <= r0:
                continue
            px = max(3, int(round(0.15 * a[7] * th)))
            ax, bx = int(round(l[0] * tw)), int(round(rr[0] * tw))
            for lo, hi in ((ax - px, ax - 1), (bx + 2, bx + 1 + px)):
                lo, hi = max(0, lo), min(tw, hi)
                if hi > lo:
                    out.append(t[r0:r1, lo:hi].reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.uint8)


def _surface_model(pics, obs, seeds):
    """The playing surface as up to two colour modes: [(a, b, radius, L_low, L_high)], or None when it cannot be
    learned. Hue (a, b) decides, brightness is only a loose range, so grass in shade is still grass."""
    px = _ground_samples(pics, obs, seeds).astype(np.float32)
    if len(px) < 60:
        return None
    ab = px[:, 1:3]
    model, remaining = [], np.ones(len(px), dtype=bool)
    total = len(px)
    for _ in range(2):
        cand = ab[remaining]
        if len(cand) < 0.12 * total:
            break
        H, xe, ye = np.histogram2d(cand[:, 0], cand[:, 1], bins=(64, 64), range=((64, 192), (64, 192)))
        # smooth a little so a mode is a hill, not one lucky bin
        Hs = cv2.GaussianBlur(H.astype(np.float32), (5, 5), 1.0)
        bx, by = np.unravel_index(int(np.argmax(Hs)), Hs.shape)
        centre = np.array([xe[bx] + 1.0, ye[by] + 1.0])
        near = np.linalg.norm(cand - centre, axis=1) <= 8.0
        if near.sum() < 0.10 * total:
            break
        mu = np.median(cand[near], axis=0)
        d = np.linalg.norm(ab - mu, axis=1)
        member = (d <= 8.0) & remaining
        Ls = px[member, 0]
        remaining &= ~(d <= 16.0)
        lo, hi = float(np.percentile(Ls, 5)), float(np.percentile(Ls, 95))
        neutral = float(np.hypot(mu[0] - 128.0, mu[1] - 128.0)) < 8.0
        # A real surface is one coherent colour. A grey mode spread over the whole brightness range is
        # white lines, shirts and shadows mixed together, and would match the stands: not a surface.
        # (Grey surfaces such as ice are bright and make up most of the samples.)
        if hi - lo > 95.0 or (neutral and (member.sum() < 0.55 * total or lo < 120.0)):
            continue
        spread = float(np.median(np.abs(d[member] - np.median(d[member])))) * 1.4826
        model.append({"mu": mu, "r": float(np.clip(7.0 + 2.0 * spread, 7.0, 13.0)),
                      "L": (lo - 16.0, hi + 16.0)})
    return model or None


def _surface_mask(t, model):
    """Pixels that are the playing-surface colour."""
    tf = t.astype(np.float32)
    ok = np.zeros(t.shape[:2], dtype=bool)
    for m in model:
        d = np.hypot(tf[..., 1] - m["mu"][0], tf[..., 2] - m["mu"][1])
        ok |= (d <= m["r"]) & (tf[..., 0] >= m["L"][0]) & (tf[..., 0] <= m["L"][1])
    return ok


def _field_region(mask):
    """The playing area as one filled blob. Thin things (painted lines, a centre logo, players) are bridged by a
    closing, the largest connected piece is kept and every hole inside it is filled. Returns (blob, distance
    in pixels from the blob), or None when there is no big piece of surface at all."""
    th, tw = mask.shape
    k = max(5, int(round(0.06 * tw)))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[big, cv2.CC_STAT_AREA] < 0.04 * th * tw:
        return None
    region = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    dist = cv2.distanceTransform((1 - filled).astype(np.uint8), cv2.DIST_L2, 3)
    return filled.astype(bool), dist


def _distance_outside(region, x, y, hn):
    """How far (in body-heights) a person's feet are outside the playing area; 0 when they are inside it."""
    blob, dist = region
    th, tw = blob.shape
    px, py = int(np.clip(round(x * tw), 0, tw - 1)), int(np.clip(round(y * th), 0, th - 1))
    return float(dist[py, px]) / max(1.0, hn * th)


def _rejoin_in_place(data, max_gap_s=1.2):
    """Detections flicker, so someone who stands still is often tracked as a string of short pieces. Pieces that
    reappear in (almost) the same spot, at the same size and in the same kit, are the same person: join them.
    This cannot hop between people (the allowed distance is under one body-length), and it gives the tests
    below the long tracks they need: a staff member who stands still for ten seconds is one track, not six."""
    frames, tracks = data["frames"], data["tracks"]
    obs = _observations(data)
    info = {}
    for tid, r in obs.items():
        if tid not in tracks or len(r) < 2:
            continue
        a, z = r[0], r[-1]
        cols = tracks[tid].get("colors") or []
        info[tid] = {"f0": a[0], "f1": z[0], "p0": (a[1], a[2]), "p1": (z[1], z[2]),
                     "h": float(np.median([q[3] for q in r])),
                     "c": np.mean(cols, axis=0) if cols else None}
    order = sorted(info, key=lambda t: info[t]["f0"])
    parent = {t: t for t in info}
    taken = set()

    def root(t):
        while parent[t] != t:
            t = parent[t]
        return t

    for b in order:
        best, best_cost = None, 1e9
        ib = info[b]
        for a in order:
            ia = info[a]
            if a == b or a in taken or ia["f1"] >= ib["f0"]:
                continue
            gap = frames[ib["f0"]]["t"] - frames[ia["f1"]]["t"]
            if gap > max_gap_s:
                continue
            href = max(ia["h"], ib["h"])
            d = float(np.hypot(ia["p1"][0] - ib["p0"][0], ia["p1"][1] - ib["p0"][1]))
            if d > 0.5 * href + 1.0 * href * gap:
                continue
            if max(ia["h"], ib["h"]) / max(1.0, min(ia["h"], ib["h"])) > 1.35:
                continue
            if ia["c"] is not None and ib["c"] is not None and float(np.linalg.norm(ia["c"] - ib["c"])) > 28:
                continue
            cost = d / href + gap
            if cost < best_cost:
                best, best_cost = a, cost
        if best is not None and root(best) != root(b):
            parent[b] = root(best)
            taken.add(best)
    remap = {t: root(t) for t in info}
    merged = sum(1 for t in info if remap[t] != t)
    if not merged:
        return 0
    for t in order:
        r = remap[t]
        if r == t:
            continue
        dst, src = tracks[r], tracks.pop(t)
        for k in ("colors", "colors2", "crops"):
            dst[k] = dst.get(k, []) + src.get(k, [])
        dst["pts"] = sorted(dst["pts"] + src["pts"], key=lambda q: q[0])
    for rec in frames:
        seen, out = set(), []
        for p in rec["p"]:
            nid = remap.get(p[0], p[0])
            if nid in seen:
                continue
            seen.add(nid)
            out.append([nid, *p[1:]])
        rec["p"] = out
    return merged


def apply(data, profile):
    """Remove tracks that are not on the playing area from data["tracks"] (frames are cleaned later by
    select_players). Returns a dict describing what was done, for the report and for debugging."""
    out = {"applied": False, "ignored_tracks": 0, "ignored_people": 0, "off_surface": 0, "sideline": 0,
           "movers": 0, "reason": ""}
    thumbs, frames, tracks = data.get("thumbs"), data["frames"], data["tracks"]
    if not thumbs or not frames or not tracks or not any(t is not None for t in thumbs):
        out["reason"] = "no pictures kept"
        return out
    n = len(frames)
    out["rejoined"] = _rejoin_in_place(data)
    tracks = data["tracks"]
    obs = _observations(data)
    stats = _track_stats(obs, n, [r["t"] for r in frames])
    if len(stats) < 4:
        out["reason"] = "too few tracks"
        return out
    pics = _Pictures(thumbs, frames)

    # ---- who clearly moves like a player?
    def movers_at(threshold):
        return [t for t, s in stats.items() if s["ext"] >= threshold and s["pres"] >= MIN_PRESENCE and s["n"] >= 5]

    movers = movers_at(MOVER_EXTENT)
    if len(movers) < 4:
        movers = movers_at(1.4)
    out["movers"] = len(movers)
    if len(movers) < 3:
        out["reason"] = "not enough movement to find the playing area"
        return out

    # ---- the surface is the colour beside the movers' feet; the playing area is its largest filled blob
    # People whose feet are clearly outside that blob (the stands, the advertising boards, the track behind the
    # far touchline) are not in the game, however much they move.
    model = _surface_model(pics, obs, movers)
    off, away = set(), {}
    if model:
        for tid in stats:
            vals = []
            r = obs[tid]
            for a in r[:: max(1, len(r) // 12)]:
                j, l, rr = pics.at(a)
                if j is None:
                    continue
                if j not in pics.masks:
                    pics.masks[j] = _field_region(_surface_mask(pics.thumbs[j], model))
                reg = pics.masks[j]
                if reg is None:
                    continue
                vals.append(_distance_outside(reg, (l[0] + rr[0]) / 2, l[1], a[7]))
            away[tid] = float(np.median(vals)) if vals else 0.0
        # sanity: the movers must be on the blob, or the learned colour was the wrong thing
        good = [away[t] for t in movers if t in away]
        share = float(np.mean([g <= 0.35 for g in good])) if good else 0.0
        out["mover_on_surface"] = share
        out["model_tried"] = model
        if share >= 0.5:
            off = {t for t in stats if away.get(t, 0.0) > 0.5}
        else:
            model = None
            out["reason"] = "surface colour was not consistent"

    # ---- the group the movers form; staff and the bench stand outside it
    side = set()
    live = [t for t in movers if t not in off]
    max_players = profile.get("max_players", 22)
    if len(live) >= 4 and max_players > 4:
        # Fast movers define where the action is; people who merely sway and wander (bench, staff) count little.
        w = np.array([stats[t]["pres"] * min(stats[t]["ext"], 6.0) * float(np.clip(stats[t]["spd"] / 2.0, 0.15, 1.0))
                      for t in live])
        xs = np.array([stats[t]["x"] for t in live])
        ys = np.array([stats[t]["y"] for t in live])
        hs = np.array([stats[t]["h"] for t in live])

        def wmedian(v):
            o = np.argsort(v)
            cw = np.cumsum(w[o])
            return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])

        cx, cy = wmedian(xs), wmedian(ys)
        sx = 1.4826 * wmedian(np.abs(xs - cx))
        sy = 1.4826 * wmedian(np.abs(ys - cy))
        h_ref = wmedian(hs)
        rx = max(3.0 * sx, 6.0 * h_ref)             # the group may be wide (a whole pitch) but not infinite
        ry = max(3.0 * sy, 3.2 * h_ref)
        for tid, s in stats.items():
            if tid in off:
                continue
            outside = abs(s["x"] - cx) > rx or abs(s["y"] - cy) > ry
            giant = s["h"] > 2.8 * h_ref and tid not in movers   # a non-mover standing at the lens
            if outside or giant:
                side.add(tid)
        out["region"] = {"cx": round(cx, 1), "cy": round(cy, 1), "rx": round(rx, 1), "ry": round(ry, 1)}

    # ---- in a sport where players keep moving, someone visible for seconds who never moves is not playing
    static = set()
    if not profile.get("static_ok", False) and len(live) >= 4 and max_players > 4:
        tvals = [r["t"] for r in frames]
        for tid, s in stats.items():
            if tid in off or tid in side or tid in movers:
                continue
            seen = [a[0] for a in obs[tid]]
            dur = tvals[max(seen)] - tvals[min(seen)]
            if dur >= 2.0 and s["ext"] < min(0.8, 0.3 * dur):   # a spectator or official: effectively planted
                static.add(tid)

    drop = off | side | static
    # safety net: never throw away most of the clearest movers; if that would happen the filter was wrong
    if movers and len([t for t in movers if t in drop]) > 0.5 * len(movers):
        out["reason"] = "filter disagreed with the moving players, left everything in"
        return out

    # ---- how many *people* is that (not how many track fragments)?
    per_frame = {}
    for tid in drop:
        for a in obs.get(tid, []):
            per_frame[a[0]] = per_frame.get(a[0], 0) + 1
    ignored_people = int(round(float(np.median([per_frame.get(i, 0) for i in range(n)])))) if drop else 0

    for tid in drop:
        tracks.pop(tid, None)
    out.update(applied=bool(model) or bool(side) or bool(static), ignored_tracks=len(drop),
               ignored_people=ignored_people, off_surface=len(off), sideline=len(side), static=len(static),
               model=model, away=away,
               why={t: ("off-surface" if t in off else "outside-group" if t in side else "never-moves")
                    for t in drop},
               stats=stats, movers_list=list(movers))
    return out
