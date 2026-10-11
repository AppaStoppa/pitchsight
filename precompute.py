"""Analyse a clip once and add it to the searchable game library.

    python precompute.py clip.mp4 --title "Park League Final – 2nd half" --tags youth,4-3-3,counter

Only add footage you own or are licensed to use (your own recordings, CC0/CC-BY clips, etc.).
Results are stored, so the live demo shows them instantly with no GPU and no API key.
"""
import argparse
import sys

import cv2

import library
import pipeline
import sportdetect
import sports


def thumb_of(video, out):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 3)
    ok, frame = cap.read()
    cap.release()
    if ok:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (480, int(480 * h / w)))
        cv2.imwrite(out, frame)
        return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--title", required=True)
    ap.add_argument("--sport", default="auto", choices=["auto", *sports.SPORTS],
                    help="which sport the clip shows (changes team split, shape and coaching prompt); "
                         "'auto' detects it from the video")
    ap.add_argument("--tags", default="", help="comma separated")
    ap.add_argument("--description", default="")
    a = ap.parse_args()

    def progress(p, m):
        print(f"[{int(p):3d}%] {m}", file=sys.stderr)

    a.sport, confidence = sportdetect.resolve_sport(a.video, a.sport)
    if confidence is not None:
        print(f"Detected sport: {a.sport} (confidence {confidence:.0%})", file=sys.stderr)
    result = pipeline.analyse(a.video, progress, a.sport)
    thumb = thumb_of(a.video, str(library.CACHE / "_thumb.jpg"))
    entry = library.save(a.title, a.video, result, sport=a.sport,
                         tags=[t.strip() for t in a.tags.split(",") if t.strip()],
                         description=a.description, thumb_path=thumb)
    print("Added to library:", entry["id"], "| mode:", result["mode"])


if __name__ == "__main__":
    main()
