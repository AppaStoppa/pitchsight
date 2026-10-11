"""Work out which sport a clip shows, using CLIP zero-shot classification.

CLIP is a model trained on image/text pairs, so it can score "how well does this sentence describe
this picture?" without any sport-specific training. We sample a few frames, score each sentence
below, average over the frames, and take the best one. Runs locally and needs no API key.

The model (~600 MB) downloads once, on first use. If it is unavailable the caller falls back to a
default sport, so a missing model never breaks an upload.
"""
import cv2

PROMPTS = {
    "soccer": "a photo of a soccer match on a grass pitch",
    "basketball": "a photo of a basketball game on an indoor court",
    "football": "a photo of an American football game",
    "hockey": "a photo of an ice hockey game on an ice rink",
    "tennis": "a photo of a tennis match on a tennis court",
    "volleyball": "a photo of a volleyball match with a net",
    "baseball": "a photo of a baseball game on a baseball diamond",
    "cricket": "a photo of a cricket match on a cricket field",
}
MODEL_NAME = "openai/clip-vit-base-patch32"
LOW_CONFIDENCE = 0.45          # below this the UI tells the user to double-check

_model = None


def _load():
    global _model
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _model = (CLIPModel.from_pretrained(MODEL_NAME).eval(), CLIPProcessor.from_pretrained(MODEL_NAME))
    return _model


def _sample_frames(path, n):
    from PIL import Image
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    imgs = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / n))
        ok, frame = cap.read()
        if ok:
            imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return imgs


def detect_sport(path, n_frames=5):
    """Return (sport_key, confidence 0-1, {sport: probability})."""
    import torch
    model, proc = _load()
    imgs = _sample_frames(path, n_frames)
    if not imgs:
        raise RuntimeError("could not read frames from the video")
    inputs = proc(text=list(PROMPTS.values()), images=imgs, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image          # one row per frame, one column per sentence
    probs = logits.softmax(dim=-1).mean(dim=0).tolist()    # average the frames
    by_sport = dict(zip(PROMPTS, probs))
    best = max(by_sport, key=by_sport.get)
    return best, float(by_sport[best]), by_sport


def resolve_sport(path, sport):
    """Turn 'auto' into a real sport key. Returns (sport, confidence or None).
    confidence is None when the sport was chosen by the user rather than detected."""
    if sport != "auto":
        return sport, None
    try:
        best, conf, _ = detect_sport(path)
        return best, conf
    except Exception as e:     # no model / no network: keep the demo alive with a sensible default
        print("Sport auto-detect failed, defaulting to soccer:", e)
        return "soccer", 0.0
