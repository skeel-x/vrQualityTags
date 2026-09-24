"""Tag VR scenes with everything stash-vr needs to play them right.

One measurement pass per scene writes all of it: projection, lens, stereo
layout, eye order, the corner-packed alpha matte of passthrough scenes, and the
quality tier. The tag vocabulary is exactly stash-vr's default video rules, so
every tag written here has an effect in the headset.

WHERE THE ANSWER COMES FROM, IN ORDER OF AUTHORITY

  filename   explicit markers (_LR_, _TB_, _RL_, MKX200, RF52, ...). Most files
             carry none, but when present they are deliberate and beat any
             measurement.

  watermark  SLR burns "SLR 190/200/220 FOV" into the top of the left eye.
             Nothing measurable separates 190 from 200 from 220, so where the
             text exists it is the only authority. Two agreeing frames are
             required; a single OCR hit is not trusted. Not read where it
             cannot help: see fov_skip_reason().

  pixels     two frames decoded to a 256px thumbnail:
               lr / tb  correlation between the frame halves; a layout is only
                        accepted if it implies a plausible eye shape.
               bbox     shape of the lit region of one eye: a fisheye is a disc
                        (as wide as high), a 180 equirect a barrel (wider).
               corners  outside the inscribed circle of each eye a fisheye is
                        black. Passthrough scenes pack their alpha matte there
                        as solid saturated red shapes; see corner_matte().
               lower    a near-binary lower half is a packed matte, not a
                        second eye (guard against RGB-over-alpha read as TB).

The quality tier is measured from the file itself: width decides it, and in
the 6K band the bitrate has to agree too, because that is where upscales hide.

Outside the VR path a file is measured only when Stash's metadata alone says it
is VR (a 2:1 or square frame at least 3840 wide, or a VR marker in its name).
Flat (non-VR) stereoscopic 3D files there are recognised from their names alone
(3D, SBS, Half-SBS, LRF, HOU, ...) and get FLAT + SBS or FLAT + TB. Every other
file there is left untouched.

Pure standard library on purpose: the plugin interpreter has no numpy or PIL,
and arithmetic over a 256px thumbnail does not need them.
"""
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

DEFAULTS = {
    "pathFilter": "/VR/",
    # quality
    "parentTag": "HQ",
    "tag8k": "8K",
    "tag7k": "7K",
    "tag6kHbr": "6K HBR",
    "min8kWidth": 7680,
    "min7kWidth": 7000,
    "min6kWidth": 5760,
    "min6kBitrateMbit": 40,
    # projection
    "readFovWatermark": True,
    "overwrite": False,
    "minWidth": 1920,
    "ffmpegPath": "/usr/bin/ffmpeg",
    "tesseractPath": "/usr/bin/tesseract",
    # VR-shaped files outside the VR path, decided from metadata only
    "measureVrShapedOutside": True,
    # flat 3D outside the VR path, from filenames only
    "flat3dFilenameScan": True,
    # optional Stash API key; the session Stash hands a task expires after an
    # hour, which ends long runs part way
    "apiKey": "",
}

NUMERIC = ("min8kWidth", "min7kWidth", "min6kWidth", "min6kBitrateMbit", "minWidth")
BOOLEAN = ("readFovWatermark", "overwrite", "measureVrShapedOutside", "flat3dFilenameScan")

# ---------------------------------------------------------------- vocabulary
# Mirrors stash-vr's default video rules (internal/config/settings.go).
DOME, SPHERE, FISHEYE, FLAT = "DOME", "SPHERE", "FISHEYE", "FLAT"
RF52, MKX200, MKX220, VRCA220 = "RF52", "MKX200", "MKX220", "VRCA220"
SBS, TB, MONO, RL = "SBS", "TB", "MONO", "RL"
ALPHA = "Alpha"
UNRESOLVED = "VRP: Unresolved"      # probed but not classifiable; stops endless re-probing
SKIP = "VRP: Skip"                  # user-applied opt-out; never written or removed here

SCREEN_TAGS = (DOME, SPHERE, FISHEYE, FLAT)
LENS_TAGS = (RF52, MKX200, MKX220, VRCA220)
STEREO_TAGS = (SBS, TB, MONO)
PROJECTION_TAGS = SCREEN_TAGS + LENS_TAGS + STEREO_TAGS + (RL, ALPHA, UNRESOLVED)

# A scene carrying any of these has been classified before (by this plugin or
# by hand) and is not re-measured unless asked to.
SETTLED_TAGS = SCREEN_TAGS + LENS_TAGS + ("CUBEMAP", "EAC", UNRESOLVED)

FOV_LENS = {"190": RF52, "200": MKX200, "220": MKX220}

ANALYSIS_W = 256
ALPHA_BIMODAL = 0.75    # lower half this binary is a matte, not a second eye
FISH_BBOX_LO, FISH_BBOX_HI, FISH_BLKOUT = 0.94, 1.06, 0.72

# Corner matte (see README "How detection was calibrated").
MATTE_RED_MIN = 96          # R of a matte pixel after downscaling
MATTE_RED_RATIO = 0.30      # G and B at most this fraction of R
MATTE_BLACK_MAX = 32        # max(R,G,B) below this counts as black
MATTE_MIN_SHARE = 0.006     # red share of the corner pixels
MATTE_MIN_CLEAN = 0.80      # black + red share: the corners hold nothing else


def log(level, msg):
    # Stash reads plugin logs off stderr, one prefixed line at a time
    print(f"\x01{level}\x02{msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ filename

_ALNUM_SPLIT = re.compile(r"[^0-9a-z]+")
_SEG_180 = re.compile(r"^180(?![\d ])")
_SEG_360 = re.compile(r"^360(?![\d ])")
_180X180 = re.compile(r"(?<!\d)180x180(?!\d)")
_AR_PHRASE = re.compile(r"pass[\s_-]?through|alpha[\s_-]?packed|packed[\s_-]?alpha")

_SEG_STEREO = {"lr": SBS, "sbs": SBS, "tb": TB, "ou": TB, "mono": MONO, "2d": MONO}
# Flat (non-VR) stereoscopic 3D. The strong words only ever describe flat 3D,
# so they count everywhere; the loose ones ("3D", "SBS", "OU") are only trusted
# outside the VR path, where they cannot mean a VR layout.
_FLAT3D_STRONG = {"hsbs": SBS, "fsbs": SBS, "lrf": SBS, "hou": TB, "tab": TB, "tbf": TB}
_FLAT3D_PAIR = re.compile(r"(?<![a-z0-9])(?:half|full)[\s_.-]?(sbs|ou|tb)(?![a-z0-9])")
_FLAT3D_LOOSE = {"sbs": SBS, "ou": TB}
_VR_WORDS = {"vr", "vr180", "vr360", "180", "360", "180x180", "fisheye", "3dh", "3dv"}
# words that make a file outside the VR path worth measuring; a bare "180" or
# "360" is too common in titles, the _180 / _360 segments count via "screen"
_VR_NAME_WORDS = {"vr", "vr180", "vr360", "180x180", "fisheye", "fisheye180", "3dh", "3dv"}
_WORD_LENS = {"fisheye190": RF52, "rf52": RF52, "fisheye200": MKX200, "mkx200": MKX200,
              "mkx220": MKX220, "vrca220": VRCA220, "fisheye220": MKX220}
# lens names written with a separator before the number: "MKX-220", "mkx 200",
# "Fisheye_190", "RF 52"
_LENS_SPLIT = re.compile(r"(?<![a-z0-9])(mkx|vrca|fisheye|rf)[\s_.-]+(\d{2,3})(?!\d)")
# DeoVR's naming convention: 3dh = side by side, 3dv = over-under
_WORD_STEREO = {"3dh": SBS, "3dv": TB}


def _one(values):
    """The single value of a set, or None when it is empty or contradicts itself."""
    return next(iter(values)) if len(values) == 1 else None


def parse_filename(path):
    """Explicit markers in a file name.

    Layout markers (_LR_, _TB_, _RL_, _MONO_, _180, _360 ...) only count as a
    whole underscore-delimited segment: "Mono Lake" in a title is not a marker.
    Lens names are distinctive enough to count as any word. Contradicting
    markers cancel out, so the pixels decide instead.
    """
    base = os.path.basename(path or "").lower()
    stem = os.path.splitext(base)[0]
    segments = stem.split("_")
    words = set(_ALNUM_SPLIT.split(base))

    stereo, screen = set(), set()
    rl = False
    # a name without underscores has no delimited segments at all; the first
    # and last segment border an underscore on one side, which still counts
    # ("LR_180" or "..._LR")
    for seg in segments if len(segments) > 1 else ():
        if seg in _SEG_STEREO:
            stereo.add(_SEG_STEREO[seg])
        if seg == "rl":
            rl = True
            stereo.add(SBS)
        if _SEG_180.match(seg):
            screen.add(DOME)
        if _SEG_360.match(seg):
            screen.add(SPHERE)
    if _180X180.search(base):
        screen.add(DOME)

    for w, value in _WORD_STEREO.items():
        if w in words:
            stereo.add(value)
    if "fisheye" in words or "fisheye180" in words:
        screen.add(FISHEYE)
    lens_words = {w for w in words if w in _WORD_LENS}
    lens_words |= {a + b for a, b in _LENS_SPLIT.findall(base) if a + b in _WORD_LENS}
    lens = _one({_WORD_LENS[w] for w in lens_words})
    alpha_candidate = "alpha" in words or bool(_AR_PHRASE.search(base))

    strong = {_FLAT3D_STRONG[w] for w in words if w in _FLAT3D_STRONG}
    strong |= {SBS if m == "sbs" else TB for m in _FLAT3D_PAIR.findall(base)}
    loose = set(strong)
    if not (words & _VR_WORDS or screen or lens):
        # "3D SBS" in a name that also says VR is a VR file kept elsewhere
        loose |= {_FLAT3D_LOOSE[w] for w in words if w in _FLAT3D_LOOSE}
        if not loose and "3d" in words:
            loose = {SBS}           # "3D" alone: half side-by-side is the norm
    flat3d = _one(strong)

    out = {
        "stereo": _one(stereo),
        "rl": rl and _one(stereo) == SBS,
        "screen": _one(screen),
        "lens": lens,
        "alpha_candidate": alpha_candidate,
        "flat3d": flat3d,
        "flat3d_loose": _one(loose),
        "vr_word": bool(words & _VR_NAME_WORDS),
    }
    if flat3d:
        # a flat 3D marker settles both questions, whatever else the name says
        out.update({"screen": FLAT, "stereo": flat3d, "lens": None})
    return out


# ---------------------------------------------------------------- frame maths

_MASK_CACHE = {}


def _masks(ew, eh):
    """Pixel indices outside and well inside the inscribed circle of one eye."""
    key = (ew, eh)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    outside, inside = [], []
    cx, cy = (ew - 1) / 2.0, (eh - 1) / 2.0
    for y in range(eh):
        ny = (y - cy) / cy if cy else 0.0
        for x in range(ew):
            nx = (x - cx) / cx if cx else 0.0
            r = math.hypot(nx, ny)
            i = y * ew + x
            if r > 1.03:
                outside.append(i)
            elif r < 0.85:
                inside.append(i)
    _MASK_CACHE[key] = (outside, inside)
    return _MASK_CACHE[key]


def _corr(buf, w, h, ax, ay, bx, by, cw, ch):
    """Correlation between two equally sized windows of a greyscale buffer."""
    n = cw * ch
    if n < 64:
        return 0.0
    sa = sb = 0
    for j in range(ch):
        ra = (ay + j) * w + ax
        rb = (by + j) * w + bx
        sa += sum(buf[ra:ra + cw])
        sb += sum(buf[rb:rb + cw])
    ma, mb = sa / n, sb / n
    num = da = db = 0.0
    for j in range(ch):
        ra = (ay + j) * w + ax
        rb = (by + j) * w + bx
        rowa = buf[ra:ra + cw]
        rowb = buf[rb:rb + cw]
        for k in range(cw):
            u = rowa[k] - ma
            v = rowb[k] - mb
            num += u * v
            da += u * u
            db += v * v
    d = math.sqrt(da * db)
    return num / d if d > 0 else 0.0


def _bimodal_lower(buf, w, h):
    """Fraction of the lower half that is pure black or pure white.

    An alpha matte is a near-binary silhouette, so it sits at the extremes; real
    imagery does not. This is the guard against reading a packed RGB-over-alpha
    frame as top/bottom stereo -- the matte correlates with the picture above it,
    so correlation alone happily calls it a stereo pair.
    """
    hh = h // 2
    lo = buf[hh * w:2 * hh * w]
    if not lo:
        return 0.0
    return sum(1 for v in lo if v < 14 or v > 241) / len(lo)


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _eye_metrics(eye, ew, eh):
    """blk_out / blk_in (black outside vs inside the inscribed circle) and the
    aspect of the content bounding box."""
    vals = sorted(eye)
    p95 = _percentile(vals, 0.95)
    thr_black = max(8.0, 0.15 * p95)
    thr_bright = max(8.0, 0.16 * p95)

    outside, inside = _masks(ew, eh)
    if not outside or not inside:
        return None
    blk_out = sum(1 for i in outside if eye[i] < thr_black) / len(outside)
    blk_in = sum(1 for i in inside if eye[i] < thr_black) / len(inside)

    xs, ys = [], []
    for y in range(eh):
        row = y * ew
        for x in range(ew):
            if eye[row + x] > thr_bright:
                xs.append(x)
                ys.append(y)
    if len(xs) < 200:
        return None
    xs.sort()
    ys.sort()
    xe = _percentile(xs, 0.99) - _percentile(xs, 0.01)
    ye = _percentile(ys, 0.99) - _percentile(ys, 0.01)
    if ye <= 0 or eh == 0:
        return None
    bbox = (xe / ew) / (ye / eh)
    return {"blk_out": blk_out, "blk_in": blk_in, "bbox": bbox}


def is_matte_red(r, g, b):
    """Solid saturated red: what the corner-packed alpha matte is drawn in."""
    return r >= MATTE_RED_MIN and g <= r * MATTE_RED_RATIO and b <= r * MATTE_RED_RATIO


def eye_boxes(tw, th, wide):
    """(x, y, w, h) of each eye in the thumbnail: two halves for a wide frame."""
    if wide:
        hw = tw // 2
        return [(0, 0, hw, th), (hw, 0, hw, th)]
    return [(0, 0, tw, th)]


def corner_matte(rgb, tw, th, eyes):
    """Share of corner pixels (outside each eye's inscribed circle) that are
    matte red, and share that are black.

    A fisheye leaves its corners black. Passthrough scenes that pack their alpha
    matte into the corners fill part of them with solid red silhouettes, so the
    red share is well above zero while the corners hold nothing but black and
    red. An equirect frame whose corners are full of picture fails the second
    half even when that picture is red (a red sheet in a 180 scene).
    """
    total = red = black = 0
    for x0, y0, ew, eh in eyes:
        outside, _ = _masks(ew, eh)
        for i in outside:
            y, x = divmod(i, ew)
            k = ((y0 + y) * tw + x0 + x) * 3
            r, g, b = rgb[k], rgb[k + 1], rgb[k + 2]
            total += 1
            if max(r, g, b) < MATTE_BLACK_MAX:
                black += 1
            elif is_matte_red(r, g, b):
                red += 1
    if not total:
        return {"red": 0.0, "black": 0.0}
    return {"red": red / total, "black": black / total}


def matte_present(stats):
    return (stats["red"] >= MATTE_MIN_SHARE and
            stats["red"] + stats["black"] >= MATTE_MIN_CLEAN)


def rgb_to_grey(rgb):
    """BT.601 luma of an rgb24 buffer (used when no luma plane is at hand)."""
    n = len(rgb) // 3
    out = bytearray(n)
    for p in range(n):
        k = p * 3
        out[p] = (77 * rgb[k] + 150 * rgb[k + 1] + 29 * rgb[k + 2] + 128) >> 8
    return bytes(out)


def _clip(v):
    return 0 if v < 0 else 255 if v > 255 else int(v + 0.5)


def yuv_to_rgb(yuv, n):
    """Full-range planar yuv444p -> rgb24 (BT.601). Only feeds the red test,
    which is far coarser than the difference between colour matrices."""
    ys, us, vs = yuv[:n], yuv[n:2 * n], yuv[2 * n:3 * n]
    out = bytearray(3 * n)
    for i in range(n):
        y, u, v = ys[i], us[i] - 128, vs[i] - 128
        out[3 * i] = _clip(y + 1.402 * v)
        out[3 * i + 1] = _clip(y - 0.344136 * u - 0.714136 * v)
        out[3 * i + 2] = _clip(y + 1.772 * u)
    return bytes(out)


def blank_corners(grey, tw, eyes):
    """Black out everything outside each eye's inscribed circle."""
    buf = bytearray(grey)
    for x0, y0, ew, eh in eyes:
        outside, _ = _masks(ew, eh)
        for i in outside:
            y, x = divmod(i, ew)
            buf[(y0 + y) * tw + x0 + x] = 0
    return bytes(buf)


def frame_metrics(rgb, tw, th, wide, grey=None):
    """Everything measured on one frame. grey is the luma plane when the
    decoder supplied one; otherwise it is derived from rgb."""
    eyes = eye_boxes(tw, th, wide)
    matte = corner_matte(rgb, tw, th, eyes)
    has_matte = matte_present(matte)
    buf = grey if grey is not None else rgb_to_grey(rgb)
    if has_matte:
        # the corners belong to the matte, not to the picture: leaving the
        # silhouettes in would widen the content box and cost the disc its shape
        buf = blank_corners(buf, tw, eyes)
    hw, hh = tw // 2, th // 2
    lr = _corr(buf, tw, th, 0, 0, hw, 0, hw, th)
    tb = _corr(buf, tw, th, 0, 0, 0, hh, tw, hh)
    # measure the projection on the presumed left eye of wide frames
    if wide:
        ew, eh = hw, th
        eye = bytearray()
        for y in range(th):
            eye += buf[y * tw:y * tw + hw]
    else:
        ew, eh = tw, th
        eye = buf
    m = _eye_metrics(eye, ew, eh)
    if not m:
        return None
    m.update({"lr": lr, "tb": tb, "alpha_lower": _bimodal_lower(buf, tw, th),
              "matte_red": matte["red"], "matte_black": matte["black"],
              "matte": has_matte})
    return m


def combine_frames(frames):
    """Median of each metric over the frames; the matte must be in every one
    (matte_any records whether at least one frame showed it)."""
    if not frames:
        return None
    out = {}
    for k in ("lr", "tb", "blk_out", "blk_in", "bbox", "alpha_lower",
              "matte_red", "matte_black"):
        vals = sorted(f[k] for f in frames)
        out[k] = round(_percentile(vals, 0.5), 4)
    out["matte"] = len(frames) >= 2 and all(f["matte"] for f in frames)
    # a frame can miss the matte (a fade, a dark cut); when the filename
    # already says passthrough/alpha, one matching frame is enough
    out["matte_any"] = any(f["matte"] for f in frames)
    out["frames"] = len(frames)
    return out


def thumb_size(w, h):
    tw = ANALYSIS_W - ANALYSIS_W % 2
    th = max(2, int(round(ANALYSIS_W / (w / h))))
    th -= th % 2                      # keep halves equal; an odd height splits unevenly
    return tw, th


def grab(cfg, path, ts, tw, th):
    """One downscaled frame as full-range planar yuv444p bytes, or None.

    The Y plane is byte for byte what "-pix_fmt gray" gives, which is what the
    shape thresholds were validated on; U and V feed the red test. One decode
    serves both.
    """
    # Decode keyframes only: the nearest keyframe is as good a sample as the
    # exact timestamp and avoids decoding every frame since the last one
    # (about 15 s -> 1.3 s per 8K HEVC frame on a network share).
    cmd = [cfg["ffmpegPath"], "-nostdin", "-v", "error", "-skip_frame", "nokey", "-ss", f"{ts:.3f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale={tw}:{th}:out_range=pc",
           "-pix_fmt", "yuv444p", "-f", "rawvideo", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0 or len(p.stdout) < tw * th * 3:
        return None
    return p.stdout[:tw * th * 3]


def probe(cfg, path, w, h, duration):
    """Median metrics over two frames."""
    tw, th = thumb_size(w, h)
    wide = w / h > 1.5
    frames = []
    n = tw * th
    for frac in (0.4, 0.6):
        yuv = grab(cfg, path, (duration or 600) * frac, tw, th)
        if yuv is None:
            continue
        m = frame_metrics(yuv_to_rgb(yuv, n), tw, th, wide, grey=yuv[:n])
        if m:
            frames.append(m)
    return combine_frames(frames)


# ------------------------------------------------------------- classification

def eye_aspect(w, h, stereo):
    if stereo == SBS:
        return (w / 2) / h
    if stereo == TB:
        return w / (h / 2)
    return w / h


def screen_from_eye(a):
    """One eye's shape names the projection: 180 covers a square, 360 a 2:1
    equirect, 16:9 (and 16:10) is ordinary flat video. A full side-by-side
    flat 3D file (3840x1080, 7680x2160) is two 16:9 eyes, so its whole-frame
    aspect of 3.2-3.7 reads as FLAT once the SBS split is accepted."""
    if abs(a - 1.0) < 0.28:
        return DOME
    if 1.6 <= a < 1.87:
        return FLAT
    if abs(a - 2.0) < 0.13:
        return SPHERE
    return None


def classify(w, h, res):
    """(screen, stereo, why) from frame measurements alone. screen is None when
    the shape is not recognised or the frame looks like a packed matte."""
    if not res:
        return None, None, "no probe"
    why = []
    # stereo: accept a layout only if it implies a plausible eye shape
    if res["lr"] >= 0.55 and screen_from_eye(eye_aspect(w, h, SBS)):
        stereo = SBS
        why.append(f"lr={res['lr']:.2f}")
    elif res["tb"] >= 0.60 and screen_from_eye(eye_aspect(w, h, TB)):
        if res.get("alpha_lower", 0.0) > ALPHA_BIMODAL:
            # lower half is a binary matte, not an eye: packed alpha, not stereo
            return None, None, f"packed alpha? lower half {res['alpha_lower']:.2f} binary"
        stereo = TB
        why.append(f"tb={res['tb']:.2f}")
    else:
        stereo = MONO
        why.append(f"lr={res['lr']:.2f} tb={res['tb']:.2f}")

    ea = eye_aspect(w, h, stereo)
    by_shape = screen_from_eye(ea)
    if (by_shape != FLAT and FISH_BBOX_LO <= res["bbox"] <= FISH_BBOX_HI
            and res["blk_out"] >= FISH_BLKOUT):
        screen = FISHEYE
        why.append(f"disc bbox={res['bbox']:.2f} blk_out={res['blk_out']:.2f}")
    elif by_shape == DOME and res.get("matte"):
        # the corner-packed matte is a fisheye format: it lives in the corners
        # the discs leave free, and every matte scene seen during calibration
        # was a fisheye. The disc test alone misses a few whose picture is dark
        # at the top of the disc.
        screen = FISHEYE
        why.append(f"corner matte, bbox={res['bbox']:.2f}")
    else:
        screen = by_shape
        why.append(f"eye={ea:.2f} bbox={res['bbox']:.2f}")
    return screen, stereo, ", ".join(why)


def resolve(fn, screen_px, stereo_px, alpha_px, fov=None):
    """Merge filename markers, the watermark FOV and the pixel verdict into the
    projection tags a scene should carry. fov is ("190"|"200"|"220", vrca)."""
    stereo = fn["stereo"] or stereo_px
    lens = fn["lens"]
    if lens:
        screen = FISHEYE
    else:
        screen = fn["screen"] or screen_px
    if screen == FISHEYE and not lens and fov:
        value, vrca = fov
        lens = VRCA220 if (vrca and value == "220") else FOV_LENS.get(value)

    want = set()
    if alpha_px:
        want.add(ALPHA)
    if not screen:
        want.add(UNRESOLVED)
        return want
    want.add(screen)
    if lens:
        want.add(lens)
    # FLAT already means mono 2D; MONO is only for mono VR (DOME/SPHERE + MONO)
    if stereo and not (screen == FLAT and stereo == MONO):
        want.add(stereo)
    if fn["rl"] and stereo == SBS:
        want.add(RL)
    return want


# "190°" (OCR often renders the degree sign as o, O, 0 or º), or the number
# directly before "FOV" when the sign is lost altogether
_FOV_RE = re.compile(r"\b(180|190|200|220)\s*[°oOº]")
_FOV_WORD_RE = re.compile(r"\b(180|190|200|220)\s*[°oOº0*]?\s*F\s*[O0]\s*V\b", re.IGNORECASE)


def fov_from_text(txt):
    """(fov, vrca) from one OCR pass, or None."""
    m = _FOV_RE.search(txt or "") or _FOV_WORD_RE.search(txt or "")
    if not m:
        return None
    return m.group(1), "vrca" in (txt or "").lower()


# Where SLR burns the "SLR 190° FOV FISHEYE" text: centred at the top of the
# frame, across the dead space between the two eyes (current releases), or
# near the top of the left eye's right half (older ones). The centre crop is
# tried first; a crop that splits the text at the seam still often reads.
FOV_CROPS = (
    "crop=iw*0.40:ih*0.12:iw*0.30:0,scale=iw*2:-1,format=gray",
    "crop=iw/2:ih:0:0,crop=iw*0.50:ih*0.20:iw*0.50:ih*0.00,scale=iw*2:-1,format=gray",
)


_SLR_RE = re.compile(r"(?<![a-z0-9])(slr|sexlikereal)", re.IGNORECASE)


def fov_skip_reason(fn, screen_px, alpha, path):
    """Why reading the SLR watermark cannot help this scene, or None when it can.

    The watermark only names a fisheye lens, so there is nothing to read when
    the filename already names the lens or the scene does not end up FISHEYE
    (a filename screen marker beats the pixels). SLR's own passthrough
    releases carry the watermark, other studios' corner-matte scenes never do,
    so a matte scene is only read when its path mentions SLR / SexLikeReal.
    """
    if fn["lens"]:
        return "lens from filename"
    if (fn["screen"] or screen_px) != FISHEYE:
        return "not fisheye"
    if alpha and not _SLR_RE.search(path or ""):
        return "passthrough not from SLR"
    return None


def read_fov(cfg, path, duration):
    """Read the burned-in SLR FOV. Requires two agreeing frames."""
    if not os.path.exists(cfg["tesseractPath"]):
        return None
    tmp = f"/tmp/vrq_fov_{os.getpid()}.png"
    votes = {}
    try:
        for frac in (0.30, 0.50, 0.70, 0.20, 0.60, 0.80, 0.40, 0.90):
            ts = (duration or 1200) * frac
            hit = None
            for crop in FOV_CROPS:
                r = subprocess.run([cfg["ffmpegPath"], "-nostdin", "-v", "error", "-y",
                                    "-skip_frame", "nokey", "-ss", f"{ts:.3f}",
                                    "-i", path, "-frames:v", "1", "-vf", crop, tmp],
                                   capture_output=True, timeout=300)
                if r.returncode != 0 or not os.path.exists(tmp):
                    continue
                txt = subprocess.run([cfg["tesseractPath"], tmp, "-", "--psm", "6"],
                                     capture_output=True, text=True, timeout=120).stdout
                hit = fov_from_text(txt)
                if hit:
                    break
            if hit:
                votes[hit] = votes.get(hit, 0) + 1
                if votes[hit] >= 2:
                    return hit
    except subprocess.TimeoutExpired:
        pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return None


# ------------------------------------------------------------------ quality

def quality_names(cfg):
    return (cfg["tag8k"], cfg["tag7k"], cfg["tag6kHbr"], cfg["parentTag"])


def tier_of(scene, cfg):
    """Which quality tag (setting key) this scene earns, or None."""
    files = scene.get("files") or []
    if not files:
        return None
    # a scene can hold several files; the biggest one is the one worth judging
    f = max(files, key=lambda x: x.get("size") or 0)
    width = f.get("width") or 0
    mbit = (f.get("bit_rate") or 0) / 1e6
    if width >= cfg["min8kWidth"]:
        return "tag8k"
    if width >= cfg["min7kWidth"]:
        return "tag7k"
    if width >= cfg["min6kWidth"] and mbit >= cfg["min6kBitrateMbit"]:
        return "tag6kHbr"
    return None


def quality_want(scene, cfg):
    tier = tier_of(scene, cfg)
    return {cfg[tier], cfg["parentTag"]} if tier else set()


# ------------------------------------------------------------------ tag diffing

def diff_tags(have, scope, want):
    """New tag set, or None when nothing changes.

    Only tags in scope are this pass's business: those not wanted are dropped,
    wanted ones added. Everything outside scope is kept as it is.
    """
    new = (set(have) - set(scope)) | set(want)
    return None if new == set(have) else new


def settled(have_names):
    return bool(set(have_names) & set(SETTLED_TAGS))


def hook_should_skip(ctx):
    """Our own tag write fires Scene.Update.Post again; ignoring updates that
    touched nothing but tags stops the hook from calling itself."""
    fields = set(ctx.get("inputFields") or [])
    return ctx.get("type") == "Scene.Update.Post" and bool(fields) and fields <= {"id", "tag_ids"}


def path_matches(cfg, path):
    return not cfg["pathFilter"] or cfg["pathFilter"] in (path or "")


VR_SHAPE_MIN_WIDTH = 3840
VR_SHAPE_TOLERANCE = 0.05


def looks_vr(scene):
    """Whether a scene outside the path filter is VR, from Stash's metadata of
    its primary file alone (nothing is decoded): a 2:1 or square frame at least
    3840 wide, or a VR marker in the file name. A flat 3D name never counts."""
    f = (scene.get("files") or [{}])[0]
    fn = parse_filename(f.get("path"))
    if fn["flat3d"]:
        return False
    if fn["screen"] or fn["lens"] or fn["vr_word"]:
        return True
    w, h = f.get("width") or 0, f.get("height") or 0
    if w < VR_SHAPE_MIN_WIDTH or not h:
        return False
    a = w / h
    return abs(a - 2.0) <= VR_SHAPE_TOLERANCE or abs(a - 1.0) <= VR_SHAPE_TOLERANCE


def load_config(stored):
    cfg = dict(DEFAULTS)
    for k, v in (stored or {}).items():
        if v not in (None, ""):
            cfg[k] = v
    for k in NUMERIC:
        cfg[k] = float(cfg[k])
    for k in BOOLEAN:
        cfg[k] = bool(cfg[k])
    return cfg


# ------------------------------------------------------------------ stash

class Stash:
    def __init__(self, conn):
        scheme = conn.get("Scheme") or "http"
        port = conn.get("Port") or 9999
        self.url = f"{scheme}://localhost:{port}/graphql"
        cookie = conn.get("SessionCookie") or {}
        self.headers = {"Content-Type": "application/json"}
        if cookie.get("Name"):
            self.headers["Cookie"] = f"{cookie['Name']}={cookie['Value']}"

    def use_api_key(self, key):
        """Authenticate with an API key instead of the task's session cookie,
        which Stash expires after an hour."""
        self.headers.pop("Cookie", None)
        self.headers["ApiKey"] = key

    def call(self, query, variables=None):
        body = json.dumps({"query": query, "variables": variables or {}}).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        if out.get("errors"):
            raise RuntimeError(out["errors"][0].get("message"))
        return out["data"]


TAG_BY_NAME = """query($n:String!){findTags(tag_filter:{name:{value:$n,modifier:EQUALS}},
  filter:{per_page:2}){tags{id name parents{id}}}}"""
TAG_CREATE = "mutation($i:TagCreateInput!){tagCreate(input:$i){id name}}"
TAG_UPDATE = "mutation($i:TagUpdateInput!){tagUpdate(input:$i){id}}"
SCENE_FIELDS = "id tags{id name} files{width height duration bit_rate size path}"
SCENE_PAGE = """query($p:Int!,$f:String!){findScenes(
  scene_filter:{path:{value:$f,modifier:INCLUDES}},
  filter:{per_page:100,page:$p,sort:"id",direction:ASC}){
  count scenes{%s}}}""" % SCENE_FIELDS
SCENE_ONE = "query($id:ID!){findScene(id:$id){%s}}" % SCENE_FIELDS
SCENE_PAGE_REGEX = SCENE_PAGE.replace("modifier:INCLUDES", "modifier:MATCHES_REGEX")
# candidates for the flat 3D filename scan; parse_filename() has the last word
FLAT3D_PATH_REGEX = (r"(?i)(^|[^a-z0-9])(3d|sbs|hsbs|fsbs|lrf|ou|hou|tab|tbf|"
                     r"(half|full)(sbs|ou|tb))([^a-z0-9]|$)")
FLAT3D_SCOPE = (FLAT, SBS, TB, MONO, RL)
# candidates for the VR-shaped check outside the path filter; looks_vr() has the
# last word. MIN(width, height) > 1439 (Stash's FULL_HD GREATER_THAN) holds for
# every 2:1 or square frame at least 3840 wide.
SCENE_PAGE_BIG = """query($p:Int!,$f:String!){findScenes(
  scene_filter:{resolution:{value:FULL_HD,modifier:GREATER_THAN},
                path:{value:$f,modifier:EXCLUDES}},
  filter:{per_page:100,page:$p,sort:"id",direction:ASC}){
  count scenes{%s}}}""" % SCENE_FIELDS
VR_NAME_PATH_REGEX = r"(?i)(^|[^a-z0-9])(vr|180|360|fisheye|3dh|3dv|mkx|vrca|rf)"
SCENE_UPDATE = "mutation($i:SceneUpdateInput!){sceneUpdate(input:$i){id}}"


def ensure_tags(stash, cfg):
    """Look up or create every managed tag, file the quality tags under their
    parent, and return {name: id}."""
    found = {}

    def get_or_create(name):
        hits = stash.call(TAG_BY_NAME, {"n": name})["findTags"]["tags"]
        if hits:
            found[name] = hits[0]
            return hits[0]
        made = stash.call(TAG_CREATE, {"i": {"name": name}})["tagCreate"]
        made["parents"] = []
        log("i", f"created tag {name}")
        found[name] = made
        return made

    for name in PROJECTION_TAGS + (SKIP,) + quality_names(cfg):
        if name not in found:
            get_or_create(name)

    parent = found[cfg["parentTag"]]
    for key in ("tag8k", "tag7k", "tag6kHbr"):
        child = found[cfg[key]]
        current = {p["id"] for p in (child.get("parents") or [])}
        if parent["id"] not in current:
            stash.call(TAG_UPDATE, {"i": {"id": child["id"],
                                          "parent_ids": sorted(current | {parent["id"]})}})
            log("i", f"filed {cfg[key]} under {cfg['parentTag']}")
    return {n: t["id"] for n, t in found.items()}


def measure_projection(cfg, scene):
    """(wanted projection tag names, reason) or (None, reason) when the scene
    cannot be measured and its projection tags must be left alone."""
    files = scene.get("files") or []
    if not files:
        return None, "no file"
    f = files[0]                    # the primary file is the one stash-vr streams
    w, h, dur, path = f.get("width"), f.get("height"), f.get("duration"), f.get("path")
    if not w or not h or w < cfg["minWidth"]:
        return None, "too small"
    if not path or not os.path.exists(path):
        return None, "file missing"

    fn = parse_filename(path)
    res = probe(cfg, path, w, h, dur)
    if res and fn["alpha_candidate"] and res.get("matte_any"):
        res["matte"] = True
    screen, stereo, why = classify(w, h, res)
    alpha = bool(res and res["matte"])
    if res:
        why += f", corners red={res['matte_red']:.3f} black={res['matte_black']:.2f}"
    if alpha and not fn["alpha_candidate"]:
        why += ", matte without a filename marker"
    elif fn["alpha_candidate"] and not alpha:
        why += ", named passthrough/alpha but no corner matte"

    fov = None
    skip = fov_skip_reason(fn, screen, alpha, path) if cfg["readFovWatermark"] else "off"
    if skip is None:
        fov = read_fov(cfg, path, dur)
        if fov:
            why += f", watermark {fov[0]}deg"
    elif skip == "passthrough not from SLR":
        why += ", watermark not read (passthrough not from SLR)"
    marks = [k for k in ("stereo", "screen", "lens") if fn[k]] + (["rl"] if fn["rl"] else [])
    if marks:
        why += ", filename " + "+".join(str(fn[k]) if k != "rl" else "RL" for k in marks)
    return resolve(fn, screen, stereo, alpha, fov), why


def process_scene(stash, cfg, scene, ids, mode):
    """Apply this pass to one scene; returns a log line when it changed."""
    have_names = {t["name"] for t in scene.get("tags") or []}
    if SKIP in have_names:
        return None
    have = {t["id"] for t in scene.get("tags") or []}

    if mode == "clear":
        scope_names = set(PROJECTION_TAGS) | set(quality_names(cfg))
        want_names, why = set(), "cleared"
    else:
        scope_names = set(quality_names(cfg))
        want_names = quality_want(scene, cfg)
        why = "quality"
        remeasure = mode == "retag" or cfg["overwrite"]
        if remeasure or not settled(have_names):
            proj, reason = measure_projection(cfg, scene)
            if proj is not None:
                scope_names |= set(PROJECTION_TAGS)
                want_names |= proj
                why = reason
            elif reason == "file missing":
                log("w", f"scene {scene['id']}: file missing, projection skipped")

    new = diff_tags(have, {ids[n] for n in scope_names}, {ids[n] for n in want_names})
    if new is None:
        return None
    stash.call(SCENE_UPDATE, {"i": {"id": scene["id"], "tag_ids": sorted(new)}})
    managed = scope_names & {n for n in ids if ids[n] in new}
    return f"{' '.join(sorted(managed)) or '(none)'}  ({why})"


def process_flat_scene(stash, cfg, scene, ids, mode):
    """Flat 3D outside the VR path: filename only, nothing is decoded. A file
    without a flat 3D marker is not touched at all (no tag reads as flat 2D)."""
    have_names = {t["name"] for t in scene.get("tags") or []}
    if SKIP in have_names:
        return None
    path = ((scene.get("files") or [{}])[0].get("path")) or ""
    stereo = parse_filename(path)["flat3d_loose"]
    if not stereo:
        return None
    want_names = set() if mode == "clear" else {FLAT, stereo}
    have = {t["id"] for t in scene.get("tags") or []}
    new = diff_tags(have, {ids[n] for n in FLAT3D_SCOPE}, {ids[n] for n in want_names})
    if new is None:
        return None
    stash.call(SCENE_UPDATE, {"i": {"id": scene["id"], "tag_ids": sorted(new)}})
    return f"{' '.join(sorted(want_names)) or '(none)'}  (flat 3D filename)"


def _pages(stash, query, value):
    """Yield (scene, fraction done) over every page of a scene query."""
    page, seen = 1, 0
    while True:
        d = stash.call(query, {"p": page, "f": value})["findScenes"]
        if not d["scenes"]:
            return
        for sc in d["scenes"]:
            seen += 1
            yield sc, min(1.0, seen / max(d["count"], 1))
        if seen >= d["count"]:
            return
        page += 1


def candidates(stash, cfg):
    """Every scene a task run visits as (scene, handler, kind), in ascending id
    order. Each scene is visited once: the path filter wins, then the VR-shaped
    check, then the flat 3D name check."""
    chosen = {}

    def add(sc, handler, kind):
        if sc["id"] not in chosen:
            chosen[sc["id"]] = (sc, handler, kind)

    for sc, _ in _pages(stash, SCENE_PAGE, cfg["pathFilter"]):
        add(sc, process_scene, "VR")

    def outside(query, value):
        for sc, _ in _pages(stash, query, value):
            if sc["id"] not in chosen and not path_matches(
                    cfg, (sc.get("files") or [{}])[0].get("path")):
                yield sc

    if cfg["measureVrShapedOutside"]:
        for query, value in ((SCENE_PAGE_BIG, cfg["pathFilter"]),
                             (SCENE_PAGE_REGEX, VR_NAME_PATH_REGEX)):
            for sc in outside(query, value):
                if looks_vr(sc):
                    add(sc, process_scene, "VR-shaped")
    if cfg["flat3dFilenameScan"]:
        for sc in outside(SCENE_PAGE_REGEX, FLAT3D_PATH_REGEX):
            add(sc, process_flat_scene, "flat 3D")
    return sorted(chosen.values(), key=lambda c: int(c[0]["id"]))


STATE_FILE = "vrQualityTags.state.json"
RESUME_MAX_AGE = 7 * 86400          # an older unfinished retag starts over


def progress(fraction):
    # Stash reads "\x01p\x02<float>" as the task's progress, 0 to 1
    log("p", f"{min(1.0, max(0.0, fraction)):.4f}")


def state_path(conn):
    """The resume state lives next to the plugin: Stash passes its directory as
    server_connection.PluginDir."""
    d = (conn or {}).get("PluginDir") or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(d, STATE_FILE)


class RetagState:
    """Where an unfinished retag stopped: the last scene it completed and when
    the run started. Scenes are visited in ascending id order, so everything up
    to that id is done."""

    def __init__(self, path, now=None):
        self.path = path
        self.started = time.time() if now is None else now
        self.last_id = None
        self.warned = False

    @classmethod
    def load(cls, path, now=None):
        """The saved state of an unfinished retag, or None when there is none,
        it is unreadable or older than RESUME_MAX_AGE."""
        now = time.time() if now is None else now
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        started, last = data.get("started"), data.get("last_id")
        if not isinstance(started, (int, float)) or isinstance(started, bool):
            return None
        if not isinstance(last, int) or isinstance(last, bool):
            return None
        if not 0 <= now - started < RESUME_MAX_AGE:
            return None
        st = cls(path, started)
        st.last_id = last
        return st

    def done(self, scene_id):
        self.last_id = int(scene_id)
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"started": self.started, "last_id": self.last_id}, f)
            os.replace(tmp, self.path)
        except OSError as e:
            if not self.warned:
                self.warned = True
                log("w", f"cannot save the resume state to {self.path}: {e}; "
                         "an interrupted retag will start over")

    def clear(self):
        for p in (self.path, self.path + ".tmp"):
            try:
                os.remove(p)
            except OSError:
                pass


def run_all(stash, cfg, ids, mode, state=None):
    """One task run over every candidate scene. state (a RetagState) makes the
    run resumable: scenes up to state.last_id are skipped, each completed scene
    is recorded, and the state is cleared once the run completes."""
    todo = candidates(stash, cfg)
    kinds = {}
    for _, _, kind in todo:
        kinds[kind] = kinds.get(kind, 0) + 1
    log("i", f"{mode}: {len(todo)} scenes to examine ("
             + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) + ")")
    start = 0
    if state is not None and state.last_id is not None:
        start = sum(1 for sc, _, _ in todo if int(sc["id"]) <= state.last_id)
        log("i", f"resuming after scene {state.last_id}: {start} of {len(todo)} scenes "
                 f"already done by the run started "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(state.started))}")
    progress(start / len(todo) if todo else 1.0)
    changed = 0
    for n, (sc, handler, _) in enumerate(todo[start:], start + 1):
        try:
            r = handler(stash, cfg, sc, ids, mode)
        except Exception as e:
            log("e", f"scene {sc['id']}: {type(e).__name__}: {e}")
            r = None
        if r:
            changed += 1
            log("i", f"scene {sc['id']}: {r}")
        if state is not None:
            state.done(sc["id"])
        progress(n / len(todo))
    if state is not None:
        state.clear()
    log("i", f"done ({mode}): {len(todo) - start} scenes examined, {changed} changed")


def route(cfg, scene):
    """The handler for one scene in the hook, or None to leave it alone."""
    if not scene:
        return None
    if path_matches(cfg, (scene.get("files") or [{}])[0].get("path")):
        return process_scene
    if cfg["measureVrShapedOutside"] and looks_vr(scene):
        return process_scene
    if cfg["flat3dFilenameScan"]:
        return process_flat_scene
    return None


def main():
    payload = json.loads(sys.stdin.read())
    stash = Stash(payload.get("server_connection") or {})
    args = payload.get("args") or {}
    try:
        stored = (stash.call("{configuration{plugins}}")["configuration"]["plugins"]
                  or {}).get("vrQualityTags") or {}
    except Exception:
        stored = {}
    cfg = load_config(stored)
    if cfg.get("apiKey"):
        stash.use_api_key(cfg["apiKey"])
    mode = args.get("mode") or "hook"
    if mode == "all":                   # task name of the pre-merge quality plugin
        mode = "untagged"
    ids = ensure_tags(stash, cfg)

    if mode in ("retag", "retag_fresh"):
        path = state_path(payload.get("server_connection"))
        state = RetagState.load(path) if mode == "retag" else None
        if state is None:
            if mode == "retag_fresh":
                log("i", "retag from the beginning; any saved progress is discarded")
            state = RetagState(path)
        run_all(stash, cfg, ids, "retag", state)
    elif mode in ("untagged", "clear"):
        run_all(stash, cfg, ids, mode)
    else:
        ctx = args.get("hookContext") or {}
        sid = ctx.get("id") or args.get("scene_id")
        if sid is None or hook_should_skip(ctx):
            print(json.dumps({"output": "ok"}))
            return
        scene = stash.call(SCENE_ONE, {"id": str(sid)})["findScene"]
        handler = route(cfg, scene)
        r = handler(stash, cfg, scene, ids, "untagged") if handler else None
        if r:
            log("i", f"scene {sid}: {r}")

    print(json.dumps({"output": "ok"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("e", f"{type(e).__name__}: {e}")
        if getattr(e, "code", None) == 401:
            log("e", "Stash rejected the request. Long tasks outlive the hour-long session "
                     "Stash gives a plugin; set an API key in the plugin settings "
                     "(Settings -> Security -> API key) and run the task again.")
        print(json.dumps({"error": str(e)}))
