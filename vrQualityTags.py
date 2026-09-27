"""Tag VR scenes with everything stash-vr needs to play them right.

One measurement pass per scene writes all of it: projection, lens, stereo
layout, eye order, the corner-packed alpha matte of passthrough scenes, and the
quality tier. The tag vocabulary is exactly stash-vr's default video rules, so
every tag written here has an effect in the headset.

WHERE THE ANSWER COMES FROM, IN ORDER OF AUTHORITY

  metadata   the file's own spherical / stereo 3D metadata, read with one
             ffprobe call before any frame is decoded. Rare, and often wrong
             in the wild (an equirect 360 claim on every 180 file of a
             studio, 2D on a stereo pair), so each part only counts where the
             frame agrees with it: see vet_claim().

  SLR        with the slrLookup setting, SexLikeReal's scene API for scenes
             that carry a sexlikereal.com URL: 180 or 360, the fisheye lens,
             stereo, and passthrough (native alpha, chroma key). Cached for
             90 days. Its projection must agree with the frame in coarse
             shape (a download can be an equirect conversion of a fisheye
             scene), otherwise the answer is set aside; only SLR's equirect
             180 beats a disc-only fisheye (vignetted 180s pass the disc
             test). A lens only inferred from viewAngle yields to the
             watermark. See SlrLookup, vet_claim().

  filename   explicit markers (_LR_, _TB_, _RL_, MKX200, RF52, F180, ...). Most files
             carry none, but when present they are deliberate and beat any
             measurement.

  watermark  SLR burns "SLR 190/200/220 FOV" into the top of the frame.
             Nothing measurable separates 190 from 200 from 220, so where the
             text exists it is the only authority. Two agreeing frames are
             required; a single OCR hit is not trusted. Not read where it
             cannot help: see fov_skip_reason().

  pixels     two frames decoded to a 256px thumbnail:
               lr / tb  how well the frame halves match, tile by tile, with
                        the parallax between the eyes searched; a layout is
                        only accepted if it implies a plausible eye shape.
               wrap     whether the right edge continues into the left one:
                        only then is a mono 2:1 frame a 360 (stereo_tiles(),
                        wrap_ratio()).
               bbox     shape of the lit region of one eye: a fisheye is a disc
                        (as wide as high), a 180 equirect a barrel (wider).
               corners  outside the inscribed circle of each eye a fisheye is
                        black. Passthrough scenes pack their alpha matte there
                        as solid saturated red shapes; see corner_matte().
               lower    a near-binary lower half is a packed matte, not a
                        second eye (guard against RGB-over-alpha read as TB).

The pixel-measured corner matte (Alpha) always stands, whatever the sources
above say.

The quality tier is measured from the file itself: width decides it, and in
the 6K band the bitrate has to agree too, because that is where upscales hide.
A file whose pixels lack the detail of their resolution (an upscale), or whose
bitrate is too low to keep what little they show, also gets Low Detail: see
low_detail().

For a stereo 180 or fisheye scene, the vertical offset between its eyes is
measured from four keyframes (two shared with the projection) and stored in
the scene custom field vr_vertical_offset, in degrees: see the "stereo
alignment" section below and README "Stereo alignment".

Outside the VR path a file is measured only when Stash's metadata alone says it
is VR (a 2:1 or square frame at least 3840 wide, or a VR marker in its name).
Flat (non-VR) stereoscopic 3D files there are recognised from their names alone
(3D, SBS, Half-SBS, LRF, HOU, ...) and get FLAT + SBS or FLAT + TB. Every other
file there is left untouched.

Pure standard library on purpose: the plugin interpreter has no numpy or PIL,
and arithmetic over a 256px thumbnail does not need them. The stereo matching
at 1024 px per eye gets its correlations from big-integer multiplication.
"""
import array
import cmath
import concurrent.futures
import itertools
import json
import math
import multiprocessing
import operator
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

VERSION = "2.7.1"

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
    # tag scenes whose file lacks the detail of its tier with LOW_DETAIL
    "detectLowDetail": True,
    # measure the vertical offset between the eyes of stereo scenes into the
    # scene custom field vr_vertical_offset
    "detectVerticalOffset": True,
    # projection
    "readFovWatermark": True,
    "overwrite": False,
    "minWidth": 1920,
    "ffmpegPath": "/usr/bin/ffmpeg",
    "ffprobePath": "/usr/bin/ffprobe",
    "tesseractPath": "/usr/bin/tesseract",
    # VR-shaped files outside the VR path, decided from metadata only
    "measureVrShapedOutside": True,
    # flat 3D outside the VR path, from filenames only
    "flat3dFilenameScan": True,
    # optional Stash API key; the session Stash hands a task expires after an
    # hour, which ends long runs part way
    "apiKey": "",
    # ask SexLikeReal's API about scenes that carry a sexlikereal.com URL
    "slrLookup": False,
    # scenes measured at the same time by a task (the hook does one)
    "workers": 4,
}

NUMERIC = ("min8kWidth", "min7kWidth", "min6kWidth", "min6kBitrateMbit", "minWidth")
MAX_WORKERS = 16
BOOLEAN = ("readFovWatermark", "overwrite", "measureVrShapedOutside", "flat3dFilenameScan",
           "slrLookup", "detectLowDetail", "detectVerticalOffset")

# ---------------------------------------------------------------- vocabulary
# Mirrors stash-vr's default video rules (internal/config/settings.go).
DOME, SPHERE, FISHEYE, FLAT = "DOME", "SPHERE", "FISHEYE", "FLAT"
RF52, MKX200, MKX220, VRCA220 = "RF52", "MKX200", "MKX220", "VRCA220"
SBS, TB, MONO, RL = "SBS", "TB", "MONO", "RL"
ALPHA = "Alpha"
CHROMA = "Chroma Key"                # green-screen passthrough, from the SLR lookup
UNRESOLVED = "VRP: Unresolved"      # probed but not classifiable; stops endless re-probing
SKIP = "VRP: Skip"                  # user-applied opt-out; never written or removed here

SCREEN_TAGS = (DOME, SPHERE, FISHEYE, FLAT)
LENS_TAGS = (RF52, MKX200, MKX220, VRCA220)
STEREO_TAGS = (SBS, TB, MONO)
PROJECTION_TAGS = SCREEN_TAGS + LENS_TAGS + STEREO_TAGS + (RL, ALPHA, CHROMA, UNRESOLVED)

# A scene carrying any of these has been classified before (by this plugin or
# by hand) and is not re-measured unless asked to.
SETTLED_TAGS = SCREEN_TAGS + LENS_TAGS + ("CUBEMAP", "EAC", UNRESOLVED)

FOV_LENS = {"190": RF52, "200": MKX200, "220": MKX220}

ANALYSIS_W = 256
ALPHA_BIMODAL = 0.75    # lower half this binary is a matte, not a second eye
FISH_BBOX_LO, FISH_BBOX_HI, FISH_BLKOUT = 0.94, 1.06, 0.72

# Stereo and 360 (see README "How detection was calibrated").
STEREO_TILE = 32            # tile side in thumbnail pixels
STEREO_TILE_MIN_STD = 6     # a tile flatter than this has nothing to match
STEREO_SHIFT = 0.06         # parallax searched, as a fraction of the eye width
SBS_MIN, TB_MIN = 0.55, 0.60    # median tile match of a stereo pair
STEREO_NONE = 0.30          # below this the halves have nothing in common
MIN_TEXTURED = 0.25         # share of textured tiles below which a frame is a
                            # fade or a title card and is not measured
WRAP_MAX = 0.25             # seam difference / far difference of a 360
WRAP_MIN_STD = 5            # edge columns flatter than this prove nothing
WRAP_MIN_FAR = 4

# Corner matte (see README "How detection was calibrated").
MATTE_RED_MIN = 96          # R of a matte pixel after downscaling
MATTE_RED_RATIO = 0.30      # G and B at most this fraction of R
MATTE_BLACK_MAX = 32        # max(R,G,B) below this counts as black
MATTE_MIN_SHARE = 0.006     # red share of the corner pixels
MATTE_MIN_CLEAN = 0.80      # black + red share: the corners hold nothing else


_LOG_LOCK = threading.Lock()


def log(level, msg):
    # Stash reads plugin logs off stderr, one prefixed line at a time; the
    # lock keeps lines from worker threads whole
    with _LOG_LOCK:
        print(f"\x01{level}\x02{msg}", file=sys.stderr, flush=True)


# The pure-Python arithmetic (the frame metrics and the detail spectra) runs
# in this process pool while a task has one: worker threads only overlap the
# ffmpeg decodes, since Python runs one thread's arithmetic at a time.
_CPU = None
_CPU_LOCK = threading.Lock()


def start_cpu_pool(n):
    """Give the task n processes for the arithmetic. "spawn" starts them
    clean: forking a process that already runs threads is unsafe."""
    global _CPU
    try:
        _CPU = concurrent.futures.ProcessPoolExecutor(
            n, mp_context=multiprocessing.get_context("spawn"))
    except (OSError, ValueError, NotImplementedError) as e:
        log("w", f"no worker processes ({e}); the arithmetic runs in the threads")
        _CPU = None


def stop_cpu_pool():
    global _CPU
    pool, _CPU = _CPU, None
    if pool is not None:
        pool.shutdown(cancel_futures=True)


def crunch(fn, *args):
    """fn(*args) in the task's process pool, or here without one. A pool
    that breaks (a worker killed, processes not allowed) is dropped once with
    a warning and the rest of the run computes in the threads: the result is
    the same either way."""
    global _CPU
    pool = _CPU
    if pool is not None:
        try:
            return pool.submit(fn, *args).result()
        except (concurrent.futures.BrokenExecutor, OSError, RuntimeError) as e:
            with _CPU_LOCK:
                if _CPU is pool:
                    _CPU = None
                    log("w", f"worker processes failed ({type(e).__name__}: {e}); "
                             "the arithmetic runs in the threads from now on")
            pool.shutdown(wait=False, cancel_futures=True)
    return fn(*args)


def temp_path(kind, ext):
    """A /tmp file name private to this process and thread."""
    return f"/tmp/vrq_{kind}_{os.getpid()}_{threading.get_ident()}.{ext}"


# ------------------------------------------------------------------ filename

_ALNUM_SPLIT = re.compile(r"[^0-9a-z]+")
_SEG_180 = re.compile(r"^180(?![\d ])")
_SEG_360 = re.compile(r"^360(?![\d ])")
_180X180 = re.compile(r"(?<!\d)180x180(?!\d)")
_AR_PHRASE = re.compile(r"pass[\s_-]?through|alpha[\s_-]?packed|packed[\s_-]?alpha")

_SEG_STEREO = {"lr": SBS, "sbs": SBS, "tb": TB, "ou": TB, "mono": MONO, "2d": MONO}
# XBVR's file scanner conventions: "mono_180" / "180_mono" (and 360), joined by
# one separator but not by a space, which a title could hold
_MONO_PAIR = re.compile(r"(?<![a-z0-9])(?:mono[_.-]?(180|360)|(180|360)[_.-]?mono)(?![a-z0-9])")
# Flat (non-VR) stereoscopic 3D. The strong words only ever describe flat 3D,
# so they count everywhere; the loose ones ("3D", "SBS", "OU") are only trusted
# outside the VR path, where they cannot mean a VR layout.
_FLAT3D_STRONG = {"hsbs": SBS, "fsbs": SBS, "lrf": SBS, "hou": TB, "tab": TB, "tbf": TB}
_FLAT3D_PAIR = re.compile(r"(?<![a-z0-9])(?:half|full)[\s_.-]?(sbs|ou|tb)(?![a-z0-9])")
_FLAT3D_LOOSE = {"sbs": SBS, "ou": TB}
_VR_WORDS = {"vr", "vr180", "vr360", "180", "360", "180x180", "fisheye", "f180", "180f",
             "3dh", "3dv"}
# words that make a file outside the VR path worth measuring; a bare "180" or
# "360" is too common in titles, the _180 / _360 segments count via "screen"
_VR_NAME_WORDS = {"vr", "vr180", "vr360", "180x180", "fisheye", "fisheye180", "f180", "180f",
                  "3dh", "3dv"}
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
        if seg == "flat":
            screen.add(FLAT)
        if seg == "rl":
            rl = True
            stereo.add(SBS)
        if _SEG_180.match(seg) and seg != "180f":
            screen.add(DOME)
        if _SEG_360.match(seg):
            screen.add(SPHERE)
    if _180X180.search(base):
        screen.add(DOME)

    for w, value in _WORD_STEREO.items():
        if w in words:
            stereo.add(value)
    for a, b in _MONO_PAIR.findall(base):
        stereo.add(MONO)
        screen.add(DOME if (a or b) == "180" else SPHERE)
    # "f180" / "180f": a 180 degree fisheye, lens left open (XBVR's convention)
    if words & {"fisheye", "fisheye180", "f180", "180f"}:
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


def _window(buf, w, x, y, cw, ch):
    """Rows of a cw x ch window of a greyscale buffer, with its sum and sum of
    squares."""
    rows = [buf[(y + j) * w + x:(y + j) * w + x + cw] for j in range(ch)]
    return (rows, sum(sum(r) for r in rows),
            sum(sum(map(operator.mul, r, r)) for r in rows))


def _wcorr(a, b):
    """Correlation between two equally sized windows from _window()."""
    ra, sa, qa = a
    rb, sb, qb = b
    n = len(ra) * len(ra[0])
    cross = sum(sum(map(operator.mul, x, y)) for x, y in zip(ra, rb))
    va, vb = qa - sa * sa / n, qb - sb * sb / n
    if va <= 0 or vb <= 0:
        return 0.0
    return (cross - sa * sb / n) / math.sqrt(va * vb)


def stereo_tiles(buf, w, ax, bx, y0, y1, ew, eh):
    """Best correlation of each textured tile of one eye with the other eye.

    The eyes of a stereo pair are not the same picture: everything is shifted
    sideways by its parallax, and a performer close to a 180 camera is shifted
    by several per cent of the eye width while the room behind barely moves.
    Correlating the halves pixel for pixel therefore reads close-ups as mono.
    Instead each tile of the first eye (at ax, y0) is matched against the
    second eye (at bx, y1) over horizontal shifts of up to STEREO_SHIFT of the
    eye width, and the best match counts. Parallax is horizontal in both
    layouts, so top/bottom pairs are searched sideways too. Tiles without
    texture (black borders, a fade) match anything and are left out.
    """
    t = STEREO_TILE
    maxs = max(1, int(round(STEREO_SHIFT * ew)))
    min_var = STEREO_TILE_MIN_STD ** 2 * t * t
    out = []
    for ty in range(0, eh - t + 1, t):
        for tx in range(0, ew - t + 1, t):
            a = _window(buf, w, ax + tx, y0 + ty, t, t)
            if a[2] - a[1] * a[1] / (t * t) < min_var:
                continue
            lo, hi = max(-maxs, -tx), min(maxs, ew - t - tx)
            memo = {}

            def at(s):
                if s not in memo:
                    memo[s] = _wcorr(a, _window(buf, w, bx + tx + s, y1 + ty, t, t))
                return memo[s]
            # coarse pass every other pixel, then the neighbours of the best
            best = max(range(lo, hi + 1, 2), key=at)
            best = max((s for s in (best - 1, best, best + 1) if lo <= s <= hi), key=at)
            out.append(at(best))
    return out


def wrap_ratio(buf, w, y0, h):
    """How well the right edge of a picture continues into its left edge.

    In a 360 equirect the last column and the first are neighbours on the
    sphere, so they differ about as little as any two adjacent columns. In
    anything else (one eye of a 180 pair, the outer edges of a side-by-side
    frame, flat video) they are unrelated. Returns the mean difference across
    the seam divided by the typical difference between columns half the
    picture apart: near 0 for a 360, near 1 for unrelated edges. None when the
    edges carry no texture (black or uniform edges match trivially).
    """
    def col(x):
        return [buf[(y0 + y) * w + x] for y in range(h)]

    def diff(a, b):
        return sum(abs(p - q) for p, q in zip(a, b)) / len(a)

    def std(c):
        m = sum(c) / len(c)
        return math.sqrt(sum((p - m) ** 2 for p in c) / len(c))

    first, last = col(0), col(w - 1)
    if min(std(first), std(last)) < WRAP_MIN_STD:
        return None
    step = max(1, w // 32)
    far = sorted(diff(col(x), col((x + w // 2) % w)) for x in range(0, w, step))
    far = _percentile(far, 0.5)
    if far < WRAP_MIN_FAR:
        return None
    return diff(last, first) / far


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
    lr_tiles = stereo_tiles(buf, tw, 0, hw, 0, 0, hw, th)
    if len(lr_tiles) < MIN_TEXTURED * (hw // STEREO_TILE) * (th // STEREO_TILE):
        return None
    tb_tiles = stereo_tiles(buf, tw, 0, 0, 0, hh, tw, hh)
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
    m.update({"lr_tiles": lr_tiles, "tb_tiles": tb_tiles,
              "lr": _percentile(sorted(lr_tiles), 0.5),
              "tb": _percentile(sorted(tb_tiles), 0.5),
              # a 360 mono frame wraps as a whole, each eye of a 360 TB one
              "wrap": wrap_ratio(buf, tw, 0, th),
              "wrap_tb": wrap_ratio(buf, tw, 0, hh),
              "alpha_lower": _bimodal_lower(buf, tw, th),
              "matte_red": matte["red"], "matte_black": matte["black"],
              "matte": has_matte})
    return m


def combine_frames(frames):
    """Median of each metric over the frames; the matte must be in every one
    (matte_any records whether at least one frame showed it)."""
    if not frames:
        return None
    out = {}
    for k in ("blk_out", "blk_in", "bbox", "alpha_lower", "matte_red", "matte_black"):
        vals = sorted(f[k] for f in frames)
        out[k] = round(_percentile(vals, 0.5), 4)
    for k in ("lr", "tb"):
        # the textured tiles of all frames are pooled, so a fade or a title
        # card (no textured tiles) does not drag a stereo pair down
        tiles = [t for f in frames for t in f.get(k + "_tiles", ())]
        vals = sorted(tiles) if tiles else sorted(f[k] for f in frames)
        out[k] = round(_percentile(vals, 0.5), 4)
    for k in ("wrap", "wrap_tb"):
        # every frame that can tell must show the seam continuing
        vals = [f[k] for f in frames if f.get(k) is not None]
        out[k] = round(max(vals), 4) if vals else None
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


def grab_with_crop(cfg, path, ts, tw, th, box):
    """grab() plus a native-resolution grey crop (x, y, w, h) of the same
    decoded frame: (yuv, crop), either of them None when it failed. One decode
    serves both (the decode is the expensive part)."""
    yuv, crop, _ = grab_multi(cfg, path, ts, tw, th, box=box)
    return yuv, crop


def grab_multi(cfg, path, ts, tw, th, box=None, size=None):
    """grab() plus, from the same decoded frame, a native-resolution grey
    crop box (x, y, w, h) and a grey copy scaled to size (w, h) with area
    averaging (what the stereo alignment measures): (yuv, crop, scaled),
    each None when it failed or was not asked for. The extra outputs go
    through temporary files because raw outputs cannot share stdout."""
    extra = []
    if box:
        x, y, cw, ch = box
        extra.append(("c", f"crop={cw}:{ch}:{x}:{y}", cw * ch, temp_path("crop", "raw")))
    if size:
        extra.append(("g", f"scale={size[0]}:{size[1]}:flags=area,format=gray",
                      size[0] * size[1], temp_path("stereo", "raw")))
    if not extra:
        return grab(cfg, path, ts, tw, th), None, None
    n = len(extra) + 1
    graph = (f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n)) +
             f";[s0]scale={tw}:{th}:out_range=pc[t]" +
             "".join(f";[s{i + 1}]{flt}[{lab}]" for i, (lab, flt, _, _) in enumerate(extra)))
    cmd = [cfg["ffmpegPath"], "-nostdin", "-v", "error", "-y", "-skip_frame", "nokey",
           "-ss", f"{ts:.3f}", "-i", path, "-filter_complex", graph,
           "-map", "[t]", "-frames:v", "1", "-pix_fmt", "yuv444p", "-f", "rawvideo", "-"]
    for lab, _, _, tmp in extra:
        cmd += ["-map", f"[{lab}]", "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", tmp]
    got = {}
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
        for lab, _, _, tmp in extra:
            if os.path.exists(tmp):
                with open(tmp, "rb") as f:
                    got[lab] = f.read()
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    finally:
        for _, _, _, tmp in extra:
            if os.path.exists(tmp):
                os.remove(tmp)
    ok = p.returncode == 0
    yuv = p.stdout[:tw * th * 3] if ok and len(p.stdout) >= tw * th * 3 else None
    out = {}
    for lab, _, need, _ in extra:
        data = got.get(lab)
        out[lab] = data[:need] if ok and data is not None and len(data) >= need else None
    return yuv, out.get("c"), out.get("g")


def grab_grey(cfg, path, ts, size):
    """One frame as grey bytes scaled to size (w, h) with area averaging, or
    None: what the stereo alignment measures."""
    w, h = size
    cmd = [cfg["ffmpegPath"], "-nostdin", "-v", "error", "-skip_frame", "nokey",
           "-ss", f"{ts:.3f}", "-i", path, "-frames:v", "1",
           "-vf", f"scale={w}:{h}:flags=area,format=gray", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0 or len(p.stdout) < w * h:
        return None
    return p.stdout[:w * h]


def grab_crop(cfg, path, ts, box):
    """Only the native-resolution grey crop (x, y, w, h) of one frame, or
    None: what the detail task needs, without the analysis thumbnail."""
    x, y, cw, ch = box
    cmd = [cfg["ffmpegPath"], "-nostdin", "-v", "error", "-skip_frame", "nokey",
           "-ss", f"{ts:.3f}", "-i", path, "-frames:v", "1",
           "-vf", f"crop={cw}:{ch}:{x}:{y}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0 or len(p.stdout) < cw * ch:
        return None
    return p.stdout[:cw * ch]


def measure_detail(cfg, f):
    """{"verdict": detail_verdict() or None} of a file from the same two
    frames probe() uses, or {} when it cannot be read (nothing decoded)."""
    w, h, path = f.get("width"), f.get("height"), f.get("path")
    if not w or not h or not path or not os.path.exists(path):
        return {}
    box = detail_box(w, h)
    if box is None:
        return {"verdict": None}
    blocks, read = [], False
    for frac in (0.4, 0.6):
        crop = grab_crop(cfg, path, (f.get("duration") or 600) * frac, box)
        if crop:
            read = True
            blocks.append(crunch(detail_blocks, crop, box[2], box[3]))
    return {"verdict": detail_verdict(blocks, w)} if read else {}


def yuv_metrics(yuv, n, tw, th, wide):
    """frame_metrics() of one grabbed thumbnail (one crunch() call)."""
    return frame_metrics(yuv_to_rgb(yuv, n), tw, th, wide, grey=yuv[:n])


def probe(cfg, path, w, h, duration, detail=False, stereo=None):
    """Median metrics over two frames. With detail, the same two decodes also
    yield a native crop of the eye centre each, and res["detail"] holds
    detail_verdict() of them (None: not measurable). stereo: a dict that
    receives the same two frames scaled for the stereo alignment, as
    {"size": (w, h), "layout": the layout assumed for that size,
    "frames": {fraction of the duration: grey bytes or None}}."""
    tw, th = thumb_size(w, h)
    wide = w / h > 1.5
    box = detail_box(w, h) if detail else None
    size = None
    if stereo is not None:
        layout = voffset_layout(w, h)
        size = voffset_size(w, h, layout)
        stereo.update({"size": size, "layout": layout, "frames": {}})
    frames, blocks = [], []
    n = tw * th
    for frac in (0.4, 0.6):
        ts = (duration or 600) * frac
        if size:
            yuv, crop, big = grab_multi(cfg, path, ts, tw, th, box, size)
            stereo["frames"][frac] = big
        elif box:
            yuv, crop = grab_with_crop(cfg, path, ts, tw, th, box)
        else:
            yuv, crop = grab(cfg, path, ts, tw, th), None
        if box and crop:
            blocks.append(crunch(detail_blocks, crop, box[2], box[3]))
        if yuv is None:
            continue
        m = crunch(yuv_metrics, yuv, n, tw, th, wide)
        if m:
            frames.append(m)
    res = combine_frames(frames)
    if res is not None and box:
        res["detail"] = detail_verdict(blocks, w)
    return res


# ------------------------------------------------------------- honest resolution
#
# A file can have 8K pixels without 8K detail: an upscale of a smaller master,
# or a bitrate too low to keep the finest detail. Such a scene keeps its tier
# tag and also gets LOW_DETAIL. The pixels are judged on a native-resolution
# crop from the centre of the left (or only, or top) eye, where a VR lens is
# sharpest: the power spectrum along its rows and columns says how much of the
# picture's gradient energy sits above half the Nyquist frequency. Genuine
# footage at the file's resolution keeps a real share there; an upscale from
# half the size has next to none, whatever the pixel count claims.
# See README "Honest resolution" for the calibration.

LOW_DETAIL = "Low Detail"
DETAIL_CROP = 1024          # side of the native crop, in file pixels
DETAIL_BLOCK = 512          # FFT length; the crop is cut into blocks this size
DETAIL_LINE_STEP = 2        # every other row and column of a block is enough
DETAIL_SPLIT = 0.5          # "high" frequencies: above this fraction of Nyquist
DETAIL_TOP = 0.95           # ignored above this: a spike at Nyquist (dithering,
                            # line patterns of some encoders) is not detail
DETAIL_MIN_TEXTURE = 3.0    # mean squared luma step per pixel; flatter blocks
                            # (a wall, a fade, darkness) only measure noise
DETAIL_MIN_BLOCKS = 2       # textured blocks a verdict needs (of 4 per frame)
DETAIL_EFF_SHARE = 0.05     # effective resolution: the frequency above which
                            # this share of the energy lies
DETAIL_MIN_RATIO = 0.08     # below this share (in the sharpest frame) the
                            # file lacks the detail of its resolution
LOW_DETAIL_BITS = 0.8       # bits per pixel and second: 26.8 Mbit/s at
                            # 8192x4096, 23.6 at 7680x3840, 20.7 at 7200x3600
DETAIL_STARVED_RATIO = 0.12 # a bitrate below LOW_DETAIL_BITS only counts
                            # when the ratio is below this too: a clearly
                            # sharp file keeps its detail whatever its bitrate

_FFT_PLANS = {}


def _fft_plan(n):
    """Bit-reversal order, per-stage twiddles and a Hann window for length n."""
    if n not in _FFT_PLANS:
        bits = n.bit_length() - 1
        rev = [int(format(i, f"0{bits}b")[::-1], 2) for i in range(n)]
        stages, h = [], 1
        while h < n:
            stages.append((h, [cmath.exp(-1j * math.pi * k / h) for k in range(h)]))
            h *= 2
        win = [0.5 - 0.5 * math.cos(2 * math.pi * (i + 0.5) / n) for i in range(n)]
        _FFT_PLANS[n] = (rev, stages, win)
    return _FFT_PLANS[n]


def fft(values):
    """Iterative radix-2 FFT of a list of complex numbers (length a power of 2)."""
    rev, stages, _ = _fft_plan(len(values))
    a = [values[i] for i in rev]
    n = len(a)
    for h, tw in stages:
        for s in range(0, n, 2 * h):
            for k in range(h):
                u, v = a[s + k], a[s + k + h] * tw[k]
                a[s + k], a[s + k + h] = u + v, u - v
    return a


def gradient_spectrum(lines, n):
    """Summed power spectrum of equally long lines of pixels, weighted by the
    response of a first difference (the spectrum of the gradient), for
    frequency bins 0 .. n/2 (Nyquist). Each line has its mean removed and a
    Hann window applied; two real lines share one complex FFT."""
    _, _, win = _fft_plan(n)
    half = n // 2
    p = [0.0] * (half + 1)
    for i in range(0, len(lines) - 1, 2):
        a, b = lines[i], lines[i + 1]
        ma, mb = sum(a) / n, sum(b) / n
        z = fft([complex((x - ma) * w, (y - mb) * w) for x, y, w in zip(a, b, win)])
        for k in range(1, half + 1):
            p[k] += (abs(z[k]) ** 2 + abs(z[-k]) ** 2) / 2
    return [pk * (2 * math.sin(math.pi * k / n)) ** 2 for k, pk in enumerate(p)]


def detail_box(w, h):
    """(x, y, side, side) of the native crop: centred in the left eye of a wide
    frame, in the top half of any other (the upper eye of a top/bottom pair,
    well inside the picture of a mono one). None when the eye is too small."""
    if w / h > 1.5:
        ew, eh, cx, cy = w // 2, h, w // 4, h // 2
    else:
        ew, eh, cx, cy = w, h // 2, w // 2, h // 4
    side = min(DETAIL_CROP, ew, eh) // DETAIL_BLOCK * DETAIL_BLOCK
    if side < DETAIL_BLOCK:
        return None
    x = min(max(0, cx - side // 2), w - side)
    y = min(max(0, cy - side // 2), h - side)
    return x, y, side, side


def detail_blocks(buf, cw, ch):
    """The normalised gradient spectrum of every textured block of a grey
    crop (cw x ch bytes). Flat or dark blocks are left out: what they hold
    above half Nyquist is noise, not detail."""
    n, step = DETAIL_BLOCK, DETAIL_LINE_STEP
    out = []
    for by in range(0, ch - n + 1, n):
        for bx in range(0, cw - n + 1, n):
            rows = [buf[(by + y) * cw + bx:(by + y) * cw + bx + n] for y in range(0, n, step)]
            mean = sum(sum(r) for r in rows) / (len(rows) * n)
            if not 20 <= mean <= 235:
                continue
            cols = [buf[by * cw + bx + x:(by + n) * cw:cw] for x in range(0, n, step)]
            g = gradient_spectrum(rows, n)
            gc = gradient_spectrum(cols, n)
            g = [a + b for a, b in zip(g, gc)]
            total = sum(g)
            # Parseval with the Hann window (mean square 0.375): the mean
            # squared step between neighbouring pixels
            texture = total / (0.375 * n * n * (len(rows) + len(cols)) / 2)
            if texture < DETAIL_MIN_TEXTURE:
                continue
            out.append([v / total for v in g])
    return out


def frame_detail(blocks, width):
    """{ratio, eff, blocks} of one frame's textured blocks (detail_blocks()),
    or None when it has none.

    ratio: share of the gradient energy between DETAIL_SPLIT and DETAIL_TOP of
    Nyquist, of all of it up to DETAIL_TOP, averaged over the blocks.
    eff: effective resolution, the file width times the largest fraction of
    Nyquist below which all but DETAIL_EFF_SHARE of that energy lies (the
    largest downscale that removes less than that share).
    """
    if not blocks:
        return None
    half = len(blocks[0]) - 1
    pooled = [sum(b[k] for b in blocks) / len(blocks) for k in range(half + 1)]
    top = int(DETAIL_TOP * half)
    split = int(DETAIL_SPLIT * half)
    total = sum(pooled[1:top + 1])
    if total <= 0:
        return None
    ratio = sum(pooled[split + 1:top + 1]) / total
    acc, cut = 0.0, top
    while cut > 1 and acc + pooled[cut] < DETAIL_EFF_SHARE * total:
        acc += pooled[cut]
        cut -= 1
    return {"ratio": round(ratio, 4), "eff": int(round(width * cut / half)),
            "blocks": len(blocks)}


def detail_verdict(frames, width):
    """The detail of a file: frame_detail() of its sharpest frame, given the
    textured blocks of each frame; None (unknown) when the frames hold fewer
    than DETAIL_MIN_BLOCKS textured blocks in all. The sharpest frame counts
    because a soft frame proves little (focus on the far wall, motion blur),
    while an upscale has no sharp frame at all; "blocks" is the total."""
    if sum(len(b) for b in frames) < DETAIL_MIN_BLOCKS:
        return None
    measured = [d for d in (frame_detail(b, width) for b in frames) if d]
    best = dict(max(measured, key=lambda d: d["ratio"]))
    best["blocks"] = sum(len(b) for b in frames)
    return best


# ------------------------------------------------------------ stereo alignment
#
# The two eyes of a stereo pair should differ only sideways (the parallax of
# the horizontal baseline). Some releases have one eye a few pixels higher
# than the other, which is uncomfortable to watch and which a player can undo
# by pitching one eye. Textured tiles of the left eye are found in the right
# eye (horizontal and vertical shift, normalised cross-correlation), and the
# vertical shifts are fitted with
#
#     dy = c0 + c1 * xn + g * rho * dx + h * rho
#
# where rho * dx is the natural vertical parallax a near object away from the
# centre lines has in an equirect or fisheye picture (see parallax_ratio()).
# c0 is the constant vertical offset between the eyes: the misalignment. dy is
# the right eye's position minus the left eye's, image y pointing down, so a
# positive offset means the right eye's picture sits lower. The eyes are
# scaled to VOFFSET_EYE pixels wide, where the thresholds below were
# calibrated (a survey of 213 scenes); the offset is stored in degrees.
# See README "Stereo alignment".

VOFFSET_FIELD = "vr_vertical_offset"   # the scene custom field, in degrees
VOFFSET_EYE = 1024          # eye width the frames are scaled to, in pixels
VOFFSET_TILE = 64           # tile side at that width
VOFFSET_STEP = 48           # tile spacing
VOFFSET_SEARCH_X = 60       # horizontal shift searched either way (parallax)
VOFFSET_SEARCH_Y = 16       # vertical shift searched either way
VOFFSET_COARSE = 4          # the full search runs at 1/4 of the size, then
VOFFSET_FINE = 3            # the full size refines +-3 px around its answer
VOFFSET_REGION = 0.6        # equirect: tiles in the central 60 % of the eye
VOFFSET_FISH_REGION = 0.5   # fisheye: tiles within half the disc radius
VOFFSET_MIN_STD = 10        # a tile flatter than this has nothing to match
VOFFSET_MIN_MEAN = 20       # nor has a dark one
VOFFSET_MIN_GRAD = 2.0      # mean luma step down and across: a purely
                            # vertical edge says nothing about dy
VOFFSET_MIN_NCC = 0.85      # a match below this correlation is not kept
VOFFSET_UNIQUE = 0.04       # and neither is one whose peak is not this much
                            # higher than any other shift (repetitive texture)
VOFFSET_MIN_TILES = 15      # matched tiles a frame needs
VOFFSET_MAX_SE = 0.35       # standard error of c0 a frame may have, in px
VOFFSET_MIN_FRAMES = 3      # frames with a valid fit a verdict needs
VOFFSET_AGREE = 1.0         # frames within this many px give their mean
VOFFSET_SMALL_DEG = 0.25    # below this in every frame the scene is aligned
                            # (stash-vr corrects from here on): the mean too
VOFFSET_MAX_DEG = 2.0       # larger is a layout or projection error, not a
                            # misaligned rig
VOFFSET_NEAR_MAX = 2.0      # extra dy of the nearest tiles over the farthest
                            # (px, median over the frames) beyond which the
                            # cameras sit at different heights: no single
                            # shift suits both the subject and the room then
VOFFSET_FRACS = (0.4, 0.6, 0.2, 0.8, 0.3, 0.7)  # keyframes (fraction of the
                            # duration) in the order decoded: the first two
                            # are the projection's, the next two spread the
                            # sample over the scene (some scenes change their
                            # offset part way, a re-rigged camera), the last
                            # two stand in for frames that did not count
VOFFSET_PRIMARY = 4         # frames always decoded (unless they conflict)
LENS_FOV = {RF52: 190.0, MKX200: 200.0, MKX220: 220.0, VRCA220: 220.0}
FISHEYE_FOV = 190.0         # a fisheye without a lens tag


def voffset_size(w, h, stereo):
    """(width, height) a w x h frame is scaled to so that each eye is
    VOFFSET_EYE wide, for a side-by-side or top/bottom pair; None otherwise."""
    if stereo not in (SBS, TB) or not w or not h:
        return None
    fw = 2 * VOFFSET_EYE if stereo == SBS else VOFFSET_EYE
    fh = int(round(h * fw / w))
    fh += fh % 2
    return fw, fh


def voffset_layout(w, h):
    """The layout a frame is decoded for before the scene is classified: side
    by side for a wide frame, top/bottom otherwise. A scene that ends up with
    the other layout decodes its frames again."""
    return SBS if w / h > 1.5 else TB


def split_eyes(buf, fw, fh, stereo, rl=False):
    """(left rows, right rows, eye width, eye height) of a grey frame, the
    rows as bytes. rl: the right eye comes first in a side-by-side frame."""
    if stereo == SBS:
        ew, eh = fw // 2, fh
        a = [buf[y * fw:y * fw + ew] for y in range(eh)]
        b = [buf[y * fw + ew:y * fw + 2 * ew] for y in range(eh)]
        if rl:
            a, b = b, a
    else:
        ew, eh = fw, fh // 2
        a = [buf[y * fw:(y + 1) * fw] for y in range(eh)]
        b = [buf[(eh + y) * fw:(eh + y + 1) * fw] for y in range(eh)]
    return a, b, ew, eh


def shrink(rows, k):
    """Rows of the k x k box average of a grey picture given as rows."""
    out = []
    for y in range(0, len(rows) - k + 1, k):
        acc = rows[y]
        for j in range(1, k):
            acc = list(map(operator.add, acc, rows[y + j]))
        cols = acc[0::k]
        for i in range(1, k):
            cols = list(map(operator.add, cols, acc[i::k]))
        n2, half = k * k, k * k // 2
        out.append(bytes((c + half) // n2 for c in cols))
    return out


def integral(rows):
    """Summed-area tables (sum, sum of squares) of a picture given as rows:
    entry [y][x] holds the total above row y and left of column x."""
    w = len(rows[0])
    s, q = [array.array("q", bytes(8 * (w + 1)))], [array.array("q", bytes(8 * (w + 1)))]
    for r in rows:
        s.append(array.array("q", map(operator.add, s[-1],
                                      itertools.accumulate(r, initial=0))))
        q.append(array.array("q", map(operator.add, q[-1],
                                      itertools.accumulate(map(operator.mul, r, r), initial=0))))
    return s, q


def _box(t, x, y, w, h):
    return t[y + h][x + w] - t[y][x + w] - t[y + h][x] + t[y][x]


def xcorr(win, tile):
    """Cross-correlation of a tile with every position inside a window, both
    given as rows of bytes: (coefficients, base), where the sum of products
    with the tile's top-left corner at (dx, dy) of the window is
    coefficients[base + dy * window width + dx].

    One big-integer multiplication does it: each picture becomes a number
    with one 32-bit digit per pixel (the tile's reversed and laid out at the
    window's row pitch), and the digits of the product are the correlation
    sums. A sum stays below 2^32 for tiles up to 64 x 64, so no digit carries
    into the next, and CPython multiplies big integers far faster than any
    Python loop could."""
    ww, tw = len(win[0]), len(tile[0])
    flat = b"".join(win)
    wd = bytearray(4 * len(flat))
    wd[0::4] = flat
    m = (len(tile) - 1) * ww + tw
    t = bytearray(m)
    for j, r in enumerate(tile):
        t[j * ww:j * ww + tw] = r
    t.reverse()
    td = bytearray(4 * m)
    td[0::4] = t
    prod = int.from_bytes(wd, "little") * int.from_bytes(td, "little")
    n = len(flat) + m - 1
    out = array.array("I")
    out.frombytes(prod.to_bytes(4 * n, "little"))
    if sys.byteorder == "big":
        out.byteswap()
    return out, m - 1


def ncc_grid(win, tile, tables, wx, wy):
    """Normalised cross-correlation (as OpenCV's TM_CCOEFF_NORMED) of a tile
    at every position of a window whose top-left corner is (wx, wy) in the
    picture that tables (integral() of it) describe: rows of values."""
    th, tw = len(tile), len(tile[0])
    wh, ww = len(win), len(win[0])
    n = th * tw
    st = sum(sum(r) for r in tile)
    vt = sum(sum(map(operator.mul, r, r)) for r in tile) - st * st / n
    cross, base = xcorr(win, tile)
    s, q = tables
    grid = []
    for dy in range(wh - th + 1):
        row = []
        for dx in range(ww - tw + 1):
            sw = _box(s, wx + dx, wy + dy, tw, th)
            vw = _box(q, wx + dx, wy + dy, tw, th) - sw * sw / n
            den = vt * vw
            row.append((cross[base + dy * ww + dx] - st * sw / n) / math.sqrt(den)
                       if den > 0 else 0.0)
        grid.append(row)
    return grid


def _peak(grid):
    """(value, x, y) of the largest entry of a grid of rows."""
    return max((v, x, y) for y, row in enumerate(grid) for x, v in enumerate(row))


def _parabola(a, b, c):
    """Sub-pixel offset of the vertex of a parabola through (-1, a), (0, b), (1, c)."""
    den = a - 2 * b + c
    return 0.0 if den == 0 else 0.5 * (a - c) / den


def _textured(tile):
    """Whether a tile has enough texture in both directions to be matched."""
    n = len(tile) * len(tile[0])
    s = sum(sum(r) for r in tile)
    q = sum(sum(map(operator.mul, r, r)) for r in tile)
    mean = s / n
    if mean < VOFFSET_MIN_MEAN or q / n - mean * mean < VOFFSET_MIN_STD ** 2:
        return False
    gy = sum(sum(map(abs, map(operator.sub, a, b))) for a, b in zip(tile, tile[1:]))
    if gy < VOFFSET_MIN_GRAD * (len(tile) - 1) * len(tile[0]):
        return False
    gx = sum(sum(map(abs, map(operator.sub, r[1:], r))) for r in tile)
    return gx >= VOFFSET_MIN_GRAD * len(tile) * (len(tile[0]) - 1)


# ---- projections: one eye of ew x eh pixels, as a small dict

def eye_projection(screen, lens, ew, eh):
    """The projection of one eye: a 180 equirect, or an equidistant fisheye
    disc centred in the eye, its diameter the eye's smaller side and its
    field of view that of the lens tag (FISHEYE_FOV without one). None for
    anything else."""
    if screen == DOME:
        return {"kind": "equirect", "w": ew, "h": eh, "fov": 180.0}
    if screen == FISHEYE:
        fov = LENS_FOV.get(lens, FISHEYE_FOV)
        r = min(ew, eh) / 2
        return {"kind": "fisheye", "w": ew, "h": eh, "fov": fov, "cx": (ew - 1) / 2,
                "cy": (eh - 1) / 2, "r": r, "f": r / math.radians(fov / 2)}
    return None


def deg_per_px(p):
    """Degrees per pixel along the vertical at the centre of an eye."""
    if p["kind"] == "equirect":
        return 180.0 / p["h"]
    return math.degrees(1 / p["f"])


def _dir(p, u, v):
    """Unit view direction (x right, y up, z forward) of eye pixel (u, v)."""
    if p["kind"] == "equirect":
        lam = math.radians(((u + 0.5) / p["w"] - 0.5) * p["fov"])
        phi = math.pi * (0.5 - (v + 0.5) / p["h"])
        c = math.cos(phi)
        return c * math.sin(lam), math.sin(phi), c * math.cos(lam)
    x, y = u - p["cx"], p["cy"] - v
    th = math.hypot(x, y) / p["f"]
    psi = math.atan2(y, x)
    s = math.sin(th)
    return s * math.cos(psi), s * math.sin(psi), math.cos(th)


def _pix(p, d):
    """Eye pixel (u, v) of a unit view direction."""
    x, y, z = d
    if p["kind"] == "equirect":
        lam = math.atan2(x, z)
        phi = math.asin(max(-1.0, min(1.0, y)))
        return ((lam / math.radians(p["fov"]) + 0.5) * p["w"] - 0.5,
                (0.5 - phi / math.pi) * p["h"] - 0.5)
    th = math.acos(max(-1.0, min(1.0, z)))
    psi = math.atan2(y, x)
    r = th * p["f"]
    return p["cx"] + r * math.cos(psi), p["cy"] - r * math.sin(psi)


def parallax_ratio(p, u, v, eps=1e-3):
    """dy / dx of the image displacement that a purely horizontal baseline
    produces at eye pixel (u, v): a point with horizontal disparity dx there
    has the natural vertical disparity ratio * dx."""
    d = _dir(p, u, v)
    t = (1 - d[0] * d[0], -d[0] * d[1], -d[0] * d[2])      # x axis minus its radial part
    d1 = tuple(a - eps * b for a, b in zip(d, t))
    n = math.sqrt(sum(a * a for a in d1))
    d1 = tuple(a / n for a in d1)
    u0, v0 = _pix(p, d)
    u1, v1 = _pix(p, d1)
    du = u1 - u0
    if abs(du) < 1e-9:
        du = 1e-9
    return (v1 - v0) / du


def match_tiles(left, right, p):
    """Every textured tile of the left eye found in the right eye: a list of
    (x, y, dx, dy, ncc), x and y the tile centre in the eye, dx and dy the
    shift of its match (right minus left, y down) to a tenth of a pixel or
    better. The full search (VOFFSET_SEARCH_X by VOFFSET_SEARCH_Y) runs on a
    1/VOFFSET_COARSE copy of both eyes, where the match also has to be
    unique; the full size refines it, and the correlation there decides."""
    t, k = VOFFSET_TILE, VOFFSET_COARSE
    sx, sy, rf = VOFFSET_SEARCH_X, VOFFSET_SEARCH_Y, VOFFSET_FINE
    ew, eh = len(left[0]), len(left)
    cl, cr = shrink(left, k), shrink(right, k)
    ctab, ftab = integral(cr), integral(right)
    ct, csx, csy = t // k, sx // k, sy // k
    out = []
    for y in range(sy, eh - t - sy + 1, VOFFSET_STEP):
        for x in range(sx, ew - t - sx + 1, VOFFSET_STEP):
            if not _in_region(p, x + t / 2, y + t / 2):
                continue
            tile = [r[x:x + t] for r in left[y:y + t]]
            if not _textured(tile):
                continue
            # coarse: the whole search window
            cx, cy = x // k, y // k
            ctile = [r[cx:cx + ct] for r in cl[cy:cy + ct]]
            win = [r[cx - csx:cx + ct + csx] for r in cr[cy - csy:cy + ct + csy]]
            grid = ncc_grid(win, ctile, ctab, cx - csx, cy - csy)
            c, ix, iy = _peak(grid)
            if c < VOFFSET_MIN_NCC or iy in (0, len(grid) - 1) or ix in (0, len(grid[0]) - 1):
                continue
            rest = max((v for j, row in enumerate(grid) for i, v in enumerate(row)
                        if abs(i - ix) > 1 or abs(j - iy) > 1), default=-1.0)
            if rest > c - VOFFSET_UNIQUE:
                continue
            px = round(k * (ix - csx + _parabola(grid[iy][ix - 1], c, grid[iy][ix + 1])))
            py = round(k * (iy - csy + _parabola(grid[iy - 1][ix], c, grid[iy + 1][ix])))
            # fine: +-VOFFSET_FINE around it at full size
            wx, wy = x + px - rf, y + py - rf
            if wx < 0 or wy < 0 or wx + t + 2 * rf > ew or wy + t + 2 * rf > eh:
                continue
            win = [r[wx:wx + t + 2 * rf] for r in right[wy:wy + t + 2 * rf]]
            grid = ncc_grid(win, tile, ftab, wx, wy)
            c, ix, iy = _peak(grid)
            if c < VOFFSET_MIN_NCC or iy in (0, 2 * rf) or ix in (0, 2 * rf):
                continue
            dx = px + ix - rf + _parabola(grid[iy][ix - 1], c, grid[iy][ix + 1])
            dy = py + iy - rf + _parabola(grid[iy - 1][ix], c, grid[iy + 1][ix])
            if abs(dx) >= sx or abs(dy) >= sy:
                continue
            out.append((x + t / 2, y + t / 2, dx, dy, c))
    return out


def _in_region(p, u, v):
    """Whether eye pixel (u, v) lies where tiles are matched: the central
    VOFFSET_REGION of an equirect eye, or within VOFFSET_FISH_REGION of the
    disc radius of a fisheye."""
    if p["kind"] == "equirect":
        lo, hi = 0.5 - VOFFSET_REGION / 2, 0.5 + VOFFSET_REGION / 2
        return (int(p["w"] * lo) <= u < int(p["w"] * hi) and
                int(p["h"] * lo) <= v < int(p["h"] * hi))
    return math.hypot(u - p["cx"], v - p["cy"]) < VOFFSET_FISH_REGION * p["r"]


def _solve(ata, atb):
    """Solution of the small linear system ata * x = atb and the inverse of
    ata (Gauss-Jordan with partial pivoting), or None when it is singular."""
    n = len(ata)
    m = [list(ata[i]) + [atb[i]] + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [v / d for v in m[c]]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)], [m[i][n + 1:] for i in range(n)]


def _lstsq(rows, ys):
    """Least squares: (coefficients, inverse of the normal matrix) or None."""
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    atb = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    return _solve(ata, atb)


def _median(vals):
    return _percentile(sorted(vals), 0.5)


def _robust_fit(rows, dys, k):
    """Least squares of dys over the first k columns of rows in four rounds,
    each dropping the entries further than 3 MADs (at least 0.75 px) from
    the previous fit: (coefficients, residuals, kept, inverse normal
    matrix of the kept rows) or None when fewer than 10 entries remain."""
    rows = [r[:k] for r in rows]
    keep = [True] * len(rows)
    for _ in range(4):
        sol = _lstsq([r for r, w in zip(rows, keep) if w], [d for d, w in zip(dys, keep) if w])
        if sol is None:
            return None
        coef = sol[0]
        res = [d - sum(a * b for a, b in zip(r, coef)) for r, d in zip(rows, dys)]
        kept = [e for e, w in zip(res, keep) if w]
        mid = _median(kept)
        mad = _median([abs(e - mid) for e in kept]) * 1.4826 + 0.05
        keep = [abs(e) < max(3 * mad, 0.75) for e in res]
        if sum(keep) < 10:
            return None
    sol = _lstsq([r for r, w in zip(rows, keep) if w], [d for d, w in zip(dys, keep) if w])
    if sol is None:
        return None
    return coef, res, keep, sol[1]


def fit_voffset(tiles, p):
    """Robust fit of dy = c0 + c1 * xn + g * rho * dx + h * rho over matched
    tiles (xn: x from the centre in half eye widths, rho: parallax_ratio()),
    see _robust_fit(). Returns {ok, tiles, n, c0, c1, se0, near}: tiles
    matched, n of them kept by the fit, c0 and c1 in px, se0 the standard
    error of c0.

    near: how much more dy the nearest tiles have than the farthest ones
    beyond that model, from a second fit with a term k * dx added (a rig
    whose cameras sit at different heights shifts near objects vertically in
    proportion to their parallax), times the spread of dx between its 10th
    and 90th percentile. A pure misalignment gives near 0."""
    if len(tiles) < VOFFSET_MIN_TILES:
        return {"ok": False, "n": len(tiles)}
    cx = p["w"] / 2 if p["kind"] == "equirect" else p["cx"]
    rows, dys = [], []
    for x, y, dx, dy, _ in tiles:
        rho = parallax_ratio(p, x, y)
        rows.append((1.0, (x - cx) / (p["w"] / 2), rho * dx, rho, dx))
        dys.append(dy)
    fit = _robust_fit(rows, dys, 4)
    if fit is None:
        return {"ok": False, "n": len(tiles)}
    coef, res, keep, inv = fit
    n = sum(keep)
    s2 = sum(e * e for e, w in zip(res, keep) if w) / max(1, n - 4)
    se0 = math.sqrt(max(0.0, s2 * inv[0][0]))
    near = 0.0
    ext = _robust_fit(rows, dys, 5)
    if ext is not None:
        dxs = sorted(t[2] for t, w in zip(tiles, ext[2]) if w)
        near = ext[0][4] * (_percentile(dxs, 0.1) - _percentile(dxs, 0.9))
    return {"ok": True, "tiles": len(tiles), "n": n, "c0": coef[0], "c1": coef[1], "se0": se0,
            "near": near}


def frame_voffset(buf, fw, fh, stereo, rl, screen, lens):
    """fit_voffset() of one grey frame of fw x fh (voffset_size()), or
    {"ok": False} when its projection is not measured (one crunch() call)."""
    left, right, ew, eh = split_eyes(buf, fw, fh, stereo, rl)
    p = eye_projection(screen, lens, ew, eh)
    if p is None:
        return {"ok": False, "n": 0}
    return fit_voffset(match_tiles(left, right, p), p)


def frame_counts(fit):
    """Whether one frame's fit is good enough to count: VOFFSET_MIN_TILES
    matched tiles and c0 known to VOFFSET_MAX_SE."""
    return bool(fit.get("ok")) and fit["tiles"] >= VOFFSET_MIN_TILES and \
        fit["se0"] <= VOFFSET_MAX_SE


def voffset_conflict(px, dpp):
    """Whether frame offsets px (pixels) can give no verdict whatever else is
    decoded: some point up and some down while one of them is at least
    VOFFSET_SMALL_DEG. Such a scene changes its offset part way, and no
    single correction suits all of it."""
    return any(v > 0 for v in px) and any(v < 0 for v in px) and \
        any(abs(v) * dpp >= VOFFSET_SMALL_DEG for v in px)


def voffset_verdict(fits, dpp, min_frames=VOFFSET_MIN_FRAMES):
    """(degrees or None, reason) from the fits of a scene's frames, dpp the
    degrees per pixel (deg_per_px()). Known only when at least min_frames
    frames count (frame_counts()), they do not conflict (voffset_conflict()),
    the median of their "near" is at most VOFFSET_NEAR_MAX and the value at
    most VOFFSET_MAX_DEG. The value is the mean when the frames lie within
    VOFFSET_AGREE px of each other or all below VOFFSET_SMALL_DEG, else (all
    one way, by different amounts) the one closest to zero: the offset the
    whole scene has at least, so the correction is never too large for any
    part of it. Rounded to hundredths of a degree, same sign as the pixel
    offset."""
    good = [f for f in fits if frame_counts(f)]
    px = [f["c0"] for f in good]
    desc = "/".join(f"{v:+.2f}" for v in px) or "none"
    if voffset_conflict(px, dpp):
        return None, f"frames disagree in direction (c0 {desc} px)"
    if len(good) < min_frames:
        return None, f"{len(good)} of {len(fits)} frames measurable (c0 {desc} px)"
    near = _median([f["near"] for f in good])
    if abs(near) > VOFFSET_NEAR_MAX:
        return None, (f"offset grows with nearness ({near:+.2f} px from far to near, c0 {desc} "
                      "px): the cameras sit at different heights")
    if max(px) - min(px) <= VOFFSET_AGREE or all(abs(v) * dpp < VOFFSET_SMALL_DEG for v in px):
        deg, how = sum(px) / len(px) * dpp, "mean"
    else:
        deg, how = min(px, key=abs) * dpp, "smallest"
    if abs(deg) > VOFFSET_MAX_DEG:
        return None, f"{deg:+.2f} deg is implausibly large (c0 {desc} px)"
    return round(deg, 2) + 0.0, f"{deg:+.2f} deg (c0 {desc} px, {len(good)} frames, {how})"


def alignment_input(names):
    """(stereo, screen, lens, rl) the alignment is measured with, from a
    scene's projection tag names; stereo and screen None when the scene is
    not a side-by-side or top/bottom 180 or fisheye pair."""
    stereo = SBS if SBS in names else TB if TB in names else None
    screen = DOME if DOME in names else FISHEYE if FISHEYE in names else None
    if SPHERE in names or FLAT in names or MONO in names:
        stereo = screen = None
    lens = next((t for t in LENS_TAGS if t in names), None)
    return stereo, screen, lens, RL in names


def measure_alignment(cfg, f, names, cached=None):
    """{"deg": vertical offset in degrees or None, "why": text} of a scene's
    primary file f, whose projection tags are names; {} when nothing is known
    (the projection is unresolved, the file is missing or no frame decoded),
    in which case a stored value is left as it is.

    The frames are keyframes at VOFFSET_FRACS of the duration: the first
    VOFFSET_PRIMARY always, the others one at a time only while the count of
    frames that count is one short of VOFFSET_MIN_FRAMES; decoding stops as
    soon as the counting frames conflict (voffset_conflict()). cached: probe()'s stereo dict,
    whose frames are used when their size fits the layout.
    """
    if UNRESOLVED in names:
        return {}
    stereo, screen, lens, rl = alignment_input(names)
    if not stereo or not screen:
        return {"deg": None, "why": "not a stereo 180 or fisheye pair"}
    w, h, path = f.get("width"), f.get("height"), f.get("path")
    if not w or not h or not path or not os.path.exists(path):
        return {}
    size = voffset_size(w, h, stereo)
    ew, eh = (size[0] // 2, size[1]) if stereo == SBS else (size[0], size[1] // 2)
    dpp = deg_per_px(eye_projection(screen, lens, ew, eh))
    have = (cached or {}).get("frames") or {}
    if (cached or {}).get("size") != size:
        have = {}
    fits, decoded = [], 0
    for i, frac in enumerate(VOFFSET_FRACS):
        good = [x["c0"] for x in fits if frame_counts(x)]
        if voffset_conflict(good, dpp):
            break
        if i >= VOFFSET_PRIMARY and len(good) != VOFFSET_MIN_FRAMES - 1:
            break               # a stand-in only for a count one frame short
        buf = have[frac] if frac in have else grab_grey(cfg, path, (f.get("duration") or 600) * frac,
                                                          size)
        if buf is None:
            continue
        decoded += 1
        fits.append(crunch(frame_voffset, buf, size[0], size[1], stereo, rl, screen, lens))
    if not decoded:
        return {}
    deg, why = voffset_verdict(fits, dpp)
    return {"deg": deg, "why": why}


def voffset_update(fields, deg):
    """The custom_fields input that brings a scene's VOFFSET_FIELD to deg
    (None: removed), or None when it already is. fields: the scene's custom
    fields; no other field is ever named."""
    fields = fields or {}
    if deg is None:
        return {"remove": [VOFFSET_FIELD]} if VOFFSET_FIELD in fields else None
    cur = fields.get(VOFFSET_FIELD)
    try:
        if cur is not None and not isinstance(cur, bool) and abs(float(cur) - deg) < 0.005:
            return None
    except (TypeError, ValueError):
        pass
    return {"partial": {VOFFSET_FIELD: deg}}


def voffset_text(cf, why):
    """Log text of a custom field change."""
    if "remove" in cf:
        return f"{VOFFSET_FIELD} removed ({why})"
    return f"{VOFFSET_FIELD} {cf['partial'][VOFFSET_FIELD]:+.2f} ({why})"


def bits_per_pixel(f):
    """Bits per pixel and second of a Stash file record, or None."""
    w, h, br = f.get("width") or 0, f.get("height") or 0, f.get("bit_rate") or 0
    return br / (w * h) if w and h and br else None


def low_detail(f, detail):
    """(True/False/None, reason) for a file that earned a tier tag. True when
    the pixels lack the detail (ratio below DETAIL_MIN_RATIO), or the bitrate
    is below LOW_DETAIL_BITS and the ratio below DETAIL_STARVED_RATIO; False
    when neither holds and the pixels were measured; None when the pixels are
    unknown (never tagged on unknown, whatever the bitrate)."""
    bpp = bits_per_pixel(f)
    why = []
    if detail:
        why.append(f"detail {detail['ratio']:.3f} (effective width ~{detail['eff']}, "
                   f"{detail['blocks']} blocks)")
    else:
        why.append("detail unknown")
    if bpp is not None:
        why.append(f"{bpp:.2f} bit/px/s")
    if not detail:
        return None, ", ".join(why)
    starved = bpp is not None and bpp < LOW_DETAIL_BITS and \
        detail["ratio"] < DETAIL_STARVED_RATIO
    if starved or detail["ratio"] < DETAIL_MIN_RATIO:
        return True, ", ".join(why)
    return False, ", ".join(why)


# ------------------------------------------------------------- classification

def eye_aspect(w, h, stereo):
    if stereo == SBS:
        return (w / 2) / h
    if stereo == TB:
        return w / (h / 2)
    return w / h


def screen_from_eye(a):
    """One eye's shape names the projection: 180 covers a square, 360 a 2:1
    equirect, 16:9 (and 16:10, and DCI 4K's 1.9) is ordinary flat video. A
    full side-by-side flat 3D file (3840x1080, 7680x2160) is two 16:9 eyes, so
    its whole-frame aspect of 3.2-3.8 reads as FLAT once the SBS split is
    accepted. A 2:1 shape alone does not make a 360: see classify()."""
    if abs(a - 1.0) < 0.28:
        return DOME
    if 1.6 <= a < 1.92:
        return FLAT
    if 1.92 <= a < 2.13:
        return SPHERE
    return None


def wraps(ratio):
    return ratio is not None and ratio <= WRAP_MAX


def classify(w, h, res):
    """(screen, stereo, why) from frame measurements alone. screen is None when
    the shape is not recognised or the frame looks like a packed matte."""
    if not res:
        return None, None, "no probe"
    why = []
    wrap, wrap_tb = res.get("wrap"), res.get("wrap_tb")
    tb_eye = eye_aspect(w, h, TB)
    # a 2:1 frame holding a top/bottom 360 pair squeezes each eye to 4:1
    # (early 360 releases); only the seam of each eye can vouch for that
    squeezed_tb = abs(tb_eye - 4.0) < 0.26 and wraps(wrap_tb)
    # stereo: accept a layout only if it implies a plausible eye shape; when
    # both qualify, the better match wins (a 360 TB frame's halves also match
    # side by side somewhat: the room runs level all the way round)
    sbs_ok = res["lr"] >= SBS_MIN and screen_from_eye(eye_aspect(w, h, SBS))
    tb_ok = res["tb"] >= TB_MIN and (screen_from_eye(tb_eye) or squeezed_tb)
    if sbs_ok and not (tb_ok and res["tb"] > res["lr"]):
        stereo = SBS
        why.append(f"lr={res['lr']:.2f}")
    elif tb_ok:
        if res.get("alpha_lower", 0.0) > ALPHA_BIMODAL:
            # lower half is a binary matte, not an eye: packed alpha, not stereo
            return None, None, f"packed alpha? lower half {res['alpha_lower']:.2f} binary"
        stereo = TB
        why.append(f"tb={res['tb']:.2f}")
        if squeezed_tb:
            why.append(f"squeezed 360 TB, wrap={wrap_tb:.2f}")
            return SPHERE, TB, ", ".join(why)
    else:
        stereo = MONO
        why.append(f"lr={res['lr']:.2f} tb={res['tb']:.2f}")

    ea = eye_aspect(w, h, stereo)
    by_shape = screen_from_eye(ea)
    if stereo == MONO and by_shape == SPHERE:
        # A mono 2:1 frame is a 360 only if its seam closes. Without that the
        # likelier reading is a 180 pair whose halves match poorly (a close-up
        # with a lot of parallax); halves with nothing in common at all are
        # neither, most often flat video in a 2:1 frame.
        wtxt = "none" if wrap is None else f"{wrap:.2f}"
        if wraps(wrap):
            why.append(f"360 wrap={wtxt}")
        elif res["lr"] >= STEREO_NONE:
            stereo = SBS
            ea = eye_aspect(w, h, SBS)
            by_shape = screen_from_eye(ea)
            why.append(f"no 360 seam (wrap={wtxt}), read as a 180 pair")
        else:
            why.append(f"2:1 but neither stereo nor 360 (wrap={wtxt})")
            return None, None, ", ".join(why)
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


# ------------------------------------------------------------- file metadata

# ffprobe's names (libavutil/stereo3d.c, spherical.c); the Matroska StereoMode
# element also arrives as the stream tag stereo_mode
_META_STEREO = {"2d": MONO, "side by side": SBS, "side by side (quincunx subsampling)": SBS,
                "top and bottom": TB}
_MKV_STEREO = {"mono": (MONO, False), "left_right": (SBS, False), "right_left": (SBS, True),
               "top_bottom": (TB, False), "bottom_top": (TB, False)}
_META_EQUIRECT = ("equirectangular", "tiled equirectangular")
FOV_DEG_LENS = {190: RF52, 200: MKX200, 220: MKX220}


def _number(v):
    """A float from ffprobe's JSON, which prints rationals as "num/den"."""
    try:
        if isinstance(v, str) and "/" in v:
            num, den = v.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_ffprobe(data):
    """What a file's own metadata claims about its first video stream, from
    ffprobe's JSON (stream width, side data and tags). Keys, each only when
    the metadata says so: screen, stereo, rl, lens, and raw (a short summary
    for the log). None when there is nothing.

    Spherical Mapping: equirectangular is a 360 unless its left and right
    bounds crop the picture ("tiled equirectangular"), half equirectangular
    is a 180, fisheye is FISHEYE; cubemap and the other projections are not
    in this plugin's vocabulary and are left to the pixels. Stereo 3D: 2D,
    side by side, top and bottom; inverted means the right eye comes first.
    A horizontal field of view of 190, 200 or 220 degrees (Apple's spatial
    and immersive video, newer ffprobe) names the fisheye lens.
    """
    streams = (data or {}).get("streams") or []
    if not streams:
        return None
    st = streams[0]
    width = st.get("width") or 0
    out, raw = {}, []
    fov = None
    for sd in st.get("side_data_list") or []:
        kind = sd.get("side_data_type")
        if kind == "Spherical Mapping":
            proj = str(sd.get("projection") or "").lower()
            if proj in _META_EQUIRECT:
                crop = (_number(sd.get("bound_left")) or 0) + (_number(sd.get("bound_right")) or 0)
                span = 360.0 * (width - crop) / width if width else 360.0
                out["screen"] = DOME if span <= 270 else SPHERE
                raw.append(f"{proj} {span:.0f}deg")
            elif proj == "half equirectangular":
                out["screen"] = DOME
                raw.append(proj)
            elif proj == "fisheye":
                out["screen"] = FISHEYE
                raw.append(proj)
            elif proj:
                raw.append(f"{proj} (ignored)")
        elif kind == "Stereo 3D":
            kind3d = str(sd.get("type") or "").lower()
            stereo = _META_STEREO.get(kind3d)
            if stereo:
                out["stereo"] = stereo
                out["rl"] = stereo == SBS and bool(_number(sd.get("inverted")))
                raw.append(kind3d + (" inverted" if out["rl"] else ""))
            fov = _number(sd.get("horizontal_field_of_view")) or fov
    mode = str((st.get("tags") or {}).get("stereo_mode") or "").lower()
    if "stereo" not in out and mode in _MKV_STEREO:
        out["stereo"], out["rl"] = _MKV_STEREO[mode]
        out["rl"] = out["rl"] and out["stereo"] == SBS
        raw.append(f"stereo_mode {mode}")
    if fov:
        lens = FOV_DEG_LENS.get(int(round(fov)))
        raw.append(f"fov {fov:g}")
        if lens and out.get("screen") in (None, FISHEYE):
            out["screen"], out["lens"] = FISHEYE, lens
    if not out:
        return None
    out["raw"] = ", ".join(raw)
    return out


def read_metadata(cfg, path):
    """parse_ffprobe() of one ffprobe call, or None when ffprobe is missing,
    fails or finds nothing."""
    cmd = [cfg["ffprobePath"], "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height:stream_side_data:stream_tags",
           "-of", "json", path]
    try:
        p = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    try:
        return parse_ffprobe(json.loads(p.stdout or b"{}"))
    except ValueError:
        return None


def _coarse(screen):
    """Fisheye and equirect (180 or 360) are different shapes of picture."""
    return {FISHEYE: "fisheye", DOME: "equirect", SPHERE: "equirect"}.get(screen, screen)


def _eye_fits(w, h, stereo, screen):
    """Whether a projection is plausible for the eye a stereo layout leaves."""
    a = eye_aspect(w, h, stereo or MONO)
    shape = screen_from_eye(a)
    if screen in (DOME, FISHEYE):
        return shape == DOME
    if screen == SPHERE:
        return shape == SPHERE or (stereo == TB and abs(a - 4.0) < 0.26)
    return False


def vet_claim(claim, w, h, res, screen_px, stereo_px, disc_yields=False):
    """The part of an outside claim (file metadata, the SLR lookup) that the
    frame does not contradict, and what was dropped and why.

    Stereo: a claimed mono picture whose halves match as a stereo pair, or a
    claimed pair whose halves have nothing in common, is dropped, and so is a
    layout that leaves no eye of the projection the scene ends up with (mono
    on a 2:1 frame the pixels read as a 180 pair).
    Projection: the claim has to agree with the pixels in coarse shape
    (fisheye or equirect), and its 180 or 360 has to fit the eye the stereo
    layout leaves (a 2:1 side-by-side frame holds two square 180 eyes, never
    a 360). Where the pixels found no projection, the eye shape alone
    decides. A lens only comes with an accepted fisheye projection.

    disc_yields (the SLR lookup): a 180 equirect claim beats a FISHEYE verdict
    that rests on the disc test alone (no corner matte), because vignetted
    180 equirects pass that test. Never the other way round: a fisheye claim
    does not beat an equirect frame (a download can be an equirect
    conversion of a fisheye scene).
    """
    if not claim:
        return None, []
    out, dropped = {}, []
    stereo = claim.get("stereo")
    if stereo:
        if res and stereo == MONO and (res["lr"] >= SBS_MIN or res["tb"] >= TB_MIN):
            dropped.append("mono, but the halves match as a pair")
        elif res and stereo in (SBS, TB) and res["lr" if stereo == SBS else "tb"] < STEREO_NONE:
            dropped.append(f"{stereo}, but the halves have nothing in common")
        else:
            out["stereo"] = stereo
            out["rl"] = bool(claim.get("rl")) and stereo == SBS
    screen = claim.get("screen")
    if screen:
        layout = out.get("stereo") or stereo_px
        disc_only = screen_px == FISHEYE and bool(res) and not res.get("matte")
        if disc_yields and screen == DOME and disc_only:
            screen_px = None                # checked by eye shape alone below
        if screen_px and _coarse(screen) != _coarse(screen_px):
            dropped.append(f"{screen}, but the frame is {screen_px}")
        elif not _eye_fits(w, h, layout, screen):
            dropped.append(f"{screen}, but the eye of a {w}x{h} {layout or MONO} frame is not")
        else:
            out["screen"] = screen
            if "lens" in claim:
                out["lens"] = claim["lens"]
                if claim.get("lens_inferred") is not None:
                    out["lens_inferred"] = claim["lens_inferred"]
    final = out.get("screen") or screen_px
    if out.get("stereo") and final in (DOME, SPHERE, FISHEYE) and \
            not _eye_fits(w, h, out["stereo"], final):
        # a layout that leaves no eye of the projection the scene ends up with
        dropped.append(f"{out['stereo']}, but that leaves no {final} eye in a {w}x{h} frame")
        del out["stereo"], out["rl"]
    return out, dropped


def _lens_of(fov):
    value, vrca = fov
    return VRCA220 if (vrca and value == "220") else FOV_LENS.get(value)


def decide(fn, screen_px, stereo_px, fov=None, meta=None, slr=None):
    """(screen, lens, stereo, rl) from every source, in order of authority:
    file metadata and the SLR lookup (both vetted, see vet_claim()), filename
    markers, the watermark FOV, the pixels. A source that names a lens names
    a fisheye; one that settles the lens (the SLR lookup of a 180 fisheye
    names none) settles it for every source below, except that the watermark
    beats a lens SLR only inferred from its viewAngle."""
    fn_claim = {"screen": FISHEYE if fn["lens"] else fn["screen"],
                "stereo": fn["stereo"], "rl": fn["rl"]}
    if fn["lens"]:
        fn_claim["lens"] = fn["lens"]
    claims = [c for c in (meta, slr, fn_claim) if c]
    screen = next((c["screen"] for c in claims if c.get("screen")), None) or screen_px
    lens = None
    if screen == FISHEYE:
        settled = next((c for c in claims if "lens" in c), None)
        if settled is not None and not (fov and settled.get("lens_inferred")):
            lens = settled["lens"]
        elif fov:
            lens = _lens_of(fov)
    by = next((c for c in claims if c.get("stereo")), None)
    stereo = by["stereo"] if by else stereo_px
    rl = bool(by and by.get("rl")) and stereo == SBS
    return screen, lens, stereo, rl


def resolve(fn, screen_px, stereo_px, alpha_px, fov=None, meta=None, slr=None):
    """Merge file metadata, the SLR lookup, filename markers, the watermark
    FOV and the pixel verdict into the projection tags a scene should carry.
    fov is ("190"|"200"|"220", vrca); meta and slr are vetted claims from
    vet_claim(). The pixel-measured matte (alpha_px) always stands; the SLR
    lookup can add Alpha and Chroma Key but never remove the matte."""
    screen, lens, stereo, rl = decide(fn, screen_px, stereo_px, fov, meta, slr)
    want = set()
    if alpha_px or (slr and slr.get("alpha")):
        want.add(ALPHA)
    if slr and slr.get("chroma"):
        want.add(CHROMA)
    if not screen:
        want.add(UNRESOLVED)
        return want
    want.add(screen)
    if lens:
        want.add(lens)
    # FLAT already means mono 2D; MONO is only for mono VR (DOME/SPHERE + MONO)
    if stereo and not (screen == FLAT and stereo == MONO):
        want.add(stereo)
    if rl:
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


def fov_skip_reason(fn, screen_px, alpha, path, meta=None, slr=None):
    """Why reading the SLR watermark cannot help this scene, or None when it can.

    The watermark only names a fisheye lens, so there is nothing to read when
    a better source (file metadata, the SLR lookup, the filename) already
    settles the lens or the scene does not end up FISHEYE (those sources beat
    the pixels). SLR's own passthrough releases carry the watermark, other
    studios' corner-matte scenes never do, so a matte scene is only read when
    its path mentions SLR / SexLikeReal.
    """
    if meta and meta.get("lens"):
        return "lens from metadata"
    if slr and "lens" in slr and not slr.get("lens_inferred"):
        return "lens from SLR"
    if fn["lens"]:
        return "lens from filename"
    screen, _, _, _ = decide(fn, screen_px, None, None, meta, slr)
    if screen != FISHEYE:
        return "not fisheye"
    if alpha and not _SLR_RE.search(path or ""):
        return "passthrough not from SLR"
    return None


def read_fov(cfg, path, duration):
    """Read the burned-in SLR FOV. Requires two agreeing frames."""
    if not os.path.exists(cfg["tesseractPath"]):
        return None
    tmp = temp_path("fov", "png")
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


# ------------------------------------------------------------ SLR lookup

SLR_API = "https://api.sexlikereal.com/v3/scenes/"
SLR_USER_AGENT = (f"vrQualityTags/{VERSION} (Stash plugin; "
                  "+https://github.com/skeel-x/vrQualityTags)")
SLR_CACHE_FILE = "vrQualityTags.slr.json"
SLR_TTL = 90 * 86400                # a scene's answer is fetched again after this
SLR_MISS_TTL = 30 * 86400           # and "no such scene" after this
SLR_INTERVAL = 1.0                  # at most one request per second
SLR_TIMEOUT = 20
SLR_MAX_FAILURES = 3                # network failures in a row that end lookups for the run

# https://www.sexlikereal.com/scenes/<slug>-<id>, .../trans/scenes/..., or the
# short https://www.sexlikereal.com/<id>
_SLR_URL = re.compile(r"^https?://(?:www\.)?sexlikereal\.com/"
                      r"(?:(trans|gay)/)?(?:scenes/(?:[^/?#]*-)?)?(\d+)/?(?:[?#].*)?$",
                      re.IGNORECASE)
_SLR_PROJECT = {None: "1", "trans": "3", "gay": "4"}
_SLR_LENS = {"mkx200": MKX200, "mkx220": MKX220, "vrca220": VRCA220, "rf52": RF52,
             "fisheye190": RF52}
_SLR_ANGLE_LENS = {190: RF52, 200: MKX200, 220: MKX220}
_SLR_STEREO = {"sbs2l": (SBS, False), "sbs2r": (SBS, True), "ab2l": (TB, False),
               "ab2r": (TB, False), "mono": (MONO, False)}
_SLR_FORMAT = {1: TB, 2: SBS}


def slr_scene_ref(urls):
    """(scene id, project header) of the first SexLikeReal scene URL, or None."""
    for u in urls or ():
        m = _SLR_URL.match((u or "").strip())
        if m:
            return m.group(2), _SLR_PROJECT[(m.group(1) or "").lower() or None]
    return None


def _category_names(data):
    out = []
    for c in data.get("categories") or ():
        name = c.get("name") if isinstance(c, dict) else c
        if isinstance(name, str):
            out.append(name.strip())
    return out


def _slr_relevant(name):
    n = name.lower()
    return "°" in n or "fisheye" in n or "passthrough" in n or "chroma" in n


def slim_slr(data):
    """The fields of an SLR API scene this plugin reads, and nothing else (no
    title, performers or URLs), as kept in the cache."""
    pt = data.get("passthrough")
    out = {"id": data.get("id"), "viewAngle": data.get("viewAngle"),
           "projection": data.get("projection"), "stereomode": data.get("stereomode"),
           "categories": [n for n in _category_names(data) if _slr_relevant(n)]}
    pp = data.get("projectionParams")
    if isinstance(pp, dict):
        out["projectionParams"] = {k: pp[k] for k in ("format", "viewAngle", "projection",
                                                      "cameraLens") if k in pp}
    if isinstance(pt, dict):
        out["passthrough"] = {k: {"enabled": bool((pt.get(k) or {}).get("enabled"))}
                              for k in ("alpha", "aiAlpha", "chromaKey") if isinstance(pt.get(k), dict)}
    return out


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def slr_claim(data):
    """What an SLR API scene says, in this plugin's vocabulary: screen, lens
    (always present for a fisheye: SLR settles it, None for a 180 fisheye),
    stereo, rl, alpha, chroma and raw (a summary for the log). None when the
    answer says nothing usable.

    Projection: projectionParams.projection (0 equirect, 1 fisheye), else the
    top-level projection (0 equirect, 4 fisheye), else a "Fisheye" category.
    viewAngle 360 is a 360 equirect, otherwise a 180. The lens comes from
    projectionParams.cameraLens, else from viewAngle 190 / 200 / 220
    (lens_inferred: the watermark, when readable, beats such a lens).
    Stereo from stereomode (sbs2l, sbs2r, ab2l, mono), else from
    projectionParams.format (1 top/bottom, 2 side by side).
    Passthrough: alpha.enabled (SLR's category "Passthrough (Native)") is the
    alpha matte packed into the file. aiAlpha is SLR's AI mask, streamed
    separately by SLR's own player and enabled on almost every scene, 180
    ones included; the downloaded file carries no matte for it, so it is
    ignored. chromaKey.enabled is green-screen passthrough.
    """
    if not isinstance(data, dict):
        return None
    pp = data.get("projectionParams") if isinstance(data.get("projectionParams"), dict) else {}
    cats = {n.lower() for n in _category_names(data)}
    angle = _int(pp.get("viewAngle")) or _int(data.get("viewAngle"))
    kind = _int(pp.get("projection"))
    if kind in (0, 1):
        fisheye = kind == 1
    else:
        top = _int(data.get("projection"))
        fisheye = top == 4 if top in (0, 4) else ("fisheye" in cats or None)
    out, raw = {}, []
    if angle:
        raw.append(f"viewAngle {angle}")
    if fisheye:
        out["screen"] = FISHEYE
        lens_name = str(pp.get("cameraLens") or "").lower()
        out["lens"] = _SLR_LENS.get(lens_name)
        # a lens only inferred from viewAngle can be wrong (a 200 degree
        # release listed as 190): the watermark may still correct it
        out["lens_inferred"] = out["lens"] is None
        if out["lens_inferred"]:
            out["lens"] = _SLR_ANGLE_LENS.get(angle)
        raw.append("fisheye" + (f" {lens_name}" if lens_name else ""))
    elif fisheye is False or angle:
        is360 = angle == 360 or (not angle and "360°" in cats)
        out["screen"] = SPHERE if is360 else DOME
        raw.append("equirect")
    mode = str(data.get("stereomode") or "").lower()
    if mode in _SLR_STEREO:
        out["stereo"], out["rl"] = _SLR_STEREO[mode]
        raw.append(mode)
    elif _int(pp.get("format")) in _SLR_FORMAT:
        out["stereo"], out["rl"] = _SLR_FORMAT[_int(pp["format"])], False
        raw.append(f"format {pp['format']}")
    pt = data.get("passthrough")
    if isinstance(pt, dict):
        out["alpha"] = bool((pt.get("alpha") or {}).get("enabled"))
        out["chroma"] = bool((pt.get("chromaKey") or {}).get("enabled"))
    else:
        out["alpha"] = "passthrough (native)" in cats
        out["chroma"] = any("chroma" in c for c in cats)
    raw += [n for n, on in (("alpha", out["alpha"]), ("chroma key", out["chroma"])) if on]
    if not (out.get("screen") or out.get("stereo") or out["alpha"] or out["chroma"]):
        return None
    out["raw"] = ", ".join(raw)
    return out


class SlrLookup:
    """SexLikeReal's scene API, politely: one request per second at most, a
    descriptive User-Agent, a timeout, and a JSON cache next to the plugin
    (vrQualityTags.slr.json, keyed by SLR scene id) so a scene is fetched
    again only after SLR_TTL, and a scene SLR does not know after
    SLR_MISS_TTL. Every failure falls back silently to the other sources:
    network errors and server errors are not cached; after SLR_MAX_FAILURES
    of them in a row, or a 429, no more requests are made in this run.
    """

    def __init__(self, path, clock=time.time, sleep=time.sleep, opener=None):
        self.path = path
        self.clock = clock
        self.sleep = sleep
        self.opener = opener or urllib.request.urlopen
        self.next_at = 0.0
        self.failures = 0
        self.stopped = None
        self.cache = self._load()
        # worker threads share the rate limit, the cache and the failure count
        self.lock = threading.Lock()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError as e:
            log("w", f"cannot save the SLR cache to {self.path}: {e}")

    def _fresh(self, entry):
        if not isinstance(entry, dict) or not isinstance(entry.get("fetched"), (int, float)):
            return False
        ttl = SLR_TTL if entry.get("scene") else SLR_MISS_TTL
        return 0 <= self.clock() - entry["fetched"] < ttl

    def _get(self, sid, project):
        """(status, parsed JSON or None); status None on a network failure."""
        wait = self.next_at - self.clock()
        if wait > 0:
            self.sleep(wait)
        req = urllib.request.Request(SLR_API + sid, headers={
            "User-Agent": SLR_USER_AGENT, "Client-Type": "web", "project": project,
            "Accept": "application/json"})
        try:
            with self.opener(req, timeout=SLR_TIMEOUT) as r:
                status, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, body = e.code, b""
            e.close()
        except (urllib.error.URLError, OSError, ValueError):
            status, body = None, b""
        finally:
            self.next_at = self.clock() + SLR_INTERVAL
        try:
            return status, json.loads(body) if body else None
        except ValueError:
            return (None if status == 200 else status), None

    def scene(self, ref):
        """The slimmed SLR scene for (id, project), or None when SLR does not
        know it or cannot be asked right now (stale cached data is used then).
        One thread at a time: the others wait out the rate limit with it."""
        with self.lock:
            return self._scene(ref)

    def _scene(self, ref):
        sid, project = ref
        entry = self.cache.get(sid)
        if self._fresh(entry):
            return entry.get("scene")
        stale = entry.get("scene") if isinstance(entry, dict) else None
        if self.stopped:
            return stale
        status, body = self._get(sid, project)
        if status == 404 and project == "1":
            # scenes outside the main project answer on project 0 (as XBVR does)
            status, body = self._get(sid, "0")
        if status == 200 and isinstance(body, dict) and isinstance(body.get("data"), dict):
            scene = slim_slr(body["data"])
        elif status in (200, 404):
            scene = None                                # SLR does not know it
        else:
            self.failures += 1
            if status == 429:
                self.stopped = "rate limited (429)"
            elif self.failures >= SLR_MAX_FAILURES:
                self.stopped = f"{self.failures} failed requests in a row"
            if self.stopped:
                log("w", f"SLR lookup stopped for this run: {self.stopped}")
            return stale
        self.failures = 0
        self.cache[sid] = {"fetched": self.clock(), "scene": scene}
        self._save()
        return scene


def slr_cache_path(conn):
    return os.path.join(os.path.dirname(state_path(conn)), SLR_CACHE_FILE)


# ------------------------------------------------------------------ quality

def quality_names(cfg):
    return (cfg["tag8k"], cfg["tag7k"], cfg["tag6kHbr"], cfg["parentTag"])


def tier_file(scene):
    """The file a scene's quality is judged by: a scene can hold several
    files, and the biggest one is the one worth judging. None without files."""
    files = scene.get("files") or []
    return max(files, key=lambda x: x.get("size") or 0) if files else None


def tier_of(scene, cfg):
    """Which quality tag (setting key) this scene earns, or None."""
    f = tier_file(scene)
    if f is None:
        return None
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
    """Our own write (tags, the alignment custom field) fires
    Scene.Update.Post again; ignoring updates that touched nothing else stops
    the hook from calling itself."""
    fields = set(ctx.get("inputFields") or [])
    return ctx.get("type") == "Scene.Update.Post" and bool(fields) and \
        fields <= {"id", "tag_ids", "custom_fields"}


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
    if fn["screen"] not in (None, FLAT) or fn["lens"] or fn["vr_word"]:
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
    try:
        cfg["workers"] = min(MAX_WORKERS, max(1, int(float(cfg["workers"]))))
    except (TypeError, ValueError):
        cfg["workers"] = DEFAULTS["workers"]
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
SCENE_FIELDS = "id urls tags{id name} files{width height duration bit_rate size path}"
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
VR_NAME_PATH_REGEX = r"(?i)(^|[^a-z0-9])(vr|180|360|f180|mono[_.-]?(180|360)|fisheye|3dh|3dv|mkx|vrca|rf)"
SCENE_UPDATE = "mutation($i:SceneUpdateInput!){sceneUpdate(input:$i){id}}"
# asked separately, so the scene queries also work on a Stash without them
SCENE_CUSTOM = "query($id:ID!){findScene(id:$id){custom_fields}}"
CUSTOM_FIELDS_PROBE = "{findScenes(filter:{per_page:1}){scenes{id custom_fields}}}"
# every scene carrying one tag, for the stray MONO tidy
SCENE_PAGE_TAG = """query($p:Int!,$f:ID!){findScenes(
  scene_filter:{tags:{value:[$f],modifier:INCLUDES,depth:0}},
  filter:{per_page:100,page:$p,sort:"id",direction:ASC}){
  count scenes{%s}}}""" % SCENE_FIELDS


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
    if cfg.get("detectLowDetail", True):
        get_or_create(LOW_DETAIL)
    else:
        # off: an existing tag is still managed (removed), none is created
        hits = stash.call(TAG_BY_NAME, {"n": LOW_DETAIL})["findTags"]["tags"]
        if hits:
            found[LOW_DETAIL] = hits[0]

    parent = found[cfg["parentTag"]]
    for key in ("tag8k", "tag7k", "tag6kHbr"):
        child = found[cfg[key]]
        current = {p["id"] for p in (child.get("parents") or [])}
        if parent["id"] not in current:
            stash.call(TAG_UPDATE, {"i": {"id": child["id"],
                                          "parent_ids": sorted(current | {parent["id"]})}})
            log("i", f"filed {cfg[key]} under {cfg['parentTag']}")
    return {n: t["id"] for n, t in found.items()}


def lookup_slr(cfg, scene, w, h, res, screen_px, stereo_px):
    """(vetted SLR claim or None, text for the log line). SLR's projection has
    to agree with the frame in coarse shape (and fit the eye), except that an
    equirect 180 beats a disc-only fisheye (see vet_claim()); when it does
    not, the whole answer is set aside and the pixels are kept."""
    lookup = cfg.get("_slr")
    ref = slr_scene_ref(scene.get("urls")) if lookup else None
    if not ref:
        return None, ""
    claim = slr_claim(lookup.scene(ref))
    if not claim:
        return None, ""
    vetted, dropped = vet_claim(claim, w, h, res, screen_px, stereo_px, disc_yields=True)
    why = f", SLR {ref[0]} {claim['raw']}"
    if screen_px == FISHEYE and vetted.get("screen") == DOME:
        why += " (SLR equirect 180 over the disc test)"
    if claim.get("screen") and "screen" not in vetted:
        log("i", f"scene {scene.get('id')}: SLR {ref[0]} says {claim['raw']}, which the frame "
                 f"contradicts ({'; '.join(dropped)}); kept the pixels")
        return None, why + " (contradicts the frame, not used)"
    vetted["alpha"], vetted["chroma"] = claim["alpha"], claim["chroma"]
    if dropped:
        why += " (not trusted: " + "; ".join(dropped) + ")"
    return vetted, why


def measure_projection(cfg, scene, detail=None, align=None):
    """(wanted projection tag names, reason) or (None, reason) when the scene
    cannot be measured and its projection tags must be left alone.

    detail: a dict to also measure the primary file's detail into (same
    decodes, see probe()); once the frames were measured it holds "verdict",
    detail_verdict() or None. Left empty when nothing was decoded.
    align: a dict that receives the same frames scaled for the stereo
    alignment (probe()'s stereo)."""
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
    # one cheap ffprobe call before any frame is decoded
    meta_claim = read_metadata(cfg, path)
    res = probe(cfg, path, w, h, dur, detail=detail is not None, stereo=align)
    if detail is not None and res is not None:
        detail["verdict"] = res.get("detail")
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

    meta, dropped = vet_claim(meta_claim, w, h, res, screen, stereo)
    if meta_claim:
        why += f", metadata {meta_claim['raw']}"
        if dropped:
            why += " (not trusted: " + "; ".join(dropped) + ")"
    slr, slr_why = lookup_slr(cfg, scene, w, h, res, screen, stereo)
    why += slr_why

    fov = None
    skip = (fov_skip_reason(fn, screen, alpha, path, meta, slr)
            if cfg["readFovWatermark"] else "off")
    if skip is None:
        fov = read_fov(cfg, path, dur)
        if fov:
            why += f", watermark {fov[0]}deg"
            if slr and slr.get("lens_inferred") and _lens_of(fov) != slr.get("lens"):
                why += f" overrides SLR's {slr.get('lens') or 'no lens'} (from viewAngle)"
    elif skip == "passthrough not from SLR":
        why += ", watermark not read (passthrough not from SLR)"
    marks = [k for k in ("stereo", "screen", "lens") if fn[k]] + (["rl"] if fn["rl"] else [])
    if marks:
        why += ", filename " + "+".join(str(fn[k]) if k != "rl" else "RL" for k in marks)
    return resolve(fn, screen, stereo, alpha, fov, meta, slr), why


def process_scene(stash, cfg, scene, ids, mode):
    """Apply this pass to one scene; returns a log line when it changed."""
    have_names = {t["name"] for t in scene.get("tags") or []}
    if SKIP in have_names:
        return None
    have = {t["id"] for t in scene.get("tags") or []}
    align = None                    # the stereo alignment: {"deg", "why"} once measured

    if mode == "clear":
        scope_names = set(PROJECTION_TAGS) | set(quality_names(cfg)) | {LOW_DETAIL}
        want_names, why = set(), "cleared"
        if cfg.get("_custom_fields"):
            align = {"deg": None, "why": "cleared"}
    else:
        scope_names = set(quality_names(cfg))
        want_names = quality_want(scene, cfg)
        why = "quality"
        tier = tier_of(scene, cfg)
        # the detail is measured on the primary file, so only when that is
        # the file the tier was judged by
        detail = ({} if cfg["detectLowDetail"] and tier and
                  tier_file(scene) is (scene.get("files") or [None])[0] else None)
        frames = {} if cfg["detectVerticalOffset"] and cfg.get("_custom_fields") else None
        remeasure = mode == "retag" or cfg["overwrite"]
        if remeasure or not settled(have_names):
            proj, reason = measure_projection(cfg, scene, detail, frames)
            if proj is not None:
                scope_names |= set(PROJECTION_TAGS)
                want_names |= proj
                why = reason
                if frames is not None:
                    align = measure_alignment(cfg, scene["files"][0], proj, frames)
            elif reason == "file missing":
                log("w", f"scene {scene['id']}: file missing, projection skipped")
        scope, want, ld_why = low_detail_tags(cfg, scene, tier, detail)
        scope_names |= scope
        want_names |= want
        if ld_why:
            why += f", {ld_why}"

    # LOW_DETAIL has no id when the feature is off and the tag never existed
    scope_names = {n for n in scope_names if n in ids}
    new = diff_tags(have, {ids[n] for n in scope_names}, {ids[n] for n in want_names})
    cf = voffset_update(read_custom_fields(stash, scene["id"]), align["deg"]) if align else None
    if new is None and cf is None:
        return None
    update = {"id": scene["id"]}
    if new is not None:
        update["tag_ids"] = sorted(new)
    if cf is not None:
        update["custom_fields"] = cf
    stash.call(SCENE_UPDATE, {"i": update})
    managed = scope_names & {n for n in ids if ids[n] in (have if new is None else new)}
    out = f"{' '.join(sorted(managed)) or '(none)'}  ({why})"
    if cf is not None:
        out += f"; {voffset_text(cf, align['why'])}"
    return out


def read_custom_fields(stash, sid):
    """A scene's custom fields ({} when it has none)."""
    sc = stash.call(SCENE_CUSTOM, {"id": str(sid)})["findScene"]
    return (sc or {}).get("custom_fields") or {}


def custom_fields_supported(stash):
    """Whether this Stash has scene custom fields (Stash v0.31 has them)."""
    try:
        stash.call(CUSTOM_FIELDS_PROBE)
        return True
    except Exception as e:
        log("w", f"this Stash has no scene custom fields ({e}); the stereo alignment "
                 "is not measured")
        return False


def process_alignment(stash, cfg, scene, ids, mode):
    """The alignment task: measure only the vertical offset between the eyes
    of a scene and set or remove its VOFFSET_FIELD from the projection tags
    it already carries; no tag and no other custom field is touched. A scene
    without a projection tag is left alone. Returns a log line when it
    changed."""
    have_names = {t["name"] for t in scene.get("tags") or []}
    files = scene.get("files") or []
    if SKIP in have_names or not files or not have_names & set(SCREEN_TAGS):
        return None
    align = measure_alignment(cfg, files[0], have_names)
    if not align:
        return None
    cf = voffset_update(read_custom_fields(stash, scene["id"]), align["deg"])
    if cf is None:
        return None
    stash.call(SCENE_UPDATE, {"i": {"id": scene["id"], "custom_fields": cf}})
    return voffset_text(cf, align["why"])


def low_detail_tags(cfg, scene, tier, detail):
    """(scope, want, text for the log) of LOW_DETAIL for one scene.

    Off (detectLowDetail false): removed. No tier tag: removed. Measured this
    pass (detail holds a verdict): the full rule, low_detail(). Not measured:
    the tag is left as the last measurement set it (the bitrate alone never
    decides, it only tips a middling ratio).
    """
    if not cfg["detectLowDetail"] or not tier:
        return {LOW_DETAIL}, set(), ""
    f = tier_file(scene)
    if detail and "verdict" in detail:
        low, why = low_detail(f, detail["verdict"])
        return {LOW_DETAIL}, ({LOW_DETAIL} if low else set()), f"low detail: {why}" if low \
            else why
    return set(), set(), ""


def process_detail(stash, cfg, scene, ids, mode):
    """The detail task: measure only the detail of a scene with a tier tag and
    set or remove LOW_DETAIL; every other tag is left alone, and nothing but
    the native crop of two frames is decoded. Returns a log line when it
    changed."""
    have_names = {t["name"] for t in scene.get("tags") or []}
    if SKIP in have_names or LOW_DETAIL not in ids:
        return None
    tier = tier_of(scene, cfg)
    f = tier_file(scene)
    # measured on the primary file only when that is the file the tier was
    # judged by, as in process_scene()
    detail = measure_detail(cfg, f) if tier and f is (scene.get("files") or [None])[0] \
        else {}
    scope, want, why = low_detail_tags(cfg, scene, tier, detail)
    if not scope:
        return None
    have = {t["id"] for t in scene.get("tags") or []}
    new = diff_tags(have, {ids[LOW_DETAIL]}, {ids[n] for n in want})
    if new is None:
        return None
    stash.call(SCENE_UPDATE, {"i": {"id": scene["id"], "tag_ids": sorted(new)}})
    return f"{LOW_DETAIL} {'added' if want else 'removed'}  ({why or 'no tier'})"


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
DETAIL_STATE_FILE = "vrQualityTags.detail-state.json"
ALIGNMENT_STATE_FILE = "vrQualityTags.alignment-state.json"
RESUME_MAX_AGE = 7 * 86400          # an older unfinished retag starts over


def progress(fraction):
    # Stash reads "\x01p\x02<float>" as the task's progress, 0 to 1
    log("p", f"{min(1.0, max(0.0, fraction)):.4f}")


def state_path(conn, name=STATE_FILE):
    """The resume state lives next to the plugin: Stash passes its directory as
    server_connection.PluginDir."""
    d = (conn or {}).get("PluginDir") or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(d, name)


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
    """One task run over every candidate scene, cfg["workers"] of them at a
    time: threads overlap the ffmpeg decodes, and as many processes run the
    arithmetic (see crunch()). state (a RetagState) makes the run
    resumable: scenes up to state.last_id are skipped, each completed scene
    is recorded, and the state is cleared once the run completes."""
    todo = candidates(stash, cfg)
    if mode in ("detail", "alignment"):
        # flat 3D files earn no tier and have no VR stereo pair; every
        # measured scene gets the detail (or the alignment) only
        handler = process_detail if mode == "detail" else process_alignment
        todo = [(sc, handler, kind) for sc, h, kind in todo if h is process_scene]
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
    rest = todo[start:]

    def one(item):
        sc, handler, _ = item
        try:
            return handler(stash, cfg, sc, ids, mode)
        except Exception as e:
            log("e", f"scene {sc['id']}: {type(e).__name__}: {e}")
            return None

    workers = cfg.get("workers", 1)
    pool = concurrent.futures.ThreadPoolExecutor(workers) if workers > 1 else None
    if pool:
        start_cpu_pool(workers)
    # map() hands the results back in scene order, so the resume state only
    # ever records a scene once every scene before it is done too
    results = pool.map(one, rest) if pool else map(one, rest)
    changed = 0
    try:
        for n, ((sc, _, _), r) in enumerate(zip(rest, results), start + 1):
            if r:
                changed += 1
                log("i", f"scene {sc['id']}: {r}")
            if state is not None:
                state.done(sc["id"])
            progress(n / len(todo))
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
            stop_cpu_pool()
    if state is not None:
        state.clear()
    log("i", f"done ({mode}): {len(todo) - start} scenes examined, {changed} changed")


# MONO only means something next to a VR projection: stash-vr's MONO rule sets
# the stereo mode, and without a projection the scene plays flat anyway.
# This plugin writes it with DOME and SPHERE, and with FISHEYE (and so a lens)
# for a single-disc fisheye.
MONO_COMPANIONS = (DOME, SPHERE, FISHEYE) + LENS_TAGS + ("CUBEMAP", "EAC")


def stray_mono(scene):
    """Whether a scene carries MONO without any VR projection tag (left behind
    on 2D videos by older tagging, or next to FLAT). VRP: Skip scenes are never
    touched."""
    have = {t["name"] for t in scene.get("tags") or []}
    return MONO in have and SKIP not in have and not have & set(MONO_COMPANIONS)


def tidy_mono(stash, ids):
    """Remove MONO from every scene where it is stray; returns how many.

    Runs after the measuring pass of a task, so every scene that pass measured
    already carries what the measurement says and anything still stray is
    left over. The candidates are all collected before the first write: the
    writes shrink the tag query's result, which would shift later pages.
    """
    todo = [sc for sc, _ in _pages(stash, SCENE_PAGE_TAG, ids[MONO]) if stray_mono(sc)]
    removed = 0
    for sc in todo:
        keep = sorted({t["id"] for t in sc.get("tags") or []} - {ids[MONO]})
        try:
            stash.call(SCENE_UPDATE, {"i": {"id": sc["id"], "tag_ids": keep}})
        except Exception as e:
            log("e", f"scene {sc['id']}: {type(e).__name__}: {e}")
            continue
        removed += 1
        log("i", f"scene {sc['id']}: stray MONO removed (no DOME, SPHERE or FISHEYE)")
    log("i", f"stray MONO: removed from {removed} scenes")
    return removed


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
    if cfg["slrLookup"]:
        cfg["_slr"] = SlrLookup(slr_cache_path(payload.get("server_connection")))
    mode = args.get("mode") or "hook"
    if mode == "all":                   # task name of the pre-merge quality plugin
        mode = "untagged"
    ids = ensure_tags(stash, cfg)
    if cfg["detectVerticalOffset"] or mode == "clear":
        cfg["_custom_fields"] = custom_fields_supported(stash)

    if mode in ("retag", "retag_fresh"):
        path = state_path(payload.get("server_connection"))
        state = RetagState.load(path) if mode == "retag" else None
        if state is None:
            if mode == "retag_fresh":
                log("i", "retag from the beginning; any saved progress is discarded")
            state = RetagState(path)
        run_all(stash, cfg, ids, "retag", state)
        tidy_mono(stash, ids)
    elif mode == "detail":
        if not cfg["detectLowDetail"]:
            log("w", "the Low Detail setting is off: nothing to measure")
        else:
            path = state_path(payload.get("server_connection"), DETAIL_STATE_FILE)
            run_all(stash, cfg, ids, mode, RetagState.load(path) or RetagState(path))
    elif mode == "alignment":
        if not cfg["detectVerticalOffset"]:
            log("w", "the stereo alignment setting is off: nothing to measure")
        elif cfg["_custom_fields"]:
            path = state_path(payload.get("server_connection"), ALIGNMENT_STATE_FILE)
            run_all(stash, cfg, ids, mode, RetagState.load(path) or RetagState(path))
    elif mode == "untagged":
        run_all(stash, cfg, ids, mode)
        tidy_mono(stash, ids)
    elif mode == "clear":
        run_all(stash, cfg, ids, mode)
    elif mode == "tidy_mono":
        tidy_mono(stash, ids)
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
