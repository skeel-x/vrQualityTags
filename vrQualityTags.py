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

Outside the VR path a file is measured only when Stash's metadata alone says it
is VR (a 2:1 or square frame at least 3840 wide, or a VR marker in its name).
Flat (non-VR) stereoscopic 3D files there are recognised from their names alone
(3D, SBS, Half-SBS, LRF, HOU, ...) and get FLAT + SBS or FLAT + TB. Every other
file there is left untouched.

Pure standard library on purpose: the plugin interpreter has no numpy or PIL,
and arithmetic over a 256px thumbnail does not need them.
"""
import cmath
import json
import math
import operator
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

VERSION = "2.4.0"

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
}

NUMERIC = ("min8kWidth", "min7kWidth", "min6kWidth", "min6kBitrateMbit", "minWidth")
BOOLEAN = ("readFovWatermark", "overwrite", "measureVrShapedOutside", "flat3dFilenameScan",
           "slrLookup", "detectLowDetail")

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
    serves both (the decode is the expensive part); the crop goes through a
    temporary file because two raw outputs cannot share stdout."""
    x, y, cw, ch = box
    tmp = f"/tmp/vrq_crop_{os.getpid()}.raw"
    graph = (f"[0:v]split=2[a][b];[a]scale={tw}:{th}:out_range=pc[t];"
             f"[b]crop={cw}:{ch}:{x}:{y}[c]")
    cmd = [cfg["ffmpegPath"], "-nostdin", "-v", "error", "-y", "-skip_frame", "nokey",
           "-ss", f"{ts:.3f}", "-i", path, "-filter_complex", graph,
           "-map", "[t]", "-frames:v", "1", "-pix_fmt", "yuv444p", "-f", "rawvideo", "-",
           "-map", "[c]", "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", tmp]
    crop = None
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
        if os.path.exists(tmp):
            with open(tmp, "rb") as f:
                crop = f.read()
    except (OSError, subprocess.SubprocessError):
        return None, None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    yuv = p.stdout[:tw * th * 3] if p.returncode == 0 and len(p.stdout) >= tw * th * 3 else None
    if p.returncode != 0 or crop is None or len(crop) < cw * ch:
        crop = None
    return yuv, crop and crop[:cw * ch]


def probe(cfg, path, w, h, duration, detail=False):
    """Median metrics over two frames. With detail, the same two decodes also
    yield a native crop of the eye centre each, and res["detail"] holds
    detail_verdict() of them (None: not measurable)."""
    tw, th = thumb_size(w, h)
    wide = w / h > 1.5
    box = detail_box(w, h) if detail else None
    frames, blocks = [], []
    n = tw * th
    for frac in (0.4, 0.6):
        ts = (duration or 600) * frac
        if box:
            yuv, crop = grab_with_crop(cfg, path, ts, tw, th, box)
            if crop:
                blocks.append(detail_blocks(crop, box[2], box[3]))
        else:
            yuv = grab(cfg, path, ts, tw, th)
        if yuv is None:
            continue
        m = frame_metrics(yuv_to_rgb(yuv, n), tw, th, wide, grey=yuv[:n])
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
        know it or cannot be asked right now (stale cached data is used then)."""
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


def measure_projection(cfg, scene, detail=None):
    """(wanted projection tag names, reason) or (None, reason) when the scene
    cannot be measured and its projection tags must be left alone.

    detail: a dict to also measure the primary file's detail into (same
    decodes, see probe()); once the frames were measured it holds "verdict",
    detail_verdict() or None. Left empty when nothing was decoded."""
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
    res = probe(cfg, path, w, h, dur, detail=True) if detail is not None else \
        probe(cfg, path, w, h, dur)
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

    if mode == "clear":
        scope_names = set(PROJECTION_TAGS) | set(quality_names(cfg)) | {LOW_DETAIL}
        want_names, why = set(), "cleared"
    else:
        scope_names = set(quality_names(cfg))
        want_names = quality_want(scene, cfg)
        why = "quality"
        tier = tier_of(scene, cfg)
        # the detail is measured on the primary file, so only when that is
        # the file the tier was judged by
        detail = ({} if cfg["detectLowDetail"] and tier and
                  tier_file(scene) is (scene.get("files") or [None])[0] else None)
        remeasure = mode == "retag" or cfg["overwrite"]
        if remeasure or not settled(have_names):
            proj, reason = measure_projection(cfg, scene, detail)
            if proj is not None:
                scope_names |= set(PROJECTION_TAGS)
                want_names |= proj
                why = reason
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
    if new is None:
        return None
    stash.call(SCENE_UPDATE, {"i": {"id": scene["id"], "tag_ids": sorted(new)}})
    managed = scope_names & {n for n in ids if ids[n] in new}
    return f"{' '.join(sorted(managed)) or '(none)'}  ({why})"


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

    if mode in ("retag", "retag_fresh"):
        path = state_path(payload.get("server_connection"))
        state = RetagState.load(path) if mode == "retag" else None
        if state is None:
            if mode == "retag_fresh":
                log("i", "retag from the beginning; any saved progress is discarded")
            state = RetagState(path)
        run_all(stash, cfg, ids, "retag", state)
        tidy_mono(stash, ids)
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
