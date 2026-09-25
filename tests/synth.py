"""Synthetic rgb24 frames for the pixel tests."""
import math


def noise(x, y, seed=0):
    """Deterministic texture in 40..220, different per seed."""
    n = (x * 374761393 + y * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return 40 + (n >> 24) % 181


def blocky(x, y, seed=0):
    """Texture with 8px blocks, so it survives the correlation and bbox steps."""
    return noise(x // 8, y // 8, seed)


def frame(w, h, pixel):
    """Build rgb24 bytes from pixel(x, y) -> (r, g, b)."""
    out = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            k = (y * w + x) * 3
            out[k:k + 3] = bytes(pixel(x, y))
    return bytes(out)


def grey(v):
    return (v, v, v)


def in_disc(x, y, x0, ew, eh, r=0.98):
    cx, cy = x0 + (ew - 1) / 2.0, (eh - 1) / 2.0
    return math.hypot((x - cx) / (ew / 2.0), (y - cy) / (eh / 2.0)) <= r


def fisheye_sbs(w=256, h=128, corner=None, seed=1):
    """Two identical fisheye discs side by side. corner(x_in_eye, y) may paint
    the area outside the discs; black otherwise."""
    ew = w // 2

    def px(x, y):
        x0 = 0 if x < ew else ew
        if in_disc(x, y, x0, ew, h):
            return grey(blocky(x - x0, y, seed))
        if corner:
            c = corner(x - x0, y)
            if c:
                return c
        return (0, 0, 0)
    return frame(w, h, px)


def equirect_sbs(w=256, h=128, band=0.12, seed=2):
    """180 equirect pair: full eye width, black bands top and bottom."""
    ew = w // 2
    lo, hi = int(h * band), int(h * (1 - band))

    def px(x, y):
        if y < lo or y >= hi:
            return (0, 0, 0)
        return grey(blocky(x % ew, y, seed))
    return frame(w, h, px)


def mono(w, h, seed=3):
    return frame(w, h, lambda x, y: grey(blocky(x, y, seed)))


def tb(w=256, h=256, seed=4):
    hh = h // 2
    return frame(w, h, lambda x, y: grey(blocky(x, y % hh, seed)))


def packed_alpha_tb(w=256, h=256, seed=5):
    """Picture on top, its binary silhouette below."""
    hh = h // 2

    def px(x, y):
        v = blocky(x, y % hh, seed)
        if y < hh:
            return grey(v)
        return grey(255 if v > 130 else 0)
    return frame(w, h, px)


def shifted_sbs(w=256, h=128, near=7, far=5, seed=7):
    """A 180 pair with parallax: the right eye sees the lower half of the
    picture (the performer, close) shifted `near` px left and the upper half
    (the room) `far` px, so the halves do not line up pixel for pixel."""
    ew = w // 2

    def px(x, y):
        if x < ew:
            return grey(blocky(x, y, seed))
        shift = near if y >= h // 2 else far
        return grey(blocky(x - ew + shift, y, seed))
    return frame(w, h, px)


def panorama(w=256, h=128, seed=8):
    """A mono 360 equirect: blocky texture that repeats every `w` pixels with
    the seam in the middle of a block, so the last column continues into the
    first."""
    blocks = w // 8

    def px(x, y):
        return grey(noise(((x + 4) // 8) % blocks, y // 8, seed))
    return frame(w, h, px)


# ------------------------------------------------------------ detail (grey)

def grey_noise(w, h, seed=0):
    """A w x h grey buffer of per-pixel noise: detail up to Nyquist."""
    return bytes(noise(x, y, seed) for y in range(h) for x in range(w))


def soften(buf, w, h):
    """A [1, 2, 1] / 4 blur along both axes: the softness of a real lens,
    which leaves a genuine picture detail below Nyquist but not at it."""
    rows = []
    for y in range(h):
        r = buf[y * w:(y + 1) * w]
        rows.append([(r[max(0, x - 1)] + 2 * r[x] + r[min(w - 1, x + 1)]) / 4 for x in range(w)])
    out = bytearray(w * h)
    for y in range(h):
        a, b, c = rows[max(0, y - 1)], rows[y], rows[min(h - 1, y + 1)]
        for x in range(w):
            out[y * w + x] = int((a[x] + 2 * b[x] + c[x]) / 4 + 0.5)
    return bytes(out)


def upscale2(buf, w, h):
    """A bilinear 2x upscale of a w x h grey buffer (pixel centres aligned,
    as ffmpeg's scaler does)."""
    def taps(i, n):
        c = (i + 0.5) / 2 - 0.5
        j = math.floor(c)
        return max(0, j), min(n - 1, j + 1), c - j
    big_w = 2 * w
    xs = [taps(x, w) for x in range(big_w)]
    out = bytearray(big_w * 2 * h)
    for y in range(2 * h):
        y0, y1, ty = taps(y, h)
        r0, r1 = buf[y0 * w:(y0 + 1) * w], buf[y1 * w:(y1 + 1) * w]
        for x, (x0, x1, tx) in enumerate(xs):
            a = r0[x0] + (r0[x1] - r0[x0]) * tx
            b = r1[x0] + (r1[x1] - r1[x0]) * tx
            out[y * big_w + x] = int(a + (b - a) * ty + 0.5)
    return bytes(out)
