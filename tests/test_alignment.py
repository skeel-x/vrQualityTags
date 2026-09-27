"""Stereo alignment: the vertical offset between the eyes and its custom field."""
import io
import json
import math
import os
import tempfile
import time
import unittest
from unittest import mock

import vrQualityTags as v
from tests import synth

E = 512                     # eye size of the synthetic pairs (the plugin uses 1024)
EQ = v.eye_projection(v.DOME, None, E, E)


def cfg(**kw):
    c = v.load_config({})
    c.update(kw)
    return c


def depth(x, y):
    """Horizontal disparity of a room: nearer towards the bottom, with some
    relief, as a 180 camera sees it (negative: crossed, in front of the
    screen plane)."""
    return -6 - 30 * (y / E) - 8 * math.sin(x / 40)


_FRAMES = {}


def pair(name):
    """The fit of one synthetic stereo frame (built once per run)."""
    if name not in _FRAMES:
        layout, screen, shift = SCENARIOS[name]
        buf = synth.stereo_pair(E, E, shift, layout=layout)
        size = (2 * E, E) if layout == "SBS" else (E, 2 * E)
        _FRAMES[name] = v.frame_voffset(buf, size[0], size[1], layout, False, screen, None)
    return _FRAMES[name]


FISH = v.eye_projection(v.FISHEYE, None, E, E)
ROT = math.radians(0.5)


def _parallax(x, y):
    # a near object in the two quadrants where the natural vertical parallax
    # has the same sign, so a plain median of dy is biased by it
    q = (x - E / 2) * (y - E / 2) / (E / 5) ** 2
    dx = -8 - 40 * max(0.0, min(1.0, q))
    return dx, v.parallax_ratio(EQ, x, y) * dx


SCENARIOS = {
    "+0.5": ("SBS", v.DOME, lambda x, y: (depth(x, y), 0.5)),
    "-0.5": ("SBS", v.DOME, lambda x, y: (depth(x, y), -0.5)),
    "+1": ("SBS", v.DOME, lambda x, y: (depth(x, y), 1.0)),
    "-2 TB": ("TB", v.DOME, lambda x, y: (depth(x, y), -2.0)),
    "fisheye +1": ("SBS", v.FISHEYE, lambda x, y: (depth(x, y) / 2, 1.0)),
    # the right eye rolled by 0.5 degrees about the eye centre, and 1 px low
    "rotated": ("SBS", v.DOME, lambda x, y: (depth(x, y), 1.0 + ROT * (x - E / 2))),
    "parallax": ("SBS", v.DOME, _parallax),
    # the right camera mounted higher: near objects shift down in proportion
    # to their parallax, far ones hardly at all
    "height": ("SBS", v.DOME, lambda x, y: (1.3 * depth(x, y), 0.15 * 1.3 * depth(x, y))),
}


class Measure(unittest.TestCase):
    def test_known_vertical_shifts(self):
        for name, want in (("+0.5", 0.5), ("-0.5", -0.5), ("+1", 1.0), ("-2 TB", -2.0),
                           ("fisheye +1", 1.0)):
            with self.subTest(name):
                f = pair(name)
                self.assertTrue(v.frame_counts(f), f)
                self.assertAlmostEqual(f["c0"], want, delta=0.1)
                self.assertLess(abs(f["near"]), 1.0)

    def test_relative_rotation_is_not_an_offset(self):
        f = pair("rotated")
        self.assertTrue(v.frame_counts(f), f)
        # c1 is dy per half eye width: 0.5 degrees over 256 px
        self.assertAlmostEqual(f["c1"], ROT * E / 2, delta=0.15)
        self.assertAlmostEqual(f["c0"], 1.0, delta=0.25)

    def test_natural_parallax_is_not_an_offset(self):
        left, right, _, _ = v.split_eyes(
            synth.stereo_pair(E, E, _parallax), 2 * E, E, v.SBS)
        tiles = v.match_tiles(left, right, EQ)
        dys = [t[3] for t in tiles]
        self.assertGreater(abs(sum(dys) / len(dys)), 0.8)     # the raw shifts are biased
        f = v.fit_voffset(tiles, EQ)
        self.assertTrue(v.frame_counts(f), f)
        self.assertLess(abs(f["c0"]), 0.2)
        self.assertLess(abs(f["near"]), 1.0)

    def test_cameras_at_different_heights(self):
        f = pair("height")
        self.assertTrue(v.frame_counts(f), f)
        self.assertGreater(abs(f["near"]), v.VOFFSET_NEAR_MAX)
        deg, why = v.voffset_verdict([f, f, f], 0.2)
        self.assertIsNone(deg)
        self.assertIn("nearness", why)

    def test_flat_picture_is_not_measured(self):
        buf = bytes([128]) * (2 * E * E)
        f = v.frame_voffset(buf, 2 * E, E, v.SBS, False, v.DOME, None)
        self.assertFalse(v.frame_counts(f))
        self.assertEqual(v.frame_voffset(buf, 2 * E, E, v.SBS, False, v.SPHERE, None),
                         {"ok": False, "n": 0})


def fit(c0, near=0.0, tiles=30, se0=0.1):
    return {"ok": True, "tiles": tiles, "n": tiles, "c0": c0, "c1": 0.0, "se0": se0, "near": near}


class Verdict(unittest.TestCase):
    DPP = 180 / 1024

    def test_three_agreeing_frames(self):
        deg, why = v.voffset_verdict([fit(-4.85), fit(-4.77), fit(-4.9)], self.DPP)
        self.assertEqual(deg, -0.85)            # same sign as the pixels, 2 decimals
        self.assertIn("3 frames", why)
        deg, _ = v.voffset_verdict([fit(5.1), fit(5.2), fit(5.0), fit(5.3)], self.DPP)
        self.assertEqual(deg, 0.91)
        deg, _ = v.voffset_verdict([fit(0.01), fit(-0.02), fit(0.0)], self.DPP)
        self.assertEqual(deg, 0.0)
        self.assertEqual(str(deg), "0.0")       # never -0.0

    def test_needs_three_frames_that_count(self):
        self.assertIsNone(v.voffset_verdict([fit(1.0), fit(1.1)], self.DPP)[0])
        few = fit(1.0, tiles=14)
        loose = fit(1.0, se0=0.36)
        failed = {"ok": False, "n": 3}
        deg, why = v.voffset_verdict([fit(1.0), fit(1.1), few, loose, failed], self.DPP)
        self.assertIsNone(deg)
        self.assertIn("2 of 5 frames", why)
        self.assertEqual(v.voffset_verdict([fit(1.0), fit(1.1), fit(1.0, tiles=15)],
                                           self.DPP)[0], 0.18)

    def test_frames_that_change_direction(self):
        # up in some frames, down in others, one of them noticeably: unknown
        deg, why = v.voffset_verdict([fit(2.05), fit(1.86), fit(-2.19)], self.DPP)
        self.assertIsNone(deg)
        self.assertIn("disagree in direction", why)
        # two such frames are enough to know
        self.assertIn("disagree", v.voffset_verdict([fit(-2.2), fit(1.6)], self.DPP)[1])
        self.assertTrue(v.voffset_conflict([-2.2, 1.6], self.DPP))
        self.assertFalse(v.voffset_conflict([-0.5, 1.2], self.DPP))   # both below 0.25 deg
        self.assertFalse(v.voffset_conflict([2.0, 3.5], self.DPP))

    def test_agreeing_frames_give_the_mean(self):
        # within 1 px: the mean, as before
        self.assertEqual(v.voffset_verdict([fit(1.0), fit(2.0), fit(1.5)], self.DPP)[0], 0.26)

    def test_one_way_by_different_amounts_gives_the_smallest(self):
        # -0.63/-0.70/-0.69/-0.46 deg: all down, more than 1 px apart
        px = [d / self.DPP for d in (-0.63, -0.70, -0.69, -0.46)]
        deg, why = v.voffset_verdict([fit(x) for x in px], self.DPP)
        self.assertEqual(deg, -0.46)
        self.assertIn("smallest", why)
        self.assertEqual(v.voffset_verdict([fit(1.0), fit(2.05), fit(1.5)], self.DPP)[0], 0.18)

    def test_all_small_gives_the_mean_even_across_zero(self):
        px = [d / self.DPP for d in (-0.23, -0.12, 0.18)]
        deg, why = v.voffset_verdict([fit(x) for x in px], self.DPP)
        self.assertEqual(deg, -0.06)
        self.assertIn("mean", why)
        # two frames only: still too few
        self.assertIn("2 of 2 frames", v.voffset_verdict([fit(0.2), fit(1.5)], self.DPP)[1])

    def test_nearness_is_the_median_over_frames(self):
        # one frame with a large reading is noise; most frames with it is the rig
        self.assertIsNotNone(v.voffset_verdict(
            [fit(1.0, near=4.0), fit(1.0, near=0.2), fit(1.0, near=-0.3)], self.DPP)[0])
        self.assertIsNone(v.voffset_verdict(
            [fit(1.0, near=-2.5), fit(1.0, near=-2.2), fit(1.0, near=0.1)], self.DPP)[0])

    def test_implausibly_large(self):
        deg, why = v.voffset_verdict([fit(12.0), fit(12.2), fit(12.1)], self.DPP)
        self.assertIsNone(deg)
        self.assertIn("implausibly", why)


class Pieces(unittest.TestCase):
    def test_xcorr_matches_a_plain_sum(self):
        win = [bytes((x * 37 + y * 11 + x * y) % 256 for x in range(9)) for y in range(7)]
        tile = [bytes((x * 5 + y * 29 + 3) % 256 for x in range(4)) for y in range(3)]
        out, base = v.xcorr(win, tile)
        for dy in range(7 - 3 + 1):
            for dx in range(9 - 4 + 1):
                want = sum(tile[j][i] * win[j + dy][i + dx] for j in range(3) for i in range(4))
                self.assertEqual(out[base + dy * 9 + dx], want)

    def test_largest_tile_does_not_carry(self):
        tile = [bytes([255]) * v.VOFFSET_TILE] * v.VOFFSET_TILE
        win = [bytes([255]) * (v.VOFFSET_TILE + 2)] * (v.VOFFSET_TILE + 2)
        out, base = v.xcorr(win, tile)
        self.assertEqual(out[base], 255 * 255 * v.VOFFSET_TILE ** 2)
        self.assertEqual(out[base + v.VOFFSET_TILE + 2 + 1], 255 * 255 * v.VOFFSET_TILE ** 2)

    def test_ncc_grid(self):
        pic = [bytes((x * 13 + y * 7 + (x * y) % 17) % 256 for x in range(12)) for y in range(10)]
        tile = [r[3:7] for r in pic[4:8]]
        grid = v.ncc_grid(pic, tile, v.integral(pic), 0, 0)
        c, ix, iy = v._peak(grid)
        self.assertEqual((ix, iy), (3, 4))
        self.assertAlmostEqual(c, 1.0, places=9)

    def test_shrink_and_integral(self):
        rows = [bytes([1, 2, 3, 4]), bytes([5, 6, 7, 8]), bytes([9, 10, 11, 12]),
                bytes([13, 14, 15, 16])]
        self.assertEqual(v.shrink(rows, 2), [bytes([4, 6]), bytes([12, 14])])
        s, q = v.integral(rows)
        self.assertEqual(v._box(s, 1, 1, 2, 2), 6 + 7 + 10 + 11)
        self.assertEqual(v._box(q, 0, 0, 1, 2), 1 + 25)

    def test_split_eyes(self):
        sbs = bytes([1, 1, 2, 2]) * 2                      # 4 x 2, left eye 1, right eye 2
        a, b, ew, eh = v.split_eyes(sbs, 4, 2, v.SBS)
        self.assertEqual((a[0], b[0], ew, eh), (b"\x01\x01", b"\x02\x02", 2, 2))
        a, b, _, _ = v.split_eyes(sbs, 4, 2, v.SBS, rl=True)
        self.assertEqual((a[0], b[0]), (b"\x02\x02", b"\x01\x01"))
        tb = bytes([1] * 4 + [2] * 4)                       # 2 x 4, top eye 1
        a, b, ew, eh = v.split_eyes(tb, 2, 4, v.TB)
        self.assertEqual((a, b, ew, eh), ([b"\x01\x01"] * 2, [b"\x02\x02"] * 2, 2, 2))

    def test_parallax_ratio(self):
        # no natural vertical parallax on the centre lines
        self.assertAlmostEqual(v.parallax_ratio(EQ, E / 2 - 0.5, 100), 0.0, places=3)
        self.assertAlmostEqual(v.parallax_ratio(EQ, 100, E / 2 - 0.5), 0.0, places=3)
        self.assertAlmostEqual(v.parallax_ratio(FISH, FISH["cx"], 100), 0.0, places=3)
        # opposite signs in neighbouring quadrants, the same in opposite ones
        a = v.parallax_ratio(EQ, 150, 150)
        self.assertAlmostEqual(v.parallax_ratio(EQ, E - 1 - 150, E - 1 - 150), a, places=2)
        self.assertAlmostEqual(v.parallax_ratio(EQ, E - 1 - 150, 150), -a, places=2)
        self.assertGreater(abs(a), 0.05)

    def test_projection_scale(self):
        self.assertAlmostEqual(v.deg_per_px(v.eye_projection(v.DOME, None, 1024, 1024)),
                               180 / 1024)
        self.assertAlmostEqual(v.deg_per_px(v.eye_projection(v.FISHEYE, v.MKX200, 1024, 1024)),
                               100 / 512)
        self.assertAlmostEqual(v.deg_per_px(v.eye_projection(v.FISHEYE, None, 1024, 1024)),
                               95 / 512)
        self.assertIsNone(v.eye_projection(v.SPHERE, None, 1024, 512))

    def test_sizes(self):
        self.assertEqual(v.voffset_size(8192, 4096, v.SBS), (2048, 1024))
        self.assertEqual(v.voffset_size(5400, 2700, v.SBS), (2048, 1024))
        self.assertEqual(v.voffset_size(4096, 8192, v.TB), (1024, 2048))
        self.assertIsNone(v.voffset_size(8192, 4096, v.MONO))
        self.assertEqual(v.voffset_layout(8192, 4096), v.SBS)
        self.assertEqual(v.voffset_layout(4096, 8192), v.TB)

    def test_alignment_input(self):
        self.assertEqual(v.alignment_input({v.FISHEYE, v.MKX200, v.SBS, v.RL}),
                         (v.SBS, v.FISHEYE, v.MKX200, True))
        self.assertEqual(v.alignment_input({v.DOME, v.TB})[:2], (v.TB, v.DOME))
        for names in ({v.DOME, v.MONO}, {v.SPHERE, v.TB}, {v.FLAT, v.SBS}):
            self.assertEqual(v.alignment_input(names)[:2], (None, None), names)
        self.assertIsNone(v.alignment_input({v.DOME})[0])


F8K = {"width": 8192, "height": 4096, "duration": 1000, "path": "/media/VR/S/x.mp4"}


class MeasureAlignment(unittest.TestCase):
    """The decode schedule, with frames standing in for fits."""

    def run_it(self, fits, names=(v.DOME, v.SBS), cached=None, f=F8K):
        grabbed = []

        def grab(c, path, ts, size):
            grabbed.append(round(ts / f["duration"], 2))
            return None if fits.get(grabbed[-1]) == "fail" else b"%.2f" % grabbed[-1]

        def frame(buf, fw, fh, stereo, rl, screen, lens):
            return fits[float(buf)]
        with mock.patch.object(v.os.path, "exists", return_value=True), \
                mock.patch.object(v, "grab_grey", side_effect=grab), \
                mock.patch.object(v, "frame_voffset", side_effect=frame):
            got = v.measure_alignment(cfg(), f, set(names), cached)
        return got, grabbed

    def test_four_spread_frames(self):
        fits = {0.4: fit(-4.8), 0.6: fit(-4.7), 0.2: fit(-4.9), 0.8: fit(-4.8)}
        got, grabbed = self.run_it(fits)
        self.assertEqual(grabbed, [0.4, 0.6, 0.2, 0.8])
        self.assertEqual(got["deg"], -0.84)

    def test_the_projection_frames_are_reused(self):
        fits = {0.4: fit(1.0), 0.6: fit(1.0), 0.2: fit(1.0), 0.8: fit(1.0)}
        cached = {"size": (2048, 1024), "frames": {0.4: b"0.40", 0.6: b"0.60"}}
        got, grabbed = self.run_it(fits, cached=cached)
        self.assertEqual(grabbed, [0.2, 0.8])
        self.assertEqual(got["deg"], 0.18)
        # decoded for another layout: not used
        _, grabbed = self.run_it(fits, cached=dict(cached, size=(1024, 2048)))
        self.assertEqual(grabbed, [0.4, 0.6, 0.2, 0.8])

    def test_stand_in_frames_only_one_short(self):
        bad = {"ok": False, "n": 4}
        fits = {0.4: fit(1.0), 0.6: bad, 0.2: fit(1.1), 0.8: bad, 0.3: bad, 0.7: fit(0.9)}
        got, grabbed = self.run_it(fits)
        self.assertEqual(grabbed, [0.4, 0.6, 0.2, 0.8, 0.3, 0.7])
        self.assertEqual(got["deg"], 0.18)
        fits = {0.4: fit(1.0), 0.6: bad, 0.2: bad, 0.8: bad, 0.3: fit(1.0), 0.7: fit(1.0)}
        got, grabbed = self.run_it(fits)
        self.assertEqual(grabbed, [0.4, 0.6, 0.2, 0.8])     # two short: hopeless
        self.assertIsNone(got["deg"])

    def test_stops_at_disagreement(self):
        fits = {0.4: fit(2.0), 0.6: fit(-2.0), 0.2: fit(2.0), 0.8: fit(2.0)}
        got, grabbed = self.run_it(fits)
        self.assertEqual(grabbed, [0.4, 0.6])
        self.assertIsNone(got["deg"])
        self.assertIn("disagree", got["why"])

    def test_nothing_decoded_leaves_the_field(self):
        fits = {k: "fail" for k in v.VOFFSET_FRACS}
        self.assertEqual(self.run_it(fits)[0], {})

    def test_not_measured(self):
        got, grabbed = self.run_it({}, names=(v.DOME, v.MONO))
        self.assertEqual((got["deg"], grabbed), (None, []))
        got, grabbed = self.run_it({}, names=(v.SPHERE, v.TB))
        self.assertEqual((got["deg"], grabbed), (None, []))
        self.assertEqual(self.run_it({}, names=(v.UNRESOLVED,))[0], {})
        with mock.patch.object(v.os.path, "exists", return_value=False):
            self.assertEqual(v.measure_alignment(cfg(), F8K, {v.DOME, v.SBS}), {})

    def test_fisheye_lens_and_eye_order_reach_the_frame(self):
        seen = []

        def frame(buf, fw, fh, stereo, rl, screen, lens):
            seen.append((fw, fh, stereo, rl, screen, lens))
            return fit(1.0)
        with mock.patch.object(v.os.path, "exists", return_value=True), \
                mock.patch.object(v, "grab_grey", return_value=b"x"), \
                mock.patch.object(v, "frame_voffset", side_effect=frame):
            got = v.measure_alignment(cfg(), F8K, {v.FISHEYE, v.MKX200, v.SBS, v.RL})
        self.assertEqual(seen[0], (2048, 1024, v.SBS, True, v.FISHEYE, v.MKX200))
        self.assertEqual(got["deg"], 0.2)                   # 1 px at 100 / 512 deg per px


class Decode(unittest.TestCase):
    def test_one_decode_gives_thumbnail_crop_and_stereo_frame(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            with open(cmd[cmd.index("[c]") + 7], "wb") as f:
                f.write(b"\x05" * (512 * 512))
            with open(cmd[-1], "wb") as f:
                f.write(b"\x07" * (2048 * 1024))
            return mock.Mock(returncode=0, stdout=b"\x01" * (256 * 128 * 3))
        with mock.patch.object(v.subprocess, "run", side_effect=run):
            yuv, crop, big = v.grab_multi(cfg(), "/m/x.mp4", 12.0, 256, 128,
                                          (0, 0, 512, 512), (2048, 1024))
        self.assertEqual(len(calls), 1)
        graph = calls[0][calls[0].index("-filter_complex") + 1]
        self.assertIn("split=3", graph)
        self.assertIn("scale=2048:1024:flags=area,format=gray", graph)
        self.assertEqual((len(yuv), crop, big),
                         (256 * 128 * 3, b"\x05" * (512 * 512), b"\x07" * (2048 * 1024)))
        self.assertFalse(os.path.exists(calls[0][-1]))

    def test_grab_grey(self):
        with mock.patch.object(v.subprocess, "run",
                               return_value=mock.Mock(returncode=0, stdout=b"\x02" * 8)) as run:
            self.assertEqual(v.grab_grey(cfg(), "/m/x.mp4", 3.0, (4, 2)), b"\x02" * 8)
        cmd = run.call_args[0][0]
        self.assertIn("scale=4:2:flags=area,format=gray", cmd)
        self.assertIn("nokey", cmd)
        with mock.patch.object(v.subprocess, "run",
                               return_value=mock.Mock(returncode=1, stdout=b"")):
            self.assertIsNone(v.grab_grey(cfg(), "/m/x.mp4", 3.0, (4, 2)))

    def test_probe_keeps_the_stereo_frames(self):
        tw, th = v.thumb_size(8192, 4096)
        yuv = v.rgb_to_grey(synth.equirect_sbs(tw, th)) + bytes([128]) * (2 * tw * th)
        with mock.patch.object(v, "grab_multi", return_value=(yuv, None, b"big")) as g:
            stereo = {}
            res = v.probe(cfg(), "/m/x.mp4", 8192, 4096, 600, stereo=stereo)
        self.assertEqual(g.call_count, 2)
        self.assertEqual(g.call_args[0][6], (2048, 1024))
        self.assertEqual(stereo, {"size": (2048, 1024), "layout": v.SBS,
                                  "frames": {0.4: b"big", 0.6: b"big"}})
        self.assertEqual(v.classify(8192, 4096, res)[:2], (v.DOME, v.SBS))


# ------------------------------------------------------------ the custom field

class FieldStash:
    """A fake Stash that holds custom fields and tags of its scenes."""

    def __init__(self, fields=None, supported=True):
        self.fields = dict(fields or {})
        self.writes = []
        self.supported = supported

    def call(self, query, variables=None):
        if "custom_fields" in query and "findScenes" in query:
            if not self.supported:
                raise RuntimeError("Cannot query field \"custom_fields\" on type \"Scene\".")
            return {"findScenes": {"scenes": []}}
        if "findScene(" in query and "custom_fields" in query:
            return {"findScene": {"custom_fields": dict(self.fields)}}
        if "sceneUpdate" in query:
            i = variables["i"]
            self.writes.append(i)
            cf = i.get("custom_fields") or {}
            self.fields.update(cf.get("partial") or {})
            for k in cf.get("remove") or ():
                self.fields.pop(k, None)
            return {"sceneUpdate": {"id": i["id"]}}
        raise AssertionError(query)


def scene(tags=(v.DOME, v.SBS), sid="7"):
    return {"id": sid, "files": [dict(F8K, bit_rate=60e6, size=1)],
            "tags": [{"id": f"id:{n}", "name": n} for n in tags]}


IDS = {n: f"id:{n}" for n in v.PROJECTION_TAGS + (v.SKIP, v.LOW_DETAIL)
       + v.quality_names(cfg())}


class FieldUpdate(unittest.TestCase):
    def test_update(self):
        self.assertEqual(v.voffset_update({}, -0.84), {"partial": {v.VOFFSET_FIELD: -0.84}})
        self.assertEqual(v.voffset_update({v.VOFFSET_FIELD: -0.5, "other": 1}, -0.84),
                         {"partial": {v.VOFFSET_FIELD: -0.84}})
        self.assertIsNone(v.voffset_update({v.VOFFSET_FIELD: -0.84}, -0.84))
        self.assertIsNone(v.voffset_update({v.VOFFSET_FIELD: 0}, 0.0))
        self.assertEqual(v.voffset_update({v.VOFFSET_FIELD: "junk"}, 0.1),
                         {"partial": {v.VOFFSET_FIELD: 0.1}})
        self.assertEqual(v.voffset_update({v.VOFFSET_FIELD: 0.3, "other": 1}, None),
                         {"remove": [v.VOFFSET_FIELD]})
        self.assertIsNone(v.voffset_update({"other": 1}, None))
        self.assertIsNone(v.voffset_update(None, None))


class AlignmentTask(unittest.TestCase):
    def run_scene(self, sc, fields, result):
        stash = FieldStash(fields)
        with mock.patch.object(v, "measure_alignment", return_value=result) as m:
            out = v.process_alignment(stash, cfg(_custom_fields=True), sc, IDS, "alignment")
        return stash, out, m

    def test_writes_only_its_field(self):
        stash, out, m = self.run_scene(scene(), {"other": "keep"}, {"deg": -0.84, "why": "w"})
        self.assertEqual(stash.writes, [{"id": "7", "custom_fields":
                                         {"partial": {v.VOFFSET_FIELD: -0.84}}}])
        self.assertEqual(stash.fields, {"other": "keep", v.VOFFSET_FIELD: -0.84})
        self.assertIn("-0.84", out)
        self.assertEqual(m.call_args[0][2], {v.DOME, v.SBS})

    def test_idempotent(self):
        stash, out, _ = self.run_scene(scene(), {v.VOFFSET_FIELD: -0.84}, {"deg": -0.84, "why": ""})
        self.assertEqual((stash.writes, out), ([], None))

    def test_unknown_removes_a_stored_value(self):
        stash, out, _ = self.run_scene(scene(), {v.VOFFSET_FIELD: 0.5, "other": 2},
                                       {"deg": None, "why": "frames disagree"})
        self.assertEqual(stash.writes[0]["custom_fields"], {"remove": [v.VOFFSET_FIELD]})
        self.assertEqual(stash.fields, {"other": 2})
        self.assertIn("removed", out)
        stash, out, _ = self.run_scene(scene(), {"other": 2}, {"deg": None, "why": ""})
        self.assertEqual((stash.writes, out), ([], None))

    def test_nothing_known_leaves_it(self):
        stash, out, _ = self.run_scene(scene(), {v.VOFFSET_FIELD: 0.5}, {})
        self.assertEqual((stash.writes, out), ([], None))

    def test_mono_scene_loses_the_field(self):
        stash = FieldStash({v.VOFFSET_FIELD: 0.5})
        v.process_alignment(stash, cfg(_custom_fields=True), scene((v.DOME, v.MONO)), IDS,
                            "alignment")
        self.assertEqual(stash.writes[0]["custom_fields"], {"remove": [v.VOFFSET_FIELD]})

    def test_skipped_and_unclassified_scenes_are_left_alone(self):
        for tags in ((v.DOME, v.SBS, v.SKIP), ("8K",), ()):
            stash, out, m = self.run_scene(scene(tags), {v.VOFFSET_FIELD: 0.5},
                                           {"deg": None, "why": ""})
            self.assertEqual((stash.writes, out), ([], None))
            m.assert_not_called()


class SceneScan(unittest.TestCase):
    """The alignment as part of the normal measurement."""

    def run_scene(self, sc, fields=None, mode="untagged", deg=-0.84, **kw):
        stash = FieldStash(fields)
        c = cfg(_custom_fields=True, **kw)
        calls = []

        def measure(cc, s, detail=None, align=None):
            calls.append(align)
            return {v.DOME, v.SBS}, "why"
        with mock.patch.object(v, "measure_projection", side_effect=measure), \
                mock.patch.object(v, "measure_alignment",
                                  return_value={"deg": deg, "why": "w"}) as ma:
            out = v.process_scene(stash, c, sc, IDS, mode)
        return stash, out, calls, ma

    def test_new_scene_gets_tags_and_field_in_one_write(self):
        stash, out, calls, ma = self.run_scene(scene(()), {"other": 1})
        self.assertEqual(len(stash.writes), 1)
        w = stash.writes[0]
        self.assertEqual(names(w["tag_ids"]) & {v.DOME, v.SBS}, {v.DOME, v.SBS})
        self.assertEqual(w["custom_fields"], {"partial": {v.VOFFSET_FIELD: -0.84}})
        self.assertEqual(calls, [{}])               # the frames dict handed to the probe
        self.assertEqual(ma.call_args[0][2], {v.DOME, v.SBS})
        self.assertIn(v.VOFFSET_FIELD, out)

    def test_only_the_field_changes(self):
        sc = scene((v.DOME, v.SBS, "8K", "HQ"))
        stash, out, _, _ = self.run_scene(sc, {v.VOFFSET_FIELD: -0.5}, mode="retag")
        self.assertEqual(stash.writes, [{"id": "7", "custom_fields":
                                         {"partial": {v.VOFFSET_FIELD: -0.84}}}])
        stash, out, _, _ = self.run_scene(sc, {v.VOFFSET_FIELD: -0.84}, mode="retag")
        self.assertEqual((stash.writes, out), ([], None))

    def test_settled_scene_is_not_measured(self):
        stash, _, calls, ma = self.run_scene(scene((v.DOME, v.SBS, "8K", "HQ")),
                                             {v.VOFFSET_FIELD: 0.4})
        self.assertEqual((calls, stash.writes), ([], []))
        ma.assert_not_called()

    def test_off_leaves_values_alone(self):
        stash, _, calls, ma = self.run_scene(scene(()), {v.VOFFSET_FIELD: 0.4},
                                             detectVerticalOffset=False)
        self.assertEqual(calls, [None])
        ma.assert_not_called()
        self.assertNotIn("custom_fields", stash.writes[0])
        self.assertEqual(stash.fields, {v.VOFFSET_FIELD: 0.4})

    def test_clear_removes_the_field(self):
        stash, _, _, _ = self.run_scene(scene((v.DOME, v.SBS)), {v.VOFFSET_FIELD: 0.4},
                                        mode="clear")
        self.assertEqual(stash.writes[0]["custom_fields"], {"remove": [v.VOFFSET_FIELD]})

    def test_hook_ignores_the_field_write(self):
        self.assertTrue(v.hook_should_skip({"type": "Scene.Update.Post",
                                            "inputFields": ["id", "custom_fields"]}))
        self.assertTrue(v.hook_should_skip({"type": "Scene.Update.Post",
                                            "inputFields": ["id", "tag_ids", "custom_fields"]}))
        self.assertFalse(v.hook_should_skip({"type": "Scene.Update.Post",
                                             "inputFields": ["id", "files"]}))


def names(tag_ids):
    return {i.split(":", 1)[1] for i in tag_ids}


class Routing(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def test_run_all_uses_the_alignment_handler_for_measured_scenes_only(self):
        vr, flat = scene(sid="1"), scene(sid="2")
        todo = [(vr, v.process_scene, "VR"), (flat, v.process_flat_scene, "flat 3D")]
        with mock.patch.object(v, "candidates", return_value=todo), \
                mock.patch.object(v, "process_alignment", return_value=None) as pa, \
                mock.patch.object(v, "log"):
            v.run_all(FieldStash(), cfg(), IDS, "alignment")
        self.assertEqual([c[0][2]["id"] for c in pa.call_args_list], ["1"])

    def run_main(self, mode="alignment", supported=True, **kw):
        payload = {"server_connection": {"PluginDir": self.dir.name}, "args": {"mode": mode}}
        stash = FieldStash(supported=supported)
        with mock.patch.object(v, "Stash", return_value=stash), \
                mock.patch.object(v, "ensure_tags", return_value=IDS), \
                mock.patch.object(v, "load_config", return_value=cfg(**kw)), \
                mock.patch.object(v, "run_all") as ra, \
                mock.patch.object(v, "tidy_mono"), \
                mock.patch.object(v, "log") as log, mock.patch("builtins.print"), \
                mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
            v.main()
        return ra, log

    def test_task_is_resumable_with_its_own_state(self):
        path = os.path.join(self.dir.name, v.ALIGNMENT_STATE_FILE)
        with open(path, "w") as f:
            json.dump({"started": time.time() - 60, "last_id": 12}, f)
        ra, _ = self.run_main()
        (stash, c, ids, mode, state), _ = ra.call_args
        self.assertEqual(mode, "alignment")
        self.assertEqual((state.path, state.last_id), (path, 12))
        self.assertTrue(c["_custom_fields"])

    def test_task_does_nothing_when_off_or_unsupported(self):
        ra, log = self.run_main(detectVerticalOffset=False)
        ra.assert_not_called()
        ra, log = self.run_main(supported=False)
        ra.assert_not_called()
        self.assertTrue(any("custom fields" in c[0][1] for c in log.call_args_list))

    def test_scene_tasks_measure_it_only_where_supported(self):
        ra, _ = self.run_main("untagged")
        self.assertTrue(ra.call_args[0][1]["_custom_fields"])
        ra, _ = self.run_main("untagged", supported=False)
        self.assertFalse(ra.call_args[0][1]["_custom_fields"])


if __name__ == "__main__":
    unittest.main()
