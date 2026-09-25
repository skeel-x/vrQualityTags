"""Honest resolution: the detail measure and the Low Detail tag."""
import cmath
import math
import os
import unittest
from unittest import mock

import vrQualityTags as v
from tests import synth

W, H = 1024, 512            # two analysis blocks side by side


def cfg(**kw):
    c = v.load_config({})
    c.update(kw)
    return c


def f8k(mbit=60, w=8192, h=4096):
    return {"width": w, "height": h, "bit_rate": mbit * 1e6, "size": 9, "duration": 600,
            "path": "/media/VR/S/x.mp4"}


def scene(files=None, tags=(), sid="7"):
    return {"id": sid, "files": [f8k()] if files is None else files,
            "tags": [{"id": f"id:{n}", "name": n} for n in tags]}


IDS = {n: f"id:{n}" for n in v.PROJECTION_TAGS + (v.SKIP, v.LOW_DETAIL)
       + v.quality_names(cfg())}


def names(tag_ids):
    return {i.split(":", 1)[1] for i in tag_ids}


class FakeStash:
    def __init__(self, tags=()):
        self.writes, self.created = [], []
        self.tags = {n: {"id": f"id:{n}", "name": n, "parents": []} for n in tags}

    def call(self, query, variables=None):
        if "sceneUpdate" in query:
            self.writes.append(variables["i"])
            return {"sceneUpdate": {"id": variables["i"]["id"]}}
        if "findTags" in query:
            t = self.tags.get(variables["n"])
            return {"findTags": {"tags": [t] if t else []}}
        if "tagCreate" in query:
            name = variables["i"]["name"]
            self.created.append(name)
            self.tags[name] = {"id": f"id:{name}", "name": name, "parents": []}
            return {"tagCreate": dict(self.tags[name])}
        if "tagUpdate" in query:
            return {"tagUpdate": {"id": variables["i"]["id"]}}
        raise AssertionError(query)


class Fft(unittest.TestCase):
    def test_matches_a_plain_dft(self):
        x = [complex(math.sin(i * 0.7) + (i % 3), (i * 5) % 7) for i in range(16)]
        want = [sum(x[t] * cmath.exp(-2j * math.pi * k * t / 16) for t in range(16))
                for k in range(16)]
        for a, b in zip(v.fft(x), want):
            self.assertAlmostEqual(a, b, places=9)


class DetailMeasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the same lens softness on a picture sampled at the file's size and
        # on one sampled at half the size and upscaled 2x
        cls.native = synth.soften(synth.grey_noise(W, H, 0), W, H)
        cls.upscaled = synth.upscale2(synth.soften(synth.grey_noise(W // 2, H // 2, 1),
                                                   W // 2, H // 2), W // 2, H // 2)

    def detail(self, buf):
        return v.frame_detail(v.detail_blocks(buf, W, H), 8192)

    def test_sharp_noise_keeps_detail(self):
        d = v.frame_detail(v.detail_blocks(synth.grey_noise(W, H), W, H), 8192)
        self.assertGreater(d["ratio"], 0.7)
        self.assertGreater(d["eff"], 7000)
        self.assertEqual(d["blocks"], 2)

    def test_native_versus_upscaled(self):
        native, upscaled = self.detail(self.native), self.detail(self.upscaled)
        self.assertGreater(native["ratio"], 3 * v.DETAIL_MIN_RATIO)
        self.assertLess(upscaled["ratio"], v.DETAIL_MIN_RATIO / 2)
        # an upscale from half the size has about half the effective width
        self.assertLess(upscaled["eff"], 0.5 * 8192)
        self.assertGreater(native["eff"], 0.6 * 8192)

    def test_low_texture_is_unknown(self):
        flat = bytes(128 + (x * 7 + y * 13) % 3 - 1 for y in range(H) for x in range(W))
        self.assertEqual(v.detail_blocks(flat, W, H), [])
        self.assertIsNone(v.frame_detail([], 8192))
        dark = bytes(n // 16 for n in synth.grey_noise(W, H))     # sharp but black
        self.assertEqual(v.detail_blocks(dark, W, H), [])

    def test_a_verdict_needs_two_textured_blocks(self):
        half = bytearray(self.native)
        for y in range(H):
            half[y * W + 512:(y + 1) * W] = bytes([128]) * 512
        one = v.detail_blocks(bytes(half), W, H)
        self.assertEqual(len(one), 1)
        self.assertIsNone(v.detail_verdict([one, []], 8192))
        # one textured block in each frame is enough
        got = v.detail_verdict([one, one], 8192)
        self.assertEqual(got["blocks"], 2)
        self.assertAlmostEqual(got["ratio"], v.frame_detail(one, 8192)["ratio"])

    def test_sharpest_frame_counts(self):
        soft = v.detail_blocks(self.upscaled, W, H)
        sharp = v.detail_blocks(self.native, W, H)
        got = v.detail_verdict([soft, sharp], 8192)
        self.assertEqual(got, dict(v.frame_detail(sharp, 8192), blocks=4))
        self.assertEqual(v.detail_verdict([soft, []], 8192), v.frame_detail(soft, 8192))
        # a single sharp block proves the detail, whatever the other frame says
        sharp_one = sharp[:1]
        self.assertEqual(v.detail_verdict([soft, sharp_one], 8192)["ratio"],
                         v.frame_detail(sharp_one, 8192)["ratio"])
        self.assertIsNone(v.detail_verdict([[], []], 8192))
        self.assertIsNone(v.detail_verdict([], 8192))


class DetailBox(unittest.TestCase):
    def test_centre_of_left_eye(self):
        self.assertEqual(v.detail_box(8192, 4096), (1536, 1536, 1024, 1024))
        self.assertEqual(v.detail_box(5760, 2880), (928, 928, 1024, 1024))

    def test_top_half_of_a_square_frame(self):
        self.assertEqual(v.detail_box(4096, 4096), (1536, 512, 1024, 1024))

    def test_small_eye(self):
        self.assertEqual(v.detail_box(1600, 800), (144, 144, 512, 512))
        self.assertIsNone(v.detail_box(800, 400))


class BitrateFloor(unittest.TestCase):
    SHARP = {"ratio": 0.2, "eff": 6000, "blocks": 4}
    SOFT = {"ratio": 0.05, "eff": 4000, "blocks": 4}

    def test_bits_per_pixel(self):
        self.assertAlmostEqual(v.bits_per_pixel(f8k(60)), 60e6 / (8192 * 4096))
        self.assertIsNone(v.bits_per_pixel({"width": 8192, "height": 4096, "bit_rate": 0}))
        self.assertIsNone(v.bits_per_pixel({}))

    MIDDLING = {"ratio": 0.1, "eff": 5000, "blocks": 4}

    def test_floor_per_pixel(self):
        # 0.8 bit/px/s: 26.8 Mbit/s at 8192x4096, 23.6 at 7680x3840
        self.assertTrue(v.low_detail(f8k(26), self.MIDDLING)[0])
        self.assertFalse(v.low_detail(f8k(28), self.MIDDLING)[0])
        self.assertTrue(v.low_detail(f8k(23, 7680, 3840), self.MIDDLING)[0])
        self.assertFalse(v.low_detail(f8k(24, 7680, 3840), self.MIDDLING)[0])

    def test_floor_spares_a_sharp_file(self):
        # SLR Originals at 21.5 Mbit/s with a ratio of 0.18 keeps its detail
        self.assertFalse(v.low_detail(f8k(21.5), {"ratio": 0.18, "eff": 6000, "blocks": 4})[0])
        self.assertFalse(v.low_detail(f8k(20), self.SHARP)[0])
        self.assertTrue(v.low_detail(f8k(20), {"ratio": 0.119, "eff": 5000, "blocks": 4})[0])
        self.assertFalse(v.low_detail(f8k(20), {"ratio": 0.12, "eff": 5000, "blocks": 4})[0])

    def test_rule(self):
        low, why = v.low_detail(f8k(60), self.SOFT)
        self.assertTrue(low)
        self.assertIn("detail 0.050", why)
        self.assertIn("effective width ~4000", why)
        self.assertFalse(v.low_detail(f8k(60), self.SHARP)[0])
        # never tagged on unknown pixels, whatever the bitrate
        low, why = v.low_detail(f8k(60), None)
        self.assertIsNone(low)
        self.assertIn("unknown", why)
        self.assertIsNone(v.low_detail(f8k(20), None)[0])
        self.assertIsNone(v.low_detail({"width": 8192, "height": 4096}, None)[0])


class LowDetailTags(unittest.TestCase):
    def tags(self, sc=None, tier="tag8k", detail=None, **kw):
        return v.low_detail_tags(cfg(**kw), sc or scene(), tier, detail)

    def test_measured(self):
        scope, want, why = self.tags(detail={"verdict": {"ratio": 0.05, "eff": 4100, "blocks": 3}})
        self.assertEqual((scope, want), ({v.LOW_DETAIL}, {v.LOW_DETAIL}))
        self.assertIn("low detail", why)
        scope, want, _ = self.tags(detail={"verdict": {"ratio": 0.2, "eff": 6000, "blocks": 3}})
        self.assertEqual((scope, want), ({v.LOW_DETAIL}, set()))
        scope, want, why = self.tags(detail={"verdict": None})
        self.assertEqual((scope, want), ({v.LOW_DETAIL}, set()))
        self.assertIn("unknown", why)

    def test_not_measured(self):
        # nothing decoded: the tag is left alone, even at a low bitrate
        self.assertEqual(self.tags(detail={})[:2], (set(), set()))
        self.assertEqual(self.tags(detail=None)[:2], (set(), set()))
        starved = scene(files=[f8k(15)])
        self.assertEqual(self.tags(starved, detail=None)[:2], (set(), set()))

    def test_no_tier_or_off_removes(self):
        self.assertEqual(self.tags(tier=None)[:2], ({v.LOW_DETAIL}, set()))
        self.assertEqual(self.tags(detail={"verdict": {"ratio": 0.01, "eff": 1, "blocks": 2}},
                                   detectLowDetail=False)[:2], ({v.LOW_DETAIL}, set()))


class ProcessScene(unittest.TestCase):
    def run_scene(self, sc, verdict="unset", mode="untagged", ids=IDS, **kw):
        stash, seen = FakeStash(), []

        def measure(c, s, detail=None):
            seen.append(detail)
            if detail is not None and verdict != "unset":
                detail["verdict"] = verdict
            return {v.DOME, v.SBS}, "why"
        with mock.patch.object(v, "measure_projection", side_effect=measure), \
                mock.patch.object(v, "log"):
            out = v.process_scene(stash, cfg(**kw), sc, ids, mode)
        return stash, out, seen

    def test_soft_8k_gets_low_detail_with_its_tier(self):
        stash, out, seen = self.run_scene(scene(), {"ratio": 0.05, "eff": 4000, "blocks": 4})
        self.assertEqual(seen, [{"verdict": {"ratio": 0.05, "eff": 4000, "blocks": 4}}])
        self.assertEqual(names(stash.writes[0]["tag_ids"]),
                         {"DOME", "SBS", "8K", "HQ", v.LOW_DETAIL})
        self.assertIn("low detail: detail 0.050", out)

    def test_sharp_file_loses_a_stale_tag(self):
        sc = scene(tags=["DOME", "SBS", "8K", "HQ", v.LOW_DETAIL])
        stash, _, _ = self.run_scene(sc, {"ratio": 0.2, "eff": 6000, "blocks": 4}, mode="retag")
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"DOME", "SBS", "8K", "HQ"})

    def test_settled_scene_keeps_its_verdict(self):
        sc = scene(tags=["DOME", "SBS", "8K", "HQ", v.LOW_DETAIL])
        stash, out, seen = self.run_scene(sc)
        self.assertEqual(seen, [])
        self.assertEqual(stash.writes, [])
        self.assertIsNone(out)

    def test_settled_starved_scene_is_not_tagged_unmeasured(self):
        sc = scene(files=[f8k(15)], tags=["DOME", "SBS", "8K", "HQ"])
        stash, _, seen = self.run_scene(sc)
        self.assertEqual(seen, [])
        self.assertEqual(stash.writes, [])

    def test_no_tier_no_detail(self):
        sc = scene(files=[f8k(60, 3840, 1920)], tags=[v.LOW_DETAIL])
        stash, _, seen = self.run_scene(sc)
        self.assertEqual(seen, [None])              # nothing to judge the detail of
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"DOME", "SBS"})

    def test_secondary_tier_file_is_not_measured(self):
        small, big = dict(f8k(60, 3840, 1920), size=1), dict(f8k(60), size=99)
        _, _, seen = self.run_scene(scene(files=[small, big]))
        self.assertEqual(seen, [None])

    def test_off(self):
        sc = scene(tags=[v.LOW_DETAIL])
        stash, _, seen = self.run_scene(sc, {"ratio": 0.01, "eff": 1, "blocks": 2},
                                        detectLowDetail=False)
        self.assertEqual(seen, [None])
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"DOME", "SBS", "8K", "HQ"})
        # off and the tag never existed: nothing to manage
        ids = {k: i for k, i in IDS.items() if k != v.LOW_DETAIL}
        stash, _, _ = self.run_scene(scene(), detectLowDetail=False, ids=ids)
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"DOME", "SBS", "8K", "HQ"})

    def test_clear_removes_it(self):
        sc = scene(tags=["DOME", "8K", "HQ", v.LOW_DETAIL, "Other"])
        stash, _, _ = self.run_scene(sc, mode="clear")
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"Other"})


class EnsureTags(unittest.TestCase):
    def test_created_when_on_looked_up_when_off(self):
        stash = FakeStash()
        with mock.patch.object(v, "log"):
            ids = v.ensure_tags(stash, cfg())
        self.assertIn(v.LOW_DETAIL, stash.created)
        self.assertIn(v.LOW_DETAIL, ids)

        stash = FakeStash()
        with mock.patch.object(v, "log"):
            ids = v.ensure_tags(stash, cfg(detectLowDetail=False))
        self.assertNotIn(v.LOW_DETAIL, stash.created)
        self.assertNotIn(v.LOW_DETAIL, ids)

        stash = FakeStash(tags=[v.LOW_DETAIL])
        with mock.patch.object(v, "log"):
            ids = v.ensure_tags(stash, cfg(detectLowDetail=False))
        self.assertEqual(ids[v.LOW_DETAIL], f"id:{v.LOW_DETAIL}")


class Decode(unittest.TestCase):
    def test_one_decode_gives_thumbnail_and_crop(self):
        box = (1536, 1536, 1024, 1024)
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            with open(cmd[-1], "wb") as f:
                f.write(b"\x05" * (1024 * 1024))
            return mock.Mock(returncode=0, stdout=b"\x01" * (256 * 128 * 3))
        with mock.patch.object(v.subprocess, "run", side_effect=run):
            yuv, crop = v.grab_with_crop(cfg(), "/m/x.mp4", 12.0, 256, 128, box)
        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        graph = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("crop=1024:1024:1536:1536", graph)
        self.assertIn("scale=256:128:out_range=pc", graph)
        self.assertIn("nokey", cmd)
        self.assertEqual(len(yuv), 256 * 128 * 3)
        self.assertEqual(crop, b"\x05" * (1024 * 1024))
        self.assertFalse(os.path.exists(cmd[-1]))      # temporary file removed

    def test_failed_decode(self):
        def run(cmd, **kw):
            return mock.Mock(returncode=1, stdout=b"")
        with mock.patch.object(v.subprocess, "run", side_effect=run):
            self.assertEqual(v.grab_with_crop(cfg(), "/m/x.mp4", 1.0, 256, 128,
                                              (0, 0, 512, 512)), (None, None))
        with mock.patch.object(v.subprocess, "run",
                               side_effect=v.subprocess.TimeoutExpired("ffmpeg", 300)):
            self.assertEqual(v.grab_with_crop(cfg(), "/m/x.mp4", 1.0, 256, 128,
                                              (0, 0, 512, 512)), (None, None))

    def test_probe_measures_detail_from_the_same_frames(self):
        tw, th = v.thumb_size(8192, 4096)
        thumb = synth.equirect_sbs(tw, th)
        yuv = v.rgb_to_grey(thumb) + bytes([128]) * (2 * tw * th)
        crop = synth.soften(synth.grey_noise(1024, 1024), 1024, 1024)
        with mock.patch.object(v, "grab_with_crop", return_value=(yuv, crop)) as g, \
                mock.patch.object(v, "grab") as plain:
            res = v.probe(cfg(), "/m/x.mp4", 8192, 4096, 600, detail=True)
        self.assertEqual(g.call_count, 2)
        plain.assert_not_called()
        self.assertEqual(g.call_args.args[5], (1536, 1536, 1024, 1024))
        self.assertGreater(res["detail"]["ratio"], v.DETAIL_MIN_RATIO)
        self.assertEqual(res["detail"]["blocks"], 8)       # 4 per frame
        self.assertEqual(v.classify(8192, 4096, res)[:2], (v.DOME, v.SBS))

    def test_probe_without_detail_is_unchanged(self):
        with mock.patch.object(v, "grab", return_value=None) as g, \
                mock.patch.object(v, "grab_with_crop") as both:
            self.assertIsNone(v.probe(cfg(), "/m/x.mp4", 8192, 4096, 600))
        self.assertEqual(g.call_count, 2)
        both.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class DetailTask(unittest.TestCase):
    def run_scene(self, sc, measured, **kw):
        stash, seen = FakeStash(), []

        def measure(c, f):
            seen.append(f)
            return measured
        with mock.patch.object(v, "measure_detail", side_effect=measure), \
                mock.patch.object(v, "measure_projection") as proj:
            out = v.process_detail(stash, cfg(**kw), sc, IDS, "detail")
        proj.assert_not_called()
        return stash, out, seen

    def test_soft_file_gets_low_detail_and_nothing_else_changes(self):
        sc = scene(tags=["FISHEYE", "SBS", "8K", "HQ", "Custom"])
        stash, out, seen = self.run_scene(sc, {"verdict": {"ratio": 0.05, "eff": 4000, "blocks": 4}})
        self.assertEqual(len(seen), 1)
        self.assertEqual(names(stash.writes[0]["tag_ids"]),
                         {"FISHEYE", "SBS", "8K", "HQ", "Custom", v.LOW_DETAIL})
        self.assertIn("added", out)

    def test_sharp_file_loses_a_stale_tag(self):
        sc = scene(tags=["DOME", "8K", v.LOW_DETAIL])
        stash, out, _ = self.run_scene(sc, {"verdict": {"ratio": 0.2, "eff": 6000, "blocks": 4}})
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"DOME", "8K"})
        self.assertIn("removed", out)

    def test_unchanged_or_unreadable_writes_nothing(self):
        sc = scene(tags=["DOME", "8K"])
        stash, out, _ = self.run_scene(sc, {"verdict": {"ratio": 0.2, "eff": 6000, "blocks": 4}})
        self.assertEqual((stash.writes, out), ([], None))
        sc = scene(tags=["DOME", "8K", v.LOW_DETAIL])
        stash, out, _ = self.run_scene(sc, {})     # nothing decoded: left alone
        self.assertEqual((stash.writes, out), ([], None))

    def test_no_tier_is_not_decoded_and_drops_the_tag(self):
        sc = scene(files=[f8k(60, 3840, 1920)], tags=["DOME", v.LOW_DETAIL])
        stash, _, seen = self.run_scene(sc, {"verdict": None})
        self.assertEqual(seen, [])
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"DOME"})

    def test_skip_and_secondary_file(self):
        stash, out, seen = self.run_scene(scene(tags=[v.SKIP, "8K"]), {})
        self.assertEqual((seen, stash.writes, out), ([], [], None))
        big = dict(f8k(), size=99)
        stash, _, seen = self.run_scene(scene(files=[f8k(), big], tags=["8K"]), {})
        self.assertEqual((seen, stash.writes), ([], []))

    def test_measure_detail(self):
        f = f8k()
        with mock.patch.object(v.os.path, "exists", return_value=False):
            self.assertEqual(v.measure_detail(cfg(), f), {})
        with mock.patch.object(v.os.path, "exists", return_value=True):
            self.assertEqual(v.measure_detail(cfg(), dict(f, width=800, height=400)),
                             {"verdict": None})
            with mock.patch.object(v, "grab_crop", return_value=None):
                self.assertEqual(v.measure_detail(cfg(), f), {})
            crop = bytes(1024 * 1024)                  # black: no textured blocks
            with mock.patch.object(v, "grab_crop", return_value=crop) as g:
                self.assertEqual(v.measure_detail(cfg(), f), {"verdict": None})
            self.assertEqual(g.call_count, 2)
            self.assertEqual(g.call_args[0][3], v.detail_box(8192, 4096))

    def test_run_all_uses_the_detail_handler_for_measured_scenes_only(self):
        vr, flat = scene(sid="1"), scene(sid="2")
        todo = [(vr, v.process_scene, "VR"), (flat, v.process_flat_scene, "flat 3D")]
        with mock.patch.object(v, "candidates", return_value=todo), \
                mock.patch.object(v, "process_detail", return_value=None) as pd, \
                mock.patch.object(v, "log"):
            v.run_all(FakeStash(), cfg(), IDS, "detail")
        self.assertEqual([c[0][2]["id"] for c in pd.call_args_list], ["1"])
