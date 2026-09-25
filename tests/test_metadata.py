import json
import os
import subprocess
import unittest
from unittest import mock

import vrQualityTags as v

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def cfg(**kw):
    c = v.load_config({})
    c.update(kw)
    return c


# measurements of a clear 180 side-by-side pair and of a clear mono picture
PAIR = {"lr": 0.9, "tb": 0.1}
MONO_PIC = {"lr": 0.1, "tb": 0.1}


class ParseFfprobe(unittest.TestCase):
    def test_equirect_sbs_as_recorded(self):
        # what files in the wild carry: equirect with no bounds, i.e. a claimed 360
        got = v.parse_ffprobe(fixture("ffprobe_equirect_sbs.json"))
        self.assertEqual((got["screen"], got["stereo"], got["rl"]), (v.SPHERE, v.SBS, False))
        self.assertNotIn("lens", got)
        self.assertEqual(got["raw"], "side by side, equirectangular 360deg")

    def test_2d_as_recorded(self):
        got = v.parse_ffprobe(fixture("ffprobe_equirect_2d.json"))
        self.assertEqual((got["screen"], got["stereo"]), (v.SPHERE, v.MONO))

    def test_nothing(self):
        self.assertIsNone(v.parse_ffprobe(fixture("ffprobe_none.json")))
        self.assertIsNone(v.parse_ffprobe({}))
        self.assertIsNone(v.parse_ffprobe(None))
        self.assertIsNone(v.parse_ffprobe({"streams": []}))

    def test_bounds_make_a_180(self):
        got = v.parse_ffprobe(fixture("ffprobe_tiled_180_rl.json"))
        self.assertEqual((got["screen"], got["stereo"], got["rl"]), (v.DOME, v.SBS, True))
        self.assertIn("180deg", got["raw"])

    def test_fisheye_with_fov_names_the_lens(self):
        got = v.parse_ffprobe(fixture("ffprobe_fisheye_fov.json"))
        self.assertEqual((got["screen"], got["lens"]), (v.FISHEYE, v.RF52))
        # "unspecified" stereo (MV-HEVC) says nothing about the layout
        self.assertNotIn("stereo", got)

    def test_fov_values(self):
        for fov, lens in (("200000/1000", v.MKX200), (220, v.MKX220), ("190.4", v.RF52),
                          ("180000/1000", None), ("x", None), ("1/0", None)):
            with self.subTest(fov=fov):
                data = {"streams": [{"width": 100, "side_data_list": [
                    {"side_data_type": "Stereo 3D", "type": "side by side",
                     "horizontal_field_of_view": fov}]}]}
                self.assertEqual(v.parse_ffprobe(data).get("lens"), lens)

    def test_fov_does_not_override_an_equirect(self):
        data = {"streams": [{"width": 100, "side_data_list": [
            {"side_data_type": "Spherical Mapping", "projection": "half equirectangular"},
            {"side_data_type": "Stereo 3D", "type": "2D", "horizontal_field_of_view": "200/1"}]}]}
        got = v.parse_ffprobe(data)
        self.assertEqual((got["screen"], got["stereo"]), (v.DOME, v.MONO))
        self.assertNotIn("lens", got)

    def test_matroska_stereo_mode(self):
        got = v.parse_ffprobe(fixture("ffprobe_mkv_tb.json"))
        self.assertEqual((got["stereo"], got["rl"]), (v.TB, False))
        self.assertNotIn("screen", got)
        for mode, want in (("right_left", (v.SBS, True)), ("left_right", (v.SBS, False)),
                           ("mono", (v.MONO, False)), ("bottom_top", (v.TB, False))):
            with self.subTest(mode=mode):
                got = v.parse_ffprobe({"streams": [{"tags": {"stereo_mode": mode}}]})
                self.assertEqual((got["stereo"], got["rl"]), want)
        self.assertIsNone(v.parse_ffprobe({"streams": [{"tags": {"stereo_mode": "block_lr"}}]}))

    def test_side_data_beats_the_tag(self):
        data = {"streams": [{"tags": {"stereo_mode": "top_bottom"}, "side_data_list": [
            {"side_data_type": "Stereo 3D", "type": "side by side", "inverted": 0}]}]}
        self.assertEqual(v.parse_ffprobe(data)["stereo"], v.SBS)

    def test_other_projections_are_ignored(self):
        for proj in ("cubemap", "rectilinear", "parametric immersive"):
            with self.subTest(proj=proj):
                data = {"streams": [{"side_data_list": [
                    {"side_data_type": "Spherical Mapping", "projection": proj}]}]}
                self.assertIsNone(v.parse_ffprobe(data))

    def test_fisheye_projection(self):
        data = {"streams": [{"side_data_list": [
            {"side_data_type": "Spherical Mapping", "projection": "fisheye"}]}]}
        self.assertEqual(v.parse_ffprobe(data)["screen"], v.FISHEYE)

    def test_unrelated_side_data(self):
        data = {"streams": [{"side_data_list": [
            {"side_data_type": "Display Matrix", "rotation": 0}]}]}
        self.assertIsNone(v.parse_ffprobe(data))


class ReadMetadata(unittest.TestCase):
    def run_with(self, **kw):
        done = subprocess.CompletedProcess([], kw.pop("rc", 0), kw.pop("out", b""), b"")
        with mock.patch.object(v.subprocess, "run", return_value=done, **kw) as run:
            got = v.read_metadata(cfg(ffprobePath="/x/ffprobe"), "/m/a.mp4")
        return got, run

    def test_one_call(self):
        out = json.dumps(fixture("ffprobe_equirect_sbs.json")).encode()
        got, run = self.run_with(out=out)
        self.assertEqual(got["stereo"], v.SBS)
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[0], "/x/ffprobe")
        self.assertEqual(cmd[-1], "/m/a.mp4")
        self.assertIn("stream_side_data", " ".join(cmd))
        run.assert_called_once()

    def test_failures_are_silent(self):
        self.assertIsNone(self.run_with(rc=1)[0])
        self.assertIsNone(self.run_with(out=b"not json")[0])
        self.assertIsNone(self.run_with(out=b"")[0])
        self.assertIsNone(self.run_with(side_effect=FileNotFoundError())[0])
        self.assertIsNone(self.run_with(
            side_effect=subprocess.TimeoutExpired("ffprobe", 60))[0])


class VetClaim(unittest.TestCase):
    def test_recorded_metadata_on_a_180_pair(self):
        # equirect without bounds says 360, but a 2:1 side-by-side frame holds
        # two square 180 eyes: the stereo stands, the 360 does not
        claim = v.parse_ffprobe(fixture("ffprobe_equirect_sbs.json"))
        got, dropped = v.vet_claim(claim, 8192, 4096, PAIR, v.DOME, v.SBS)
        self.assertEqual(got, {"stereo": v.SBS, "rl": False})
        self.assertEqual(len(dropped), 1)
        self.assertIn("SPHERE", dropped[0])

    def test_recorded_2d_on_a_stereo_pair(self):
        claim = v.parse_ffprobe(fixture("ffprobe_equirect_2d.json"))
        got, dropped = v.vet_claim(claim, 8192, 4096, PAIR, v.DOME, v.SBS)
        self.assertEqual(got, {})
        self.assertIn("mono, but the halves match as a pair", dropped)

    def test_a_consistent_claim_stands(self):
        claim = v.parse_ffprobe(fixture("ffprobe_tiled_180_rl.json"))
        got, dropped = v.vet_claim(claim, 5760, 2880, PAIR, v.DOME, v.SBS)
        self.assertEqual(got, {"stereo": v.SBS, "rl": True, "screen": v.DOME})
        self.assertEqual(dropped, [])

    def test_360_mono_claim_beats_a_weak_pair_reading(self):
        # the pixels read a mono 2:1 frame without a closing seam as a 180
        # pair; metadata that says mono 360 fits the frame and stands
        claim = {"screen": v.SPHERE, "stereo": v.MONO}
        got, _ = v.vet_claim(claim, 4096, 2048, {"lr": 0.4, "tb": 0.2}, v.DOME, v.SBS)
        self.assertEqual(got, {"stereo": v.MONO, "rl": False, "screen": v.SPHERE})

    def test_layout_must_leave_an_eye_of_the_projection(self):
        # mono on a 2:1 frame whose weak pair the pixels read as a 180: a
        # mono 2:1 eye is no 180 eye, so the claim goes
        got, dropped = v.vet_claim({"stereo": v.MONO}, 8192, 4096, {"lr": 0.4, "tb": 0.1},
                                   v.DOME, v.SBS)
        self.assertEqual(got, {})
        self.assertIn("leaves no DOME eye", dropped[0])
        # top/bottom on a square frame fits a 360 eye
        got, _ = v.vet_claim({"stereo": v.TB}, 4096, 4096, {"lr": 0.1, "tb": 0.8},
                             v.SPHERE, v.TB)
        self.assertEqual(got, {"stereo": v.TB, "rl": False})

    def test_coarse_shape_must_agree(self):
        got, dropped = v.vet_claim({"screen": v.DOME}, 8192, 4096, PAIR, v.FISHEYE, v.SBS)
        self.assertEqual(got, {})
        self.assertEqual(dropped, ["DOME, but the frame is FISHEYE"])
        got, _ = v.vet_claim({"screen": v.FISHEYE, "lens": v.MKX200}, 8192, 4096, PAIR,
                             v.DOME, v.SBS)
        self.assertEqual(got, {})
        got, _ = v.vet_claim({"screen": v.DOME}, 3840, 2160, MONO_PIC, v.FLAT, v.MONO)
        self.assertEqual(got, {})

    def test_lens_comes_with_an_accepted_fisheye(self):
        claim = {"screen": v.FISHEYE, "lens": v.RF52}
        got, _ = v.vet_claim(claim, 8192, 4096, PAIR, v.FISHEYE, v.SBS)
        self.assertEqual(got, {"screen": v.FISHEYE, "lens": v.RF52})

    def test_unrelated_halves_are_not_a_pair(self):
        got, dropped = v.vet_claim({"stereo": v.TB}, 4096, 4096, MONO_PIC, v.SPHERE, v.MONO)
        self.assertEqual(got, {})
        self.assertIn("halves have nothing in common", dropped[0])

    def test_unresolved_frame_goes_by_eye_shape(self):
        got, _ = v.vet_claim({"screen": v.DOME, "stereo": v.SBS}, 8192, 4096, None, None, None)
        self.assertEqual(got["screen"], v.DOME)
        got, _ = v.vet_claim({"screen": v.SPHERE, "stereo": v.TB}, 4096, 2048, None, None, None)
        self.assertEqual(got["screen"], v.SPHERE)     # squeezed 360 top/bottom
        got, dropped = v.vet_claim({"screen": v.SPHERE}, 1920, 1080, None, None, None)
        self.assertEqual(got, {})
        self.assertTrue(dropped)

    def test_empty(self):
        self.assertEqual(v.vet_claim(None, 1, 1, None, None, None), (None, []))


class Authority(unittest.TestCase):
    NONE = v.parse_filename("plain.mp4")

    def test_metadata_beats_filename(self):
        fn = v.parse_filename("x_LR_180.mp4")
        meta = {"screen": v.SPHERE, "stereo": v.TB, "rl": False}
        self.assertEqual(v.resolve(fn, v.SPHERE, v.TB, False, None, meta), {v.SPHERE, v.TB})

    def test_metadata_rl(self):
        meta = {"stereo": v.SBS, "rl": True}
        self.assertEqual(v.resolve(self.NONE, v.DOME, v.SBS, False, None, meta),
                         {v.DOME, v.SBS, v.RL})
        # the source that decides the layout decides the eye order
        fn = v.parse_filename("x_RL_180.mp4")
        self.assertEqual(v.resolve(fn, v.DOME, v.SBS, False, None, {"stereo": v.SBS, "rl": False}),
                         {v.DOME, v.SBS})

    def test_metadata_lens_beats_watermark_and_filename_lens_fills_in(self):
        meta = {"screen": v.FISHEYE, "lens": v.MKX220}
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, ("190", False), meta),
                         {v.FISHEYE, v.MKX220, v.SBS})
        fn = v.parse_filename("x_MKX200.mp4")
        self.assertEqual(v.resolve(fn, v.FISHEYE, v.SBS, False, None, {"screen": v.FISHEYE}),
                         {v.FISHEYE, v.MKX200, v.SBS})

    def test_metadata_screen_drops_a_filename_lens(self):
        fn = v.parse_filename("x_MKX200.mp4")
        self.assertEqual(v.resolve(fn, v.DOME, v.SBS, False, None, {"screen": v.DOME}),
                         {v.DOME, v.SBS})

    def test_metadata_rescues_an_unresolved_frame(self):
        meta = {"screen": v.DOME, "stereo": v.SBS, "rl": False}
        self.assertEqual(v.resolve(self.NONE, None, None, False, None, meta), {v.DOME, v.SBS})

    def test_watermark_skipped_when_metadata_names_the_lens(self):
        path = "/m/SLR/x.mp4"
        fn = v.parse_filename(path)
        self.assertEqual(v.fov_skip_reason(fn, v.FISHEYE, False, path,
                                           {"screen": v.FISHEYE, "lens": v.RF52}),
                         "lens from metadata")
        self.assertEqual(v.fov_skip_reason(fn, v.FISHEYE, False, path, {"screen": v.DOME}),
                         "not fisheye")
        self.assertIsNone(v.fov_skip_reason(fn, v.FISHEYE, False, path, {"stereo": v.SBS}))


class MeasureWithMetadata(unittest.TestCase):
    RES = {"lr": 0.9, "tb": 0.1, "blk_out": 0.1, "blk_in": 0.0, "bbox": 1.3,
           "alpha_lower": 0.0, "matte_red": 0.0, "matte_black": 0.1, "matte": False,
           "wrap": None, "wrap_tb": None, "frames": 2}

    def measure(self, claim):
        sc = {"id": "1", "tags": [], "files": [{"width": 8192, "height": 4096, "duration": 600,
                                                "path": "/media/VR/S/x.mp4"}]}
        order = []
        with mock.patch.object(v.os.path, "exists", return_value=True), \
                mock.patch.object(v, "read_metadata",
                                  side_effect=lambda *a: order.append("meta") or claim), \
                mock.patch.object(v, "probe",
                                  side_effect=lambda *a: order.append("probe") or self.RES):
            got = v.measure_projection(cfg(), sc)
        return got, order

    def test_metadata_is_read_before_the_frames(self):
        (want, why), order = self.measure(v.parse_ffprobe(fixture("ffprobe_equirect_sbs.json")))
        self.assertEqual(order, ["meta", "probe"])
        self.assertEqual(want, {v.DOME, v.SBS})
        self.assertIn("metadata side by side, equirectangular 360deg (not trusted: SPHERE", why)

    def test_no_metadata(self):
        (want, why), _ = self.measure(None)
        self.assertEqual(want, {v.DOME, v.SBS})
        self.assertNotIn("metadata", why)


if __name__ == "__main__":
    unittest.main()
