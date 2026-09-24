import unittest

import vrQualityTags as v
from tests import synth


def measure(rgb, w, h, frames=2):
    """Run the per-frame pipeline as probe() does, on a synthetic frame."""
    tw, th = w, h
    m = v.frame_metrics(rgb, tw, th, w / h > 1.5)
    return v.combine_frames([m] * frames if m else [])


class Classifier(unittest.TestCase):
    def classify(self, rgb, w, h, file_w=None, file_h=None):
        res = measure(rgb, w, h)
        return v.classify(file_w or w * 32, file_h or h * 32, res)

    def test_fisheye_sbs(self):
        screen, stereo, why = self.classify(synth.fisheye_sbs(), 256, 128)
        self.assertEqual((screen, stereo), (v.FISHEYE, v.SBS), why)

    def test_equirect_180_sbs(self):
        screen, stereo, why = self.classify(synth.equirect_sbs(), 256, 128)
        self.assertEqual((screen, stereo), (v.DOME, v.SBS), why)

    def test_mono_360(self):
        screen, stereo, why = self.classify(synth.mono(256, 128), 256, 128)
        self.assertEqual((screen, stereo), (v.SPHERE, v.MONO), why)

    def test_top_bottom_360(self):
        screen, stereo, why = self.classify(synth.tb(), 256, 256)
        self.assertEqual((screen, stereo), (v.SPHERE, v.TB), why)

    def test_flat_16_9(self):
        screen, stereo, why = self.classify(synth.mono(256, 144), 256, 144, 1920, 1080)
        self.assertEqual((screen, stereo), (v.FLAT, v.MONO), why)

    def test_packed_alpha_guard(self):
        res = measure(synth.packed_alpha_tb(), 256, 256)
        self.assertGreater(res["alpha_lower"], v.ALPHA_BIMODAL)
        screen, stereo, why = v.classify(4096, 4096, res)
        self.assertIsNone(screen)
        self.assertIn("packed alpha", why)

    def test_no_probe(self):
        self.assertEqual(v.classify(8192, 4096, None), (None, None, "no probe"))

    def test_blank_frame_is_not_measured(self):
        self.assertIsNone(v.frame_metrics(bytes(256 * 128 * 3), 256, 128, True))
        self.assertIsNone(v.combine_frames([]))

    def test_matte_does_not_break_fisheye(self):
        # large red silhouettes in every corner: still a fisheye pair
        def corner(x, y):
            return (230, 10, 10) if (x < 30 or x > 97) and (y < 30 or y > 97) else None
        rgb = synth.fisheye_sbs(corner=corner)
        res = measure(rgb, 256, 128)
        self.assertTrue(res["matte"])
        screen, stereo, why = v.classify(8192, 4096, res)
        self.assertEqual((screen, stereo), (v.FISHEYE, v.SBS), why)

    def test_full_sbs_flat_3d(self):
        # 7680x2160: two 16:9 eyes side by side, not a 180 pair
        w, h = 256, 72
        rgb = synth.frame(w, h, lambda x, y: synth.grey(synth.blocky(x % 128, y, 6)))
        screen, stereo, why = self.classify(rgb, w, h, 7680, 2160)
        self.assertEqual((screen, stereo), (v.FLAT, v.SBS), why)
        self.assertEqual(v.thumb_size(7680, 2160), (256, 72))

    def test_flat_3d_aspect_band(self):
        for w, h in ((3840, 1080), (7680, 2160), (3840, 1200), (6400, 2000)):
            with self.subTest(size=(w, h)):
                self.assertEqual(v.screen_from_eye(v.eye_aspect(w, h, v.SBS)), v.FLAT)
        # a 180 SBS file is 2:1 overall, each eye square
        self.assertEqual(v.screen_from_eye(v.eye_aspect(8192, 4096, v.SBS)), v.DOME)

    def test_flat_eye_is_never_a_fisheye(self):
        # a dark border around 16:9 content can look like a disc to the bbox test
        res = {"lr": 0.9, "tb": 0.1, "blk_out": 0.9, "blk_in": 0.0, "bbox": 1.0,
               "alpha_lower": 0.0, "matte": False}
        self.assertEqual(v.classify(7680, 2160, res)[:2], (v.FLAT, v.SBS))
        self.assertEqual(v.classify(8192, 4096, res)[:2], (v.FISHEYE, v.SBS))

    def test_matte_implies_fisheye(self):
        # dark top of the disc makes the content box a barrel; the matte settles it
        res = {"lr": 0.8, "tb": 0.1, "blk_out": 1.0, "blk_in": 0.0, "bbox": 1.09,
               "alpha_lower": 0.0, "matte": False}
        self.assertEqual(v.classify(8192, 4096, res)[0], v.DOME)
        res["matte"] = True
        screen, stereo, why = v.classify(8192, 4096, res)
        self.assertEqual((screen, stereo), (v.FISHEYE, v.SBS))
        self.assertIn("corner matte", why)

    def test_thumb_size_is_even(self):
        self.assertEqual(v.thumb_size(8192, 4096), (256, 128))
        self.assertEqual(v.thumb_size(4096, 4096), (256, 256))
        self.assertEqual(v.thumb_size(1920, 1080), (256, 144))
        tw, th = v.thumb_size(3000, 1001)
        self.assertEqual(th % 2, 0)

    def test_screen_from_eye(self):
        self.assertEqual(v.screen_from_eye(1.0), v.DOME)
        self.assertEqual(v.screen_from_eye(16 / 9), v.FLAT)
        self.assertEqual(v.screen_from_eye(2.0), v.SPHERE)
        self.assertIsNone(v.screen_from_eye(4.0))


class Resolve(unittest.TestCase):
    NONE = v.parse_filename("plain.mp4")

    def fn(self, **kw):
        d = dict(self.NONE)
        d.update(kw)
        return d

    def test_pixels_only(self):
        self.assertEqual(v.resolve(self.NONE, v.DOME, v.SBS, False), {v.DOME, v.SBS})

    def test_mono_sphere_carries_mono(self):
        self.assertEqual(v.resolve(self.NONE, v.SPHERE, v.MONO, False), {v.SPHERE, v.MONO})
        self.assertEqual(v.resolve(self.NONE, v.DOME, v.MONO, False), {v.DOME, v.MONO})

    def test_flat_2d_is_flat_alone(self):
        self.assertEqual(v.resolve(self.NONE, v.FLAT, v.MONO, False), {v.FLAT})
        fn = v.parse_filename("x_2D.mp4")
        self.assertEqual(v.resolve(fn, v.FLAT, v.SBS, False), {v.FLAT})

    def test_flat_3d(self):
        self.assertEqual(v.resolve(self.NONE, v.FLAT, v.SBS, False), {v.FLAT, v.SBS})
        self.assertEqual(v.resolve(self.NONE, v.FLAT, v.TB, False), {v.FLAT, v.TB})
        fn = v.parse_filename("IMG_7474_LRF_Full_SBS.mp4")
        # the marker wins even when the pixels could not decide
        self.assertEqual(v.resolve(fn, v.DOME, v.MONO, False), {v.FLAT, v.SBS})
        self.assertEqual(v.resolve(fn, None, None, False), {v.FLAT, v.SBS})

    def test_unresolved_keeps_alpha(self):
        self.assertEqual(v.resolve(self.NONE, None, None, False), {v.UNRESOLVED})
        self.assertEqual(v.resolve(self.NONE, None, None, True), {v.UNRESOLVED, v.ALPHA})

    def test_alpha(self):
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, True),
                         {v.FISHEYE, v.SBS, v.ALPHA})

    def test_filename_beats_pixels(self):
        fn = self.fn(stereo=v.TB, screen=v.SPHERE)
        self.assertEqual(v.resolve(fn, v.DOME, v.SBS, False), {v.SPHERE, v.TB})

    def test_filename_lens_makes_fisheye(self):
        fn = self.fn(lens=v.MKX220)
        self.assertEqual(v.resolve(fn, v.DOME, v.SBS, False), {v.FISHEYE, v.MKX220, v.SBS})
        # and it rescues a scene the pixels could not place
        self.assertEqual(v.resolve(fn, None, None, False), {v.FISHEYE, v.MKX220})

    def test_watermark_lens(self):
        for fov, lens in (("190", v.RF52), ("200", v.MKX200), ("220", v.MKX220)):
            with self.subTest(fov=fov):
                self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, (fov, False)),
                                 {v.FISHEYE, lens, v.SBS})
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, ("220", True)),
                         {v.FISHEYE, v.VRCA220, v.SBS})
        # 180 is not a fisheye lens stash-vr knows: plain FISHEYE
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, ("180", False)),
                         {v.FISHEYE, v.SBS})

    def test_watermark_ignored_off_fisheye(self):
        self.assertEqual(v.resolve(self.NONE, v.DOME, v.SBS, False, ("200", False)),
                         {v.DOME, v.SBS})

    def test_filename_lens_beats_watermark(self):
        fn = self.fn(lens=v.RF52)
        self.assertEqual(v.resolve(fn, v.FISHEYE, v.SBS, False, ("200", False)),
                         {v.FISHEYE, v.RF52, v.SBS})

    def test_rl(self):
        fn = v.parse_filename("x_RL_180.mp4")
        self.assertEqual(v.resolve(fn, v.DOME, v.TB, False), {v.DOME, v.SBS, v.RL})

    def test_fov_from_text(self):
        self.assertEqual(v.fov_from_text("SLR 200° FOV Fisheye"), ("200", False))
        self.assertEqual(v.fov_from_text("SLR 190o FOV"), ("190", False))
        # real OCR output of the centred watermark and of a crop split at the seam
        self.assertEqual(v.fov_from_text("SLR 190° FOV FISHEYE a Uf * Watch in SLR App"), ("190", False))
        self.assertEqual(v.fov_from_text("oe D SLR 190° FC VY jy"), ("190", False))
        # degree sign lost or misread
        self.assertEqual(v.fov_from_text("SLR 220 FOV"), ("220", False))
        self.assertEqual(v.fov_from_text("SLR 2200 FOV FISHEYE"), ("220", False))
        self.assertEqual(v.fov_from_text("SLR 190 F0V"), ("190", False))
        self.assertIsNone(v.fov_from_text("room 200 people"))
        self.assertEqual(v.fov_from_text("VRCA 220° lens"), ("220", True))
        self.assertIsNone(v.fov_from_text("Watch in SLR app"))
        self.assertIsNone(v.fov_from_text(None))


if __name__ == "__main__":
    unittest.main()
