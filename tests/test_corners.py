import unittest

import vrQualityTags as v
from tests import synth

RED = (235, 12, 18)


def stats(rgb, w=256, h=128):
    return v.corner_matte(rgb, w, h, v.eye_boxes(w, h, w / h > 1.5))


def top_blobs(size):
    """Red silhouettes of the given size in the two top corners of each eye."""
    def corner(x, y):
        if y < size and (x < size or x >= 128 - size):
            return RED
        return None
    return corner


class CornerMatte(unittest.TestCase):
    def test_black_corners(self):
        s = stats(synth.fisheye_sbs())
        self.assertEqual(s["red"], 0.0)
        self.assertGreater(s["black"], 0.99)
        self.assertFalse(v.matte_present(s))

    def test_red_corners(self):
        s = stats(synth.fisheye_sbs(corner=top_blobs(24)))
        self.assertGreater(s["red"], 0.1)
        self.assertGreater(s["red"] + s["black"], 0.99)
        self.assertTrue(v.matte_present(s))

    def test_small_silhouette_still_counts(self):
        # the matte follows the subject; a small one is still well above noise
        s = stats(synth.fisheye_sbs(corner=top_blobs(10)))
        self.assertTrue(v.matte_present(s), s)

    def test_a_logo_does_not(self):
        # SLR's red-orange logo sits outside the discs: a handful of pixels
        def logo(x, y):
            return RED if 120 <= y < 123 and 124 <= x < 128 else None
        s = stats(synth.fisheye_sbs(corner=logo))
        self.assertLess(s["red"], v.MATTE_MIN_SHARE)
        self.assertFalse(v.matte_present(s))

    def test_red_inside_the_disc_is_ignored(self):
        def px(x, y):
            x0 = 0 if x < 128 else 128
            if synth.in_disc(x, y, x0, 128, 128, 0.9):
                return RED
            return (0, 0, 0)
        s = stats(synth.frame(256, 128, px))
        self.assertEqual(s["red"], 0.0)

    def test_magenta_and_orange_are_not_matte(self):
        for colour in ((230, 40, 200), (240, 110, 20), (255, 255, 255)):
            with self.subTest(colour=colour):
                s = stats(synth.fisheye_sbs(corner=lambda x, y, c=colour: c))
                self.assertEqual(s["red"], 0.0)
                self.assertFalse(v.matte_present(s))

    def test_red_picture_in_busy_corners_is_not_matte(self):
        # a 180 equirect whose corners are full of picture, much of it a red
        # sheet (a real one measured red 0.45, black 0.03)
        def corner(x, y):
            if y > 64:
                return (200, 25, 35)
            return synth.grey(synth.blocky(x, y, 7))
        s = stats(synth.fisheye_sbs(corner=corner))
        self.assertGreater(s["red"], 0.3)
        self.assertLess(s["red"] + s["black"], v.MATTE_MIN_CLEAN)
        self.assertFalse(v.matte_present(s))

    def test_mono_frame_uses_one_eye(self):
        self.assertEqual(v.eye_boxes(256, 256, False), [(0, 0, 256, 256)])
        self.assertEqual(v.eye_boxes(256, 128, True), [(0, 0, 128, 128), (128, 0, 128, 128)])

    def test_is_matte_red(self):
        self.assertTrue(v.is_matte_red(255, 0, 0))
        self.assertTrue(v.is_matte_red(120, 30, 20))     # edge pixel blended with black
        self.assertFalse(v.is_matte_red(80, 0, 0))       # too dark to call
        self.assertFalse(v.is_matte_red(255, 100, 0))    # orange

    def test_matte_needs_both_frames(self):
        base = {"lr": 0.9, "tb": 0.1, "blk_out": 0.9, "blk_in": 0.0, "bbox": 1.0,
                "alpha_lower": 0.0, "matte_red": 0.1, "matte_black": 0.85}
        yes, no = dict(base, matte=True), dict(base, matte=False)
        self.assertTrue(v.combine_frames([yes, yes])["matte"])
        self.assertFalse(v.combine_frames([yes, no])["matte"])
        self.assertFalse(v.combine_frames([yes])["matte"])
        self.assertTrue(v.combine_frames([yes, no])["matte_any"])
        self.assertFalse(v.combine_frames([no, no])["matte_any"])

    def test_rgb_to_grey(self):
        self.assertEqual(v.rgb_to_grey(bytes([255, 0, 0, 100, 100, 100])), bytes([77, 100]))

    def test_yuv_to_rgb(self):
        # full-range BT.601: grey stays grey, pure red comes back red
        n = 3
        y = bytes([0, 128, 76])
        u = bytes([128, 128, 85])
        vv = bytes([128, 128, 255])
        rgb = v.yuv_to_rgb(y + u + vv, n)
        self.assertEqual(rgb[0:3], bytes([0, 0, 0]))
        self.assertEqual(rgb[3:6], bytes([128, 128, 128]))
        self.assertTrue(v.is_matte_red(*rgb[6:9]), rgb[6:9])

    def test_blank_corners(self):
        grey = bytes([200]) * (256 * 128)
        out = v.blank_corners(grey, 256, v.eye_boxes(256, 128, True))
        self.assertEqual(out[0], 0)                       # top-left corner of the left eye
        self.assertEqual(out[64 * 256 + 64], 200)         # centre of the left eye
        self.assertEqual(out[64 * 256 + 192], 200)        # centre of the right eye
        self.assertEqual(out[127 * 256 + 255], 0)         # bottom-right of the right eye

    def test_luma_plane_is_used_when_given(self):
        rgb = synth.fisheye_sbs()
        grey = v.rgb_to_grey(rgb)
        self.assertEqual(v.frame_metrics(rgb, 256, 128, True, grey=grey),
                         v.frame_metrics(rgb, 256, 128, True))


if __name__ == "__main__":
    unittest.main()
