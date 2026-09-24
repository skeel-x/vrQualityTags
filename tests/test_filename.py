import unittest

import vrQualityTags as v


def fn(name):
    return v.parse_filename("/media/VR/Studio/" + name)


class FilenameMarkers(unittest.TestCase):
    def test_no_markers(self):
        self.assertEqual(fn("Studio - 2024-07-23 - Some Title [VR].mp4"), {
            "stereo": None, "rl": False, "screen": None, "lens": None,
            "alpha_candidate": False, "flat3d": None, "flat3d_loose": None,
            "vr_word": True})

    def test_stereo_markers(self):
        for marker, want in (("LR", v.SBS), ("SBS", v.SBS), ("TB", v.TB), ("OU", v.TB),
                             ("MONO", v.MONO), ("2D", v.MONO)):
            with self.subTest(marker=marker):
                self.assertEqual(fn(f"Title_{marker}_8K.mp4")["stereo"], want)
                self.assertEqual(fn(f"title_{marker.lower()}.mp4")["stereo"], want)

    def test_marker_must_be_a_segment(self):
        # a title word is not a marker; neither is a marker glued to other text
        self.assertIsNone(fn("Mono Lake Sunset [VR].mp4")["stereo"])
        self.assertIsNone(fn("Studio_MonoLake_8K.mp4")["stereo"])
        self.assertIsNone(fn("Studio - Top 180 Moments.mp4")["screen"])
        self.assertIsNone(fn("Studio_180 Degrees_x.mp4")["screen"])
        self.assertIsNone(fn("Studio_1800_x.mp4")["screen"])

    def test_rl_implies_sbs(self):
        got = fn("Title_RL_180.mp4")
        self.assertEqual(got["stereo"], v.SBS)
        self.assertTrue(got["rl"])

    def test_rl_with_tb_contradicts(self):
        got = fn("Title_RL_TB.mp4")
        self.assertIsNone(got["stereo"])
        self.assertFalse(got["rl"])

    def test_projection_markers(self):
        self.assertEqual(fn("Title_180.mp4")["screen"], v.DOME)
        self.assertEqual(fn("Title_180x180_3dh.mp4")["screen"], v.DOME)
        self.assertEqual(fn("Title 180x180 LR.mp4")["screen"], v.DOME)
        self.assertEqual(fn("Title_360_TB.mp4")["screen"], v.SPHERE)
        self.assertEqual(fn("Title_360.mp4")["stereo"], None)

    def test_contradicting_projection_cancels(self):
        self.assertIsNone(fn("Title_180_360.mp4")["screen"])
        self.assertIsNone(fn("Title_LR_TB.mp4")["stereo"])

    def test_lens_markers(self):
        for word, want in (("FISHEYE190", v.RF52), ("RF52", v.RF52),
                           ("FISHEYE200", v.MKX200), ("MKX200", v.MKX200),
                           ("MKX220", v.MKX220), ("VRCA220", v.VRCA220)):
            with self.subTest(word=word):
                self.assertEqual(fn(f"Title_{word}_LR.mp4")["lens"], want)
                self.assertEqual(fn(f"Title [{word}] [VR].mp4")["lens"], want)

    def test_two_lenses_cancel(self):
        self.assertIsNone(fn("Title_MKX200_RF52.mp4")["lens"])

    def test_alpha_candidates(self):
        for name in ("Studio - 2025-03-29 - Title [Passthrough] [VR].mp4",
                     "Studio - Title (Passthrough) [VR].mp4",
                     "Studio - Pass-Through - Title [VR].mp4",
                     "Title_ALPHA_LR.mp4", "Title pass through.mp4",
                     "Title_alphapacked.mp4", "Title packed alpha.mp4"):
            with self.subTest(name=name):
                self.assertTrue(fn(name)["alpha_candidate"])
        # a candidate only: "Mission Alpha" is not passthrough, the pixels decide
        self.assertTrue(fn("Studio - Mission Alpha in Prague [VR].mp4")["alpha_candidate"])
        self.assertFalse(fn("Alphabet City [VR].mp4")["alpha_candidate"])

    def test_empty_path(self):
        self.assertIsNone(v.parse_filename(None)["stereo"])


class Flat3DMarkers(unittest.TestCase):
    def test_strong_markers_mean_flat_3d(self):
        for name, stereo in (("IMG_7474_1_apo8_prob4_LRF_Full_SBS.mp4", v.SBS),
                             ("clip_LRF.mp4", v.SBS),
                             ("Movie (2010) Half-SBS 1080p.mkv", v.SBS),
                             ("Movie.2010.Full.SBS.mkv", v.SBS),
                             ("Movie [HSBS].mkv", v.SBS),
                             ("Movie [FSBS].mp4", v.SBS),
                             ("Movie.3D.HOU.mkv", v.TB),
                             ("Movie 3D Half-OU.mkv", v.TB),
                             ("Movie 3D TAB.mkv", v.TB),
                             ("PMV - Heavy Bounce 2_TBF_fulltb.mp4", v.TB),
                             ("clip_Full-TB.mp4", v.TB)):
            with self.subTest(name=name):
                got = fn(name)
                self.assertEqual((got["screen"], got["stereo"]), (v.FLAT, stereo))
                self.assertEqual(got["flat3d"], stereo)
                self.assertEqual(got["flat3d_loose"], stereo)

    def test_strong_marker_beats_vr_markers(self):
        got = fn("clip_MKX200_Full_SBS.mp4")
        self.assertEqual((got["screen"], got["stereo"], got["lens"]), (v.FLAT, v.SBS, None))

    def test_loose_markers_only_for_the_flat_scan(self):
        for name, stereo in (("Movie 3D.mkv", v.SBS), ("Movie.3D.mkv", v.SBS),
                             ("Movie SBS.mkv", v.SBS), ("Movie (3d) sbs.mp4", v.SBS),
                             ("Movie OU.mkv", v.TB), ("Movie 3D OU.mkv", v.TB),
                             ("movie_sbs.mp4", v.SBS)):
            with self.subTest(name=name):
                got = fn(name)
                self.assertIsNone(got["flat3d"])
                self.assertEqual(got["flat3d_loose"], stereo)
                self.assertNotEqual(got["screen"], v.FLAT)

    def test_vr_names_are_not_flat(self):
        for name in ("PMV - Title_3D_VR_SBS_60fps.mp4", "Title 3D 180 SBS.mp4",
                     "Title_3D_MKX200.mp4", "Title VR180 3D.mp4"):
            with self.subTest(name=name):
                self.assertIsNone(fn(name)["flat3d_loose"])
        # an explicit flat 3D marker still counts
        self.assertEqual(fn("Title VR LRF.mp4")["flat3d_loose"], v.SBS)

    def test_tokens_must_stand_alone(self):
        for name in ("Movie 3DX.mkv", "Movie Tablet.mkv", "Our Movie.mkv", "HOUSE.mkv",
                     "Movie 1080p.mkv", "Lrfoo.mp4", "Sbsx.mp4", "Movie TB.mkv"):
            with self.subTest(name=name):
                got = fn(name)
                self.assertIsNone(got["flat3d_loose"])
                self.assertIsNone(got["flat3d"])

    def test_contradicting_flat_markers_cancel(self):
        self.assertIsNone(fn("Movie HSBS HOU.mkv")["flat3d"])
        self.assertIsNone(fn("Movie SBS OU.mkv")["flat3d_loose"])

    def test_scan_regex_finds_the_candidates(self):
        import re
        rx = re.compile(v.FLAT3D_PATH_REGEX)
        for path in ("/m/Movie 3D.mkv", "/m/Movie.Half-SBS.mkv", "/m/IMG_1_LRF_Full_SBS.mp4",
                     "/m/Movie [HOU].mkv", "/m/Movie TAB.mkv", "/m/x_fsbs.mp4",
                     "/m/PMV_TBF_fulltb.mp4", "/m/clip_fullsbs.mp4"):
            with self.subTest(path=path):
                self.assertTrue(rx.search(path))
        for path in ("/m/Movie.mkv", "/m/House.mkv", "/m/Tablet.mkv", "/m/Movie TB.mkv"):
            with self.subTest(path=path):
                self.assertFalse(rx.search(path))


if __name__ == "__main__":
    unittest.main()


class TestLensAndStereoSpellings(unittest.TestCase):
    def check(self, name, screen=None, lens=None, stereo=None):
        f = v.parse_filename("/lib/" + name)
        self.assertEqual((f["screen"], f["lens"], f["stereo"]), (screen, lens, stereo), name)

    def test_plain_fisheye_word(self):
        self.check("Title FISHEYE.mp4", screen=v.FISHEYE)
        self.check("Title_fisheye_LR.mp4", screen=v.FISHEYE, stereo=v.SBS)

    def test_fisheye_with_fov(self):
        self.check("Title_FISHEYE220.mp4", lens=v.MKX220)
        self.check("Title_Fisheye_190.mp4", screen=v.FISHEYE, lens=v.RF52)

    def test_lens_with_separator(self):
        self.check("Title_MKX-220.mp4", lens=v.MKX220)
        self.check("Title mkx 200.mp4", lens=v.MKX200)
        self.check("Title VRCA-220.mp4", lens=v.VRCA220)

    def test_deovr_3dh_3dv(self):
        self.check("Title_180_3dh.mp4", screen=v.DOME, stereo=v.SBS)
        self.check("Title_360_3dv.mp4", screen=v.SPHERE, stereo=v.TB)

    def test_contradicting_screen_words_leave_it_to_pixels(self):
        self.check("Title_fisheye_180.mp4", screen=None)
