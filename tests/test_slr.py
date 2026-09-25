import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest import mock

import vrQualityTags as v

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
DAY = 86400


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def cfg(**kw):
    c = v.load_config({})
    c.update(kw)
    return c


class SceneRef(unittest.TestCase):
    def test_url_forms(self):
        for url, want in (
                ("https://www.sexlikereal.com/scenes/some-title-49158", ("49158", "1")),
                ("https://sexlikereal.com/scenes/some-title-49158", ("49158", "1")),
                ("http://www.sexlikereal.com/scenes/some-title-17325?tab=about", ("17325", "1")),
                ("https://www.sexlikereal.com/scenes/some-title-17325/", ("17325", "1")),
                ("https://www.sexlikereal.com/scenes/2-girls-1-room-8#x", ("8", "1")),
                ("https://www.sexlikereal.com/16109", ("16109", "1")),
                ("https://www.sexlikereal.com/trans/scenes/a-title-555", ("555", "3")),
                ("https://www.sexlikereal.com/gay/scenes/a-title-777", ("777", "4")),
                (" https://WWW.SexLikeReal.com/scenes/x-12 ", ("12", "1"))):
            with self.subTest(url=url):
                self.assertEqual(v.slr_scene_ref([url]), want)

    def test_not_a_scene(self):
        for url in ("https://www.sexlikereal.com/pornstars/someone-123",
                    "https://www.sexlikereal.com/studios/studio-12",
                    "https://www.sexlikereal.com/scenes/no-number",
                    "https://www.example.com/scenes/title-123",
                    "https://notsexlikereal.com/scenes/title-123", "", None):
            with self.subTest(url=url):
                self.assertIsNone(v.slr_scene_ref([url]))
        self.assertIsNone(v.slr_scene_ref(None))
        self.assertIsNone(v.slr_scene_ref([]))

    def test_first_scene_url_wins(self):
        urls = ["https://www.example.com/x", "https://www.sexlikereal.com/scenes/a-1",
                "https://www.sexlikereal.com/scenes/b-2"]
        self.assertEqual(v.slr_scene_ref(urls), ("1", "1"))


class Claim(unittest.TestCase):
    def claim(self, name):
        return v.slr_claim(fixture(name)["data"])

    def test_mkx200(self):
        got = self.claim("slr_mkx200.json")
        self.assertEqual((got["screen"], got["lens"], got["stereo"], got["rl"]),
                         (v.FISHEYE, v.MKX200, v.SBS, False))
        # AI passthrough is SLR's separately streamed mask: no matte in the file
        self.assertEqual((got["alpha"], got["chroma"]), (False, False))
        self.assertEqual(got["raw"], "viewAngle 200, fisheye mkx200, sbs2l")

    def test_rf52_from_the_view_angle(self):
        got = self.claim("slr_rf52.json")
        self.assertEqual((got["screen"], got["lens"], got["alpha"]), (v.FISHEYE, v.RF52, False))
        # only inferred from viewAngle: the watermark may still correct it
        self.assertTrue(got["lens_inferred"])
        self.assertFalse(self.claim("slr_mkx200.json")["lens_inferred"])
        self.assertTrue(self.claim("slr_fisheye_180.json")["lens_inferred"])
        self.assertNotIn("lens_inferred", self.claim("slr_180.json"))

    def test_native_alpha(self):
        got = self.claim("slr_rf52_alpha.json")
        self.assertEqual((got["screen"], got["lens"], got["alpha"]), (v.FISHEYE, v.RF52, True))
        self.assertIn("alpha", got["raw"])

    def test_180_equirect(self):
        got = self.claim("slr_180.json")
        self.assertEqual((got["screen"], got["stereo"], got["alpha"]), (v.DOME, v.SBS, False))
        self.assertNotIn("lens", got)

    def test_360_top_bottom(self):
        got = self.claim("slr_360_tb.json")
        self.assertEqual((got["screen"], got["stereo"], got["alpha"]), (v.SPHERE, v.TB, False))

    def test_180_fisheye_settles_no_lens(self):
        got = self.claim("slr_fisheye_180.json")
        self.assertEqual(got["screen"], v.FISHEYE)
        self.assertIn("lens", got)
        self.assertIsNone(got["lens"])

    def test_chroma_key(self):
        data = fixture("slr_mkx200.json")["data"]
        data["passthrough"]["chromaKey"]["enabled"] = True
        self.assertTrue(v.slr_claim(data)["chroma"])

    def test_lens_names(self):
        for lens, want in (("mkx220", v.MKX220), ("vrca220", v.VRCA220), ("rf52", v.RF52),
                           ("MKX200", v.MKX200), ("unknown", v.MKX200)):
            with self.subTest(lens=lens):
                data = {"viewAngle": 200, "projectionParams": {"projection": 1,
                                                               "cameraLens": lens}}
                self.assertEqual(v.slr_claim(data)["lens"], want)
        self.assertEqual(v.slr_claim({"viewAngle": 220, "projection": 4})["lens"], v.MKX220)

    def test_stereo_modes(self):
        for mode, want in (("sbs2r", (v.SBS, True)), ("ab2l", (v.TB, False)),
                           ("ab2r", (v.TB, False)), ("mono", (v.MONO, False))):
            with self.subTest(mode=mode):
                got = v.slr_claim({"viewAngle": 180, "projection": 0, "stereomode": mode})
                self.assertEqual((got["stereo"], got["rl"]), want)
        got = v.slr_claim({"projectionParams": {"format": 1, "viewAngle": 360, "projection": 0}})
        self.assertEqual((got["screen"], got["stereo"]), (v.SPHERE, v.TB))

    def test_category_fallbacks(self):
        got = v.slr_claim({"categories": [{"name": "Fisheye"}, {"name": "Passthrough (Native)"}]})
        self.assertEqual((got["screen"], got["lens"], got["alpha"]), (v.FISHEYE, None, True))
        got = v.slr_claim({"projection": 0, "categories": ["360°"]})
        self.assertEqual(got["screen"], v.SPHERE)
        got = v.slr_claim({"categories": [{"name": "Chroma key"}], "stereomode": "sbs2l"})
        self.assertTrue(got["chroma"])
        self.assertNotIn("screen", got)

    def test_nothing_usable(self):
        self.assertIsNone(v.slr_claim(None))
        self.assertIsNone(v.slr_claim({}))
        self.assertIsNone(v.slr_claim({"categories": [{"name": "Blonde"}]}))
        self.assertIsNone(v.slr_claim({"projection": 7, "viewAngle": None}))

    def test_slim_keeps_what_the_claim_needs(self):
        for name in ("slr_mkx200.json", "slr_rf52_alpha.json", "slr_180.json",
                     "slr_360_tb.json", "slr_fisheye_180.json"):
            with self.subTest(name=name):
                data = fixture(name)["data"]
                data.update({"title": "T", "actors": [{"name": "A"}],
                             "thumbnailUrl": "https://x/y.jpg"})
                slim = v.slim_slr(data)
                self.assertEqual(v.slr_claim(slim), v.slr_claim(data))
                text = json.dumps(slim)
                self.assertNotIn("title", text)
                self.assertNotIn("https", text)
                self.assertNotIn("actors", text)
                self.assertNotIn("3D", slim["categories"])


class FakeResponse(io.BytesIO):
    def __init__(self, status, body):
        super().__init__(body)
        self.status = status


class FakeApi:
    """Answers by scene id; records every request."""

    def __init__(self, clock, answers):
        self.clock = clock
        self.answers = answers
        self.requests = []

    def __call__(self, req, timeout=None):
        sid = req.full_url.rsplit("/", 1)[1]
        self.requests.append((self.clock.now, sid, dict(req.header_items()), timeout))
        a = self.answers.get(sid, 404)
        if callable(a):
            a = a(req)
        if isinstance(a, Exception):
            raise a
        if isinstance(a, int):
            raise urllib.error.HTTPError(req.full_url, a, "x", {}, io.BytesIO(b"{}"))
        return FakeResponse(200, json.dumps(a).encode())


class Clock:
    def __init__(self, now=1_700_000_000.0):
        self.now = now
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.now += s


class Lookup(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, v.SLR_CACHE_FILE)
        self.clock = Clock()
        self.api = FakeApi(self.clock, {"1001": fixture("slr_mkx200.json"),
                                        "1004": fixture("slr_180.json")})
        self.log = mock.patch.object(v, "log").start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(self.dir.cleanup)

    def lookup(self):
        return v.SlrLookup(self.path, clock=self.clock, sleep=self.clock.sleep, opener=self.api)

    def test_fetch_and_headers(self):
        got = self.lookup().scene(("1001", "1"))
        self.assertEqual(got["viewAngle"], 200)
        _, sid, headers, timeout = self.api.requests[0]
        self.assertEqual(sid, "1001")
        self.assertIn("vrQualityTags", headers["User-agent"])
        self.assertEqual(headers["Client-type"], "web")
        self.assertEqual(headers["Project"], "1")
        self.assertEqual(timeout, v.SLR_TIMEOUT)

    def test_cached_answers_are_reused_until_they_expire(self):
        lk = self.lookup()
        lk.scene(("1001", "1"))
        with open(self.path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["1001"]["fetched"], self.clock.now)
        self.assertEqual(saved["1001"]["scene"]["projectionParams"]["cameraLens"], "mkx200")
        # a new run reads the file: no request
        self.clock.now += 89 * DAY
        self.assertEqual(self.lookup().scene(("1001", "1"))["viewAngle"], 200)
        self.assertEqual(len(self.api.requests), 1)
        # after 90 days it is fetched again
        self.clock.now += 2 * DAY
        self.lookup().scene(("1001", "1"))
        self.assertEqual(len(self.api.requests), 2)

    def test_misses_are_cached_for_30_days(self):
        lk = self.lookup()
        self.assertIsNone(lk.scene(("999", "3")))
        self.assertEqual(len(self.api.requests), 1)      # no project-0 retry off project 1
        self.clock.now += 29 * DAY
        self.assertIsNone(self.lookup().scene(("999", "3")))
        self.assertEqual(len(self.api.requests), 1)
        self.clock.now += 2 * DAY
        self.assertIsNone(self.lookup().scene(("999", "3")))
        self.assertEqual(len(self.api.requests), 2)

    def test_not_found_on_project_1_retries_project_0(self):
        self.api.answers["17"] = lambda req: (fixture("slr_180.json")
                                              if req.get_header("Project") == "0" else 404)
        got = self.lookup().scene(("17", "1"))
        self.assertEqual(got["viewAngle"], 180)
        self.assertEqual([r[2]["Project"] for r in self.api.requests], ["1", "0"])

    def test_error_body_is_a_miss(self):
        self.api.answers["5"] = fixture("slr_not_found.json")
        self.assertIsNone(self.lookup().scene(("5", "1")))
        with open(self.path, encoding="utf-8") as f:
            self.assertIsNone(json.load(f)["5"]["scene"])

    def test_one_request_per_second(self):
        lk = self.lookup()
        for sid in ("1001", "1004", "404a", "1001"):
            lk.scene((sid, "3"))
            self.clock.now += 0.25              # work between scenes
        times = [r[0] for r in self.api.requests]
        self.assertEqual(len(times), 3)         # the second 1001 came from the cache
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual(b - a, v.SLR_INTERVAL)
        self.assertEqual(self.clock.slept, [0.75, 0.75])

    def test_rate_limited_stops_the_run(self):
        self.api.answers["7"] = 429
        lk = self.lookup()
        self.assertIsNone(lk.scene(("7", "1")))
        self.assertIsNone(lk.scene(("1001", "1")))
        self.assertEqual(len(self.api.requests), 1)
        self.assertIn("429", self.log.call_args[0][1])
        self.assertFalse(os.path.exists(self.path))     # nothing cached

    def test_network_errors_fall_back_and_stop_after_three(self):
        for sid in ("a", "b", "c"):
            self.api.answers[sid] = urllib.error.URLError("down")
        self.api.answers["d"] = TimeoutError()
        lk = self.lookup()
        self.assertIsNone(lk.scene(("a", "1")))
        self.assertIsNone(lk.scene(("b", "1")))
        self.assertIsNone(lk.scene(("c", "1")))
        self.assertIsNone(lk.scene(("1001", "1")))
        self.assertEqual(len(self.api.requests), 3)
        self.assertTrue(lk.stopped)
        # and a new run tries again
        self.assertEqual(self.lookup().scene(("1001", "1"))["viewAngle"], 200)

    def test_one_success_resets_the_failure_count(self):
        self.api.answers["a"] = TimeoutError()
        self.api.answers["b"] = 503
        lk = self.lookup()
        lk.scene(("a", "1"))
        lk.scene(("b", "1"))
        lk.scene(("1001", "1"))
        lk.scene(("a", "1"))
        lk.scene(("b", "1"))
        self.assertFalse(lk.stopped)

    def test_stale_answer_beats_a_failed_refetch(self):
        self.lookup().scene(("1001", "1"))
        self.clock.now += 100 * DAY
        self.api.answers["1001"] = 500
        self.assertEqual(self.lookup().scene(("1001", "1"))["viewAngle"], 200)

    def test_bad_json_is_a_failure(self):
        lk = self.lookup()
        lk.opener = lambda req, timeout=None: FakeResponse(200, b"<html>")
        self.assertIsNone(lk.scene(("x", "1")))
        self.assertEqual(lk.failures, 1)

    def test_unreadable_cache_starts_empty(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("[1, 2")
        self.assertEqual(self.lookup().cache, {})
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("[1, 2]")
        self.assertEqual(self.lookup().cache, {})

    def test_unwritable_cache_warns(self):
        lk = v.SlrLookup(os.path.join(self.dir.name, "missing", "c.json"), clock=self.clock,
                         sleep=self.clock.sleep, opener=self.api)
        self.assertEqual(lk.scene(("1001", "1"))["viewAngle"], 200)
        self.assertIn("cannot save the SLR cache", self.log.call_args[0][1])

    def test_cache_path(self):
        self.assertEqual(v.slr_cache_path({"PluginDir": "/p"}), "/p/vrQualityTags.slr.json")


class StaticLookup:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def scene(self, ref):
        self.calls.append(ref)
        return self.data


PAIR = {"lr": 0.9, "tb": 0.1}
URL = "https://www.sexlikereal.com/scenes/a-title-1001"


class LookupSlr(unittest.TestCase):
    def run_lookup(self, data, screen_px, stereo_px=v.SBS, w=5800, h=2900, urls=(URL,),
                   res=PAIR):
        c = cfg(slrLookup=True)
        c["_slr"] = StaticLookup(data)
        with mock.patch.object(v, "log") as log:
            got = v.lookup_slr(c, {"id": "9", "urls": list(urls)}, w, h, res,
                               screen_px, stereo_px)
        return got, log

    def test_agreeing_answer(self):
        (got, why), log = self.run_lookup(fixture("slr_rf52_alpha.json")["data"], v.FISHEYE)
        self.assertEqual(got, {"stereo": v.SBS, "rl": False, "screen": v.FISHEYE,
                               "lens": v.RF52, "lens_inferred": True, "alpha": True,
                               "chroma": False})
        self.assertEqual(why, ", SLR 1001 viewAngle 190, fisheye, sbs2l, alpha")
        log.assert_not_called()

    def test_equirect_180_beats_a_disc_only_fisheye(self):
        # the disc test takes vignetted 180 equirects for fisheye
        (got, why), log = self.run_lookup(fixture("slr_180.json")["data"], v.FISHEYE)
        self.assertEqual(got["screen"], v.DOME)
        self.assertIn("SLR equirect 180 over the disc test", why)
        log.assert_not_called()

    def test_equirect_180_does_not_beat_a_matte_fisheye(self):
        (got, why), log = self.run_lookup(fixture("slr_180.json")["data"], v.FISHEYE,
                                          res=dict(PAIR, matte=True))
        self.assertIsNone(got)
        self.assertIn("contradicts the frame", why)
        self.assertIn("kept the pixels", log.call_args[0][1])

    def test_360_does_not_beat_a_disc_fisheye(self):
        data = dict(fixture("slr_180.json")["data"], viewAngle=360)
        data["projectionParams"] = dict(data["projectionParams"], viewAngle=360)
        (got, _), _ = self.run_lookup(data, v.FISHEYE)
        self.assertIsNone(got)

    def test_fisheye_never_beats_a_pixel_equirect(self):
        # a download can be an equirect conversion of a fisheye scene
        (got, why), _ = self.run_lookup(fixture("slr_mkx200.json")["data"], v.DOME)
        self.assertIsNone(got)
        self.assertIn("contradicts the frame", why)

    def test_flat_frame_still_wins(self):
        (got, _), _ = self.run_lookup(fixture("slr_180.json")["data"], v.FLAT, v.MONO,
                                      w=1920, h=1080)
        self.assertIsNone(got)
        # unresolved 16:9 frame whose halves do not match: no 180 eye in it
        (got, _), _ = self.run_lookup(fixture("slr_180.json")["data"], None, None,
                                      w=1280, h=720, res={"lr": 0.1, "tb": 0.1})
        self.assertIsNone(got)

    def test_off_or_no_url(self):
        c = cfg()
        self.assertEqual(v.lookup_slr(c, {"urls": [URL]}, 1, 1, None, None, None), (None, ""))
        (got, why), _ = self.run_lookup(fixture("slr_180.json")["data"], v.DOME, urls=())
        self.assertEqual((got, why), (None, ""))
        (got, why), _ = self.run_lookup(None, v.DOME)
        self.assertEqual((got, why), (None, ""))


class Authority(unittest.TestCase):
    NONE = v.parse_filename("plain.mp4")

    def slr(self, name, screen_px=v.FISHEYE):
        c = cfg(slrLookup=True)
        c["_slr"] = StaticLookup(fixture(name)["data"])
        with mock.patch.object(v, "log"):
            vetted, _ = v.lookup_slr(c, {"id": "9", "urls": [URL]}, 5800, 2900, PAIR,
                                     screen_px, v.SBS)
        return vetted

    def test_explicit_slr_lens_beats_filename_and_watermark(self):
        fn = v.parse_filename("x_RF52.mp4")
        self.assertEqual(v.resolve(fn, v.FISHEYE, v.SBS, False, ("220", False), None,
                                   self.slr("slr_mkx200.json")),
                         {v.FISHEYE, v.MKX200, v.SBS})

    def test_inferred_slr_lens_yields_to_the_watermark(self):
        # recorded: viewAngle 190 without cameraLens, the file says "SLR 200° FOV"
        slr = self.slr("slr_190_watermark_200.json")
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, ("200", False), None, slr),
                         {v.FISHEYE, v.MKX200, v.SBS})
        # no readable watermark: SLR's lens stands, and it beats the filename
        fn = v.parse_filename("x_MKX220.mp4")
        self.assertEqual(v.resolve(fn, v.FISHEYE, v.SBS, False, None, None, slr),
                         {v.FISHEYE, v.RF52, v.SBS})

    def test_metadata_beats_slr(self):
        meta = {"screen": v.FISHEYE, "lens": v.MKX220}
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, None, meta,
                                   self.slr("slr_rf52.json")),
                         {v.FISHEYE, v.MKX220, v.SBS})

    def test_180_fisheye_has_no_lens(self):
        slr = self.slr("slr_fisheye_180.json")
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, None, None, slr),
                         {v.FISHEYE, v.SBS})
        # inferred from viewAngle 180 too, so a readable watermark still counts
        self.assertEqual(v.resolve(self.NONE, v.FISHEYE, v.SBS, False, ("200", False), None, slr),
                         {v.FISHEYE, v.MKX200, v.SBS})

    def test_alpha_and_chroma(self):
        self.assertIn(v.ALPHA, v.resolve(self.NONE, v.FISHEYE, v.SBS, False, None, None,
                                         self.slr("slr_rf52_alpha.json")))
        # the pixel-measured matte always stands
        self.assertIn(v.ALPHA, v.resolve(self.NONE, v.FISHEYE, v.SBS, True, None, None,
                                         self.slr("slr_rf52.json")))
        slr = self.slr("slr_rf52.json")
        slr["chroma"] = True
        self.assertIn(v.CHROMA, v.resolve(self.NONE, v.FISHEYE, v.SBS, False, None, None, slr))

    def test_watermark_skipped_only_for_an_explicit_lens(self):
        path = "/m/SLR/x.mp4"
        fn = v.parse_filename(path)
        self.assertEqual(v.fov_skip_reason(fn, v.FISHEYE, False, path, None,
                                           self.slr("slr_mkx200.json")), "lens from SLR")
        self.assertIsNone(v.fov_skip_reason(fn, v.FISHEYE, False, path, None,
                                            self.slr("slr_rf52.json")))
        self.assertIsNone(v.fov_skip_reason(fn, v.FISHEYE, False, path, None,
                                            self.slr("slr_fisheye_180.json")))
        # the existing skip rules still apply
        other = "/m/A/Studio - X [Passthrough].mp4"
        self.assertEqual(v.fov_skip_reason(v.parse_filename(other), v.FISHEYE, True, other,
                                           None, self.slr("slr_rf52.json")),
                         "passthrough not from SLR")
        # an SLR 180 equirect over a disc-only fisheye: nothing to read
        self.assertEqual(v.fov_skip_reason(fn, v.FISHEYE, False, path, None,
                                           self.slr("slr_180.json")), "not fisheye")


class MeasureWithSlr(unittest.TestCase):
    RES = {"lr": 0.9, "tb": 0.1, "blk_out": 0.95, "blk_in": 0.0, "bbox": 1.0,
           "alpha_lower": 0.0, "matte_red": 0.0, "matte_black": 0.95, "matte": False,
           "wrap": None, "wrap_tb": None, "frames": 2}

    def measure(self, data, res=None, fov=("200", False), **kw):
        c = cfg(slrLookup=True, **kw)
        c["_slr"] = StaticLookup(data)
        sc = {"id": "1", "tags": [], "urls": [URL],
              "files": [{"width": 5800, "height": 2900, "duration": 600,
                         "path": "/media/VR/SLR/SLR Originals - X [VR].mp4"}]}
        with mock.patch.object(v.os.path, "exists", return_value=True), \
                mock.patch.object(v, "read_metadata", return_value=None), \
                mock.patch.object(v, "probe", return_value=res or self.RES), \
                mock.patch.object(v, "read_fov", return_value=fov) as read_fov, \
                mock.patch.object(v, "log"):
            want, why = v.measure_projection(c, sc)
        return want, why, read_fov

    def test_explicit_slr_lens_skips_the_ocr(self):
        want, why, fov = self.measure(fixture("slr_mkx200.json")["data"], fov=("190", False))
        self.assertEqual(want, {v.FISHEYE, v.MKX200, v.SBS})
        fov.assert_not_called()
        self.assertIn("SLR 1001", why)

    def test_watermark_overrides_an_inferred_slr_lens(self):
        # the recorded case: SLR viewAngle 190, burned-in "SLR 200° FOV"
        want, why, fov = self.measure(fixture("slr_190_watermark_200.json")["data"])
        self.assertEqual(want, {v.FISHEYE, v.MKX200, v.SBS})
        fov.assert_called_once()
        self.assertIn("watermark 200deg overrides SLR's RF52 (from viewAngle)", why)

    def test_agreeing_or_unreadable_watermark_keeps_the_slr_lens(self):
        want, why, _ = self.measure(fixture("slr_190_watermark_200.json")["data"],
                                    fov=("190", False))
        self.assertEqual(want, {v.FISHEYE, v.RF52, v.SBS})
        self.assertNotIn("overrides", why)
        want, _, _ = self.measure(fixture("slr_190_watermark_200.json")["data"], fov=None)
        self.assertEqual(want, {v.FISHEYE, v.RF52, v.SBS})

    def test_slr_180_equirect_beats_a_disc_only_fisheye(self):
        want, why, fov = self.measure(fixture("slr_180.json")["data"])
        self.assertEqual(want, {v.DOME, v.SBS})
        fov.assert_not_called()
        self.assertIn("over the disc test", why)

    def test_matte_fisheye_keeps_the_pixels(self):
        res = dict(self.RES, matte=True, matte_red=0.05)
        want, why, fov = self.measure(fixture("slr_180.json")["data"], res=res)
        self.assertEqual(want, {v.FISHEYE, v.MKX200, v.SBS, v.ALPHA})
        fov.assert_called_once()
        self.assertIn("not used", why)

    def test_nothing_from_slr(self):
        want, why, fov = self.measure(None)
        self.assertEqual(want, {v.FISHEYE, v.MKX200, v.SBS})
        self.assertNotIn("SLR 1001", why)

    def test_native_alpha_adds_the_tag(self):
        want, _, _ = self.measure(fixture("slr_rf52_alpha.json")["data"], fov=None)
        self.assertEqual(want, {v.FISHEYE, v.RF52, v.SBS, v.ALPHA})


class Wiring(unittest.TestCase):
    def test_scene_query_fetches_urls(self):
        self.assertIn(" urls ", v.SCENE_FIELDS)
        self.assertIn(" urls ", v.SCENE_ONE)

    def test_setting_default_off(self):
        self.assertFalse(v.load_config({})["slrLookup"])
        self.assertTrue(v.load_config({"slrLookup": True})["slrLookup"])

    def test_chroma_key_is_managed(self):
        self.assertIn(v.CHROMA, v.PROJECTION_TAGS)
        self.assertNotIn(v.CHROMA, v.SETTLED_TAGS)


if __name__ == "__main__":
    unittest.main()
