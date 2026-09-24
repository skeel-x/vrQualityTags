import io
import json
import unittest
from unittest import mock

import vrQualityTags as v


def cfg(**kw):
    c = v.load_config({})
    c.update(kw)
    return c


def scene(width=8192, mbit=60, tags=(), path="/media/VR/S/x.mp4", files=None, sid="7"):
    if files is None:
        files = [{"width": width, "height": width // 2, "duration": 600,
                  "bit_rate": mbit * 1e6, "size": 1, "path": path}]
    return {"id": sid, "files": files,
            "tags": [{"id": f"id:{n}", "name": n} for n in tags]}


IDS = {n: f"id:{n}" for n in v.PROJECTION_TAGS + (v.SKIP,) + v.quality_names(cfg())
       + ("Virtual Reality", "CUBEMAP")}


class FakeStash:
    def __init__(self):
        self.writes = []

    def call(self, query, variables=None):
        assert "sceneUpdate" in query
        self.writes.append(variables["i"])
        return {"sceneUpdate": {"id": variables["i"]["id"]}}


def names(tag_ids):
    return {i.split(":", 1)[1] for i in tag_ids}


class Tiers(unittest.TestCase):
    def test_widths(self):
        c = cfg()
        self.assertEqual(v.tier_of(scene(8192), c), "tag8k")
        self.assertEqual(v.tier_of(scene(7680), c), "tag8k")
        self.assertEqual(v.tier_of(scene(7200), c), "tag7k")
        self.assertEqual(v.tier_of(scene(5760, 40), c), "tag6kHbr")
        self.assertIsNone(v.tier_of(scene(5760, 39), c))
        self.assertIsNone(v.tier_of(scene(3840, 90), c))

    def test_biggest_file_decides(self):
        files = [{"width": 3840, "bit_rate": 1, "size": 10},
                 {"width": 8192, "bit_rate": 1, "size": 99}]
        self.assertEqual(v.tier_of(scene(files=files), cfg()), "tag8k")
        self.assertIsNone(v.tier_of(scene(files=[]), cfg()))

    def test_quality_want_includes_parent(self):
        self.assertEqual(v.quality_want(scene(8192), cfg()), {"8K", "HQ"})
        self.assertEqual(v.quality_want(scene(3000), cfg()), set())
        c = cfg(tag8k="8K Real", parentTag="Quality")
        self.assertEqual(v.quality_want(scene(8192), c), {"8K Real", "Quality"})


class Diffing(unittest.TestCase):
    def test_no_change_is_none(self):
        self.assertIsNone(v.diff_tags({"a", "8K"}, {"8K", "7K"}, {"8K"}))

    def test_replace_inside_scope_only(self):
        self.assertEqual(v.diff_tags({"a", "7K", "DOME"}, {"8K", "7K"}, {"8K"}),
                         {"a", "8K", "DOME"})

    def test_clear(self):
        self.assertEqual(v.diff_tags({"a", "7K"}, {"8K", "7K"}, set()), {"a"})

    def test_settled(self):
        self.assertTrue(v.settled({"FISHEYE", "SBS"}))
        self.assertTrue(v.settled({"VRP: Unresolved"}))
        self.assertTrue(v.settled({"CUBEMAP"}))
        # vrProjectionTags left mono content with only a MONO label
        self.assertFalse(v.settled({"MONO"}))
        self.assertFalse(v.settled({"SBS", "8K"}))

    def test_hook_guard(self):
        self.assertTrue(v.hook_should_skip({"type": "Scene.Update.Post",
                                            "inputFields": ["id", "tag_ids"]}))
        self.assertFalse(v.hook_should_skip({"type": "Scene.Update.Post",
                                             "inputFields": ["id", "title"]}))
        self.assertFalse(v.hook_should_skip({"type": "Scene.Create.Post",
                                             "inputFields": ["id", "tag_ids"]}))
        self.assertFalse(v.hook_should_skip({"type": "Scene.Update.Post"}))

    def test_path_filter(self):
        self.assertTrue(v.path_matches(cfg(), "/media/VR/x.mp4"))
        self.assertFalse(v.path_matches(cfg(), "/media/Movies/x.mp4"))
        self.assertTrue(v.path_matches(cfg(pathFilter=""), "/anything"))
        self.assertFalse(v.path_matches(cfg(), None))

    def test_load_config(self):
        c = v.load_config({"min8kWidth": 8000, "pathFilter": "", "tag8k": None,
                           "overwrite": True})
        self.assertEqual(c["min8kWidth"], 8000.0)
        self.assertEqual(c["pathFilter"], "/VR/")      # empty means unset
        self.assertEqual(c["tag8k"], "8K")
        self.assertIs(c["overwrite"], True)
        self.assertIs(c["readFovWatermark"], True)
        self.assertEqual(v.load_config(None)["minWidth"], 1920.0)


class ProcessScene(unittest.TestCase):
    def run_scene(self, sc, mode="untagged", measured=({v.FISHEYE, v.SBS, v.ALPHA}, "why"),
                  **kw):
        stash = FakeStash()
        with mock.patch.object(v, "measure_projection", return_value=measured) as m, \
                mock.patch.object(v, "log"):
            out = v.process_scene(stash, cfg(**kw), sc, IDS, mode)
        return stash, out, m

    def test_new_scene_gets_everything(self):
        stash, out, m = self.run_scene(scene(tags=["Virtual Reality"]))
        m.assert_called_once()
        self.assertEqual(names(stash.writes[0]["tag_ids"]),
                         {"Virtual Reality", "FISHEYE", "SBS", "Alpha", "8K", "HQ"})
        self.assertIn("Alpha", out)

    def test_settled_scene_only_gets_quality(self):
        stash, out, m = self.run_scene(scene(tags=["DOME", "SBS"]))
        m.assert_not_called()
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"DOME", "SBS", "8K", "HQ"})

    def test_idempotent(self):
        sc = scene(tags=["FISHEYE", "SBS", "Alpha", "8K", "HQ"])
        stash, out, _ = self.run_scene(sc, mode="retag")
        self.assertEqual(stash.writes, [])
        self.assertIsNone(out)

    def test_retag_replaces_stale_projection(self):
        sc = scene(tags=["DOME", "TB", "RL", "7K", "HQ", "Virtual Reality", "CUBEMAP"])
        stash, _, m = self.run_scene(sc, mode="retag")
        m.assert_called_once()
        # CUBEMAP is not ours to remove; everything managed is replaced
        self.assertEqual(names(stash.writes[0]["tag_ids"]),
                         {"FISHEYE", "SBS", "Alpha", "8K", "HQ", "Virtual Reality", "CUBEMAP"})

    def test_overwrite_setting_remeasures(self):
        _, _, m = self.run_scene(scene(tags=["DOME", "SBS"]), overwrite=True)
        m.assert_called_once()

    def test_unmeasurable_leaves_projection_alone(self):
        sc = scene(tags=["MONO"])
        stash, _, _ = self.run_scene(sc, measured=(None, "file missing"))
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"MONO", "8K", "HQ"})

    def test_skip_is_honoured(self):
        for mode in ("untagged", "retag", "clear"):
            with self.subTest(mode=mode):
                stash, out, m = self.run_scene(scene(tags=["VRP: Skip", "DOME"]), mode=mode)
                self.assertEqual(stash.writes, [])
                m.assert_not_called()

    def test_clear_removes_managed_only(self):
        sc = scene(tags=["FISHEYE", "MKX200", "SBS", "Alpha", "8K", "HQ", "Virtual Reality",
                         "CUBEMAP"])
        stash, _, m = self.run_scene(sc, mode="clear")
        m.assert_not_called()
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"Virtual Reality", "CUBEMAP"})


class FlatScan(unittest.TestCase):
    def run_flat(self, name, tags=(), mode="untagged"):
        stash = FakeStash()
        sc = scene(1920, tags=tags, path=f"/media/Movies/{name}")
        return stash, v.process_flat_scene(stash, cfg(), sc, IDS, mode)

    def test_flat_3d_gets_flat_and_layout(self):
        stash, out = self.run_flat("Movie (2010) Half-SBS.mkv", tags=["Drama"])
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"Drama", "FLAT", "SBS"})
        stash, _ = self.run_flat("Movie 3D HOU.mkv")
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"FLAT", "TB"})

    def test_plain_2d_is_left_alone(self):
        stash, out = self.run_flat("Movie (2010).mkv", tags=["MONO", "SBS"])
        self.assertEqual(stash.writes, [])
        self.assertIsNone(out)

    def test_idempotent_and_corrects_layout(self):
        stash, out = self.run_flat("Movie 3D.mkv", tags=["FLAT", "SBS"])
        self.assertEqual(stash.writes, [])
        stash, _ = self.run_flat("Movie 3D OU.mkv", tags=["FLAT", "SBS", "3D Conversion"])
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"FLAT", "TB", "3D Conversion"})

    def test_clear_and_skip(self):
        stash, _ = self.run_flat("Movie 3D.mkv", tags=["FLAT", "SBS", "8K"], mode="clear")
        self.assertEqual(names(stash.writes[0]["tag_ids"]), {"8K"})
        stash, out = self.run_flat("Movie 3D.mkv", tags=["VRP: Skip"])
        self.assertEqual(stash.writes, [])


class Library:
    """A fake Stash GraphQL endpoint over an in-memory scene list."""

    def __init__(self, scenes):
        self.scenes = {s["id"]: s for s in scenes}
        self.writes = []
        self.queries = []

    def call(self, query, variables=None):
        import re as _re
        variables = variables or {}
        if "findScenes" in query:
            f = variables["f"]
            if "MATCHES_REGEX" in query:
                rx = _re.compile(f)
                hits = [s for s in self.scenes.values() if rx.search(s["files"][0]["path"])]
            else:
                hits = [s for s in self.scenes.values() if f in s["files"][0]["path"]]
            self.queries.append(("regex" if "MATCHES_REGEX" in query else "path", f))
            page = variables["p"]
            return {"findScenes": {"count": len(hits), "scenes": hits[(page - 1) * 100:page * 100]}}
        if "findScene(" in query:
            return {"findScene": self.scenes.get(variables["id"])}
        if "sceneUpdate" in query:
            i = variables["i"]
            self.writes.append(i)
            self.scenes[i["id"]]["tags"] = [{"id": t, "name": t.split(":", 1)[1]}
                                            for t in i["tag_ids"]]
            return {"sceneUpdate": {"id": i["id"]}}
        raise AssertionError(query)


class Passes(unittest.TestCase):
    def library(self):
        return Library([
            scene(8192, sid="1", path="/media/VR/Studio/a.mp4"),
            scene(1920, sid="2", path="/media/Movies/Film 3D HSBS.mkv"),
            scene(1920, sid="3", path="/media/Movies/Film.mkv"),
            scene(3840, sid="4", path="/media/VR/Studio/b 3D.mp4"),
        ])

    def run_all(self, lib, mode="untagged", **kw):
        measured = ({v.DOME, v.SBS}, "why")
        with mock.patch.object(v, "measure_projection", return_value=measured), \
                mock.patch.object(v, "log"):
            v.run_all(lib, cfg(**kw), IDS, mode)

    def tags(self, lib, sid):
        return {t["name"] for t in lib.scenes[sid]["tags"]}

    def test_both_passes(self):
        lib = self.library()
        self.run_all(lib)
        self.assertEqual(self.tags(lib, "1"), {"DOME", "SBS", "8K", "HQ"})
        self.assertEqual(self.tags(lib, "2"), {"FLAT", "SBS"})
        self.assertEqual(self.tags(lib, "3"), set())
        # a VR-path file is handled by the VR pass only, never by the filename scan
        self.assertEqual(self.tags(lib, "4"), {"DOME", "SBS"})
        self.assertEqual([q[0] for q in lib.queries], ["path", "regex"])

    def test_flat_scan_can_be_turned_off(self):
        lib = self.library()
        self.run_all(lib, flat3dFilenameScan=False)
        self.assertEqual(self.tags(lib, "2"), set())
        self.assertEqual([q[0] for q in lib.queries], ["path"])

    def test_clear(self):
        lib = self.library()
        self.run_all(lib)
        self.run_all(lib, mode="clear")
        for sid in "1234":
            self.assertEqual(self.tags(lib, sid), set(), sid)

    def test_hook_routes_by_path(self):
        lib = self.library()
        c = cfg()
        with mock.patch.object(v, "Stash", return_value=lib), \
                mock.patch.object(v, "ensure_tags", return_value=IDS), \
                mock.patch.object(v, "load_config", return_value=c), \
                mock.patch.object(v, "measure_projection", return_value=({v.DOME, v.SBS}, "w")), \
                mock.patch.object(v, "log"), mock.patch("builtins.print"):
            for sid, ctx_type in (("1", "Scene.Create.Post"), ("2", "Scene.Update.Post"),
                                  ("3", "Scene.Create.Post")):
                payload = {"server_connection": {}, "args": {"hookContext": {
                    "id": sid, "type": ctx_type, "inputFields": ["id", "title"]}}}
                with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                    v.main()
            # the loop guard: a tag-only update is ignored
            payload = {"server_connection": {}, "args": {"hookContext": {
                "id": "4", "type": "Scene.Update.Post", "inputFields": ["id", "tag_ids"]}}}
            with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                v.main()
        self.assertEqual(self.tags(lib, "1"), {"DOME", "SBS", "8K", "HQ"})
        self.assertEqual(self.tags(lib, "2"), {"FLAT", "SBS"})
        self.assertEqual(self.tags(lib, "3"), set())
        self.assertEqual(self.tags(lib, "4"), set())


class MeasureProjection(unittest.TestCase):
    def test_small_or_missing(self):
        c = cfg()
        self.assertEqual(v.measure_projection(c, scene(files=[])), (None, "no file"))
        self.assertEqual(v.measure_projection(c, scene(1280)), (None, "too small"))
        self.assertEqual(v.measure_projection(c, scene(8192, path="/nonexistent/x.mp4")),
                         (None, "file missing"))
        self.assertEqual(v.measure_projection(c, scene(8192, path=None)), (None, "file missing"))

    def test_watermark_read_only_for_open_fisheye(self):
        res = {"lr": 0.9, "tb": 0.1, "blk_out": 0.9, "blk_in": 0.0, "bbox": 1.0,
               "alpha_lower": 0.0, "matte_red": 0.08, "matte_black": 0.9,
               "matte": True, "frames": 2}
        with mock.patch.object(v.os.path, "exists", return_value=True), \
                mock.patch.object(v, "probe", return_value=res), \
                mock.patch.object(v, "read_fov", return_value=("200", False)) as fov:
            want, why = v.measure_projection(
                cfg(), scene(path="/media/VR/A/Studio - X [Passthrough] [VR].mp4"))
            self.assertEqual(want, {v.FISHEYE, v.MKX200, v.SBS, v.ALPHA})
            self.assertIn("watermark 200", why)
            fov.assert_called_once()

            fov.reset_mock()
            want, _ = v.measure_projection(cfg(), scene(path="/media/VR/A/x_MKX220.mp4"))
            self.assertEqual(want, {v.FISHEYE, v.MKX220, v.SBS, v.ALPHA})
            fov.assert_not_called()

            want, _ = v.measure_projection(cfg(readFovWatermark=False),
                                           scene(path="/media/VR/A/x.mp4"))
            self.assertEqual(want, {v.FISHEYE, v.SBS, v.ALPHA})
            fov.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TestStashAuth(unittest.TestCase):
    def test_api_key_replaces_session_cookie(self):
        s = v.Stash({"SessionCookie": {"Name": "session", "Value": "abc"}})
        self.assertIn("Cookie", s.headers)
        s.use_api_key("k")
        self.assertNotIn("Cookie", s.headers)
        self.assertEqual(s.headers["ApiKey"], "k")

    def test_api_key_defaults_to_empty(self):
        self.assertEqual(v.load_config({})["apiKey"], "")
