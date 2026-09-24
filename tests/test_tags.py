import io
import json
import os
import time
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
        self.assertIs(c["measureVrShapedOutside"], True)
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
                kind = "regex"
                rx = _re.compile(f)
                hits = [s for s in self.scenes.values() if rx.search(s["files"][0]["path"])]
            elif "resolution" in query:
                # Stash: FULL_HD GREATER_THAN is MIN(width, height) > 1439
                kind = "resolution"
                hits = [s for s in self.scenes.values()
                        if min(s["files"][0]["width"], s["files"][0]["height"]) > 1439
                        and f not in s["files"][0]["path"]]
            else:
                kind = "path"
                hits = [s for s in self.scenes.values() if f in s["files"][0]["path"]]
            hits.sort(key=lambda s: int(s["id"]))
            self.queries.append((kind, f))
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
        self.assertEqual([q[0] for q in lib.queries],
                         ["path", "resolution", "regex", "regex"])

    def test_flat_scan_can_be_turned_off(self):
        lib = self.library()
        self.run_all(lib, flat3dFilenameScan=False, measureVrShapedOutside=False)
        self.assertEqual(self.tags(lib, "2"), set())
        self.assertEqual([q[0] for q in lib.queries], ["path"])

    def vr_outside_library(self):
        return Library([
            scene(8192, sid="1", path="/media/VR/Studio/a.mp4"),
            # 2:1 and square frames at least 3840 wide
            scene(files=[{"width": 5760, "height": 2880, "duration": 600, "bit_rate": 5e7,
                          "size": 1, "path": "/mnt/adult/Interactive/AHE VR/x.mp4"}], sid="5"),
            scene(files=[{"width": 4096, "height": 4096, "duration": 600, "bit_rate": 5e7,
                          "size": 1, "path": "/mnt/adult/Other/y.mp4"}], sid="6"),
            # VR marker in the name, frame too small for the shape rule
            scene(files=[{"width": 3840, "height": 1080, "duration": 600, "bit_rate": 5e7,
                          "size": 1, "path": "/mnt/adult/SLR_Studio_Title_LR_180.mp4"}],
                  sid="7"),
            # ordinary 4K and a 4K flat 3D film: not VR
            scene(files=[{"width": 3840, "height": 2160, "duration": 600, "bit_rate": 5e7,
                          "size": 1, "path": "/media/Movies/Film.mkv"}], sid="8"),
            scene(files=[{"width": 3840, "height": 1920, "duration": 600, "bit_rate": 5e7,
                          "size": 1, "path": "/media/Movies/Film HSBS.mkv"}], sid="9"),
            # 2:1 but below 3840
            scene(files=[{"width": 2880, "height": 1440, "duration": 600, "bit_rate": 5e7,
                          "size": 1, "path": "/media/Clips/z.mp4"}], sid="10"),
        ])

    def test_vr_shaped_outside_the_filter_is_measured(self):
        lib = self.vr_outside_library()
        with mock.patch.object(v, "measure_projection",
                               return_value=({v.DOME, v.SBS}, "why")) as m, \
                mock.patch.object(v, "log"):
            v.run_all(lib, cfg(), IDS, "untagged")
        measured = sorted(int(c.args[1]["id"]) for c in m.call_args_list)
        self.assertEqual(measured, [1, 5, 6, 7])
        self.assertEqual(self.tags(lib, "5"), {"DOME", "SBS", "6K HBR", "HQ"})
        self.assertEqual(self.tags(lib, "8"), set())
        self.assertEqual(self.tags(lib, "9"), {"FLAT", "SBS"})
        self.assertEqual(self.tags(lib, "10"), set())

    def test_vr_shaped_outside_can_be_turned_off(self):
        lib = self.vr_outside_library()
        with mock.patch.object(v, "measure_projection",
                               return_value=({v.DOME, v.SBS}, "why")) as m, \
                mock.patch.object(v, "log"):
            v.run_all(lib, cfg(measureVrShapedOutside=False), IDS, "untagged")
        self.assertEqual([c.args[1]["id"] for c in m.call_args_list], ["1"])
        self.assertEqual(self.tags(lib, "5"), set())

    def test_candidates_are_unique_and_in_id_order(self):
        lib = self.vr_outside_library()
        with mock.patch.object(v, "log"):
            todo = v.candidates(lib, cfg())
        ids = [int(sc["id"]) for sc, _, _ in todo]
        self.assertEqual(ids, [1, 5, 6, 7, 9])
        self.assertEqual([k for _, _, k in todo],
                         ["VR", "VR-shaped", "VR-shaped", "VR-shaped", "flat 3D"])

    def test_hook_measures_vr_shaped_outside(self):
        c = cfg()
        lib = self.vr_outside_library()
        self.assertIs(v.route(c, lib.scenes["5"]), v.process_scene)
        self.assertIs(v.route(c, lib.scenes["7"]), v.process_scene)
        self.assertIs(v.route(c, lib.scenes["8"]), v.process_flat_scene)
        self.assertIs(v.route(cfg(measureVrShapedOutside=False), lib.scenes["5"]),
                      v.process_flat_scene)
        self.assertIsNone(v.route(cfg(measureVrShapedOutside=False,
                                      flat3dFilenameScan=False), lib.scenes["5"]))
        self.assertIsNone(v.route(c, None))


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


class LooksVr(unittest.TestCase):
    def shaped(self, w, h, path="/media/Other/x.mp4"):
        return v.looks_vr(scene(files=[{"width": w, "height": h, "path": path}]))

    def test_frame_shape(self):
        self.assertTrue(self.shaped(3840, 1920))
        self.assertTrue(self.shaped(8192, 4096))
        self.assertTrue(self.shaped(5760, 2880))
        self.assertTrue(self.shaped(3840, 3840))
        self.assertTrue(self.shaped(3840, 1900))       # 2.02, within 0.05
        self.assertFalse(self.shaped(3840, 1800))      # 2.13
        self.assertFalse(self.shaped(3840, 2160))      # 16:9
        self.assertFalse(self.shaped(2880, 1440))      # too narrow
        self.assertFalse(self.shaped(3840, 0))

    def test_name_markers(self):
        for name in ("SLR_Studio_Title_LR_180.mp4", "Title 3dh.mp4", "Title [VR].mp4",
                     "Title FISHEYE.mp4", "Title_MKX200.mp4", "Title_360_x.mp4",
                     "Title VR180.mp4"):
            with self.subTest(name=name):
                self.assertTrue(self.shaped(1920, 1080, "/media/Other/" + name))
        for name in ("Film.mkv", "Top 180 Moments.mp4", "Film 3D HSBS VR.mkv", "Every.mp4"):
            with self.subTest(name=name):
                self.assertFalse(self.shaped(1920, 1080, "/media/Other/" + name))

    def test_no_file(self):
        self.assertFalse(v.looks_vr(scene(files=[])))


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


class Resume(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, v.STATE_FILE)

    def tearDown(self):
        self.dir.cleanup()

    def library(self):
        return Library([scene(8192, sid=str(i), path=f"/media/VR/S/{i}.mp4")
                        for i in (3, 1, 12, 7, 20)])

    def run_retag(self, lib, state, fail_on=None):
        seen, logs = [], []

        def measure(c, sc):
            seen.append(sc["id"])
            if sc["id"] == fail_on:
                raise KeyboardInterrupt           # the task is killed mid-scene
            return {v.DOME, v.SBS}, "why"

        with mock.patch.object(v, "measure_projection", side_effect=measure), \
                mock.patch.object(v, "log", side_effect=lambda lv, m: logs.append((lv, m))):
            try:
                v.run_all(lib, cfg(), IDS, "retag", state)
            except KeyboardInterrupt:
                pass
        return seen, logs

    def saved(self):
        with open(self.path) as f:
            return json.load(f)

    def test_state_path(self):
        self.assertEqual(v.state_path({"PluginDir": "/x/plugins/vrq"}),
                         os.path.join("/x/plugins/vrq", v.STATE_FILE))
        here = os.path.dirname(os.path.abspath(v.__file__))
        self.assertEqual(v.state_path({}), os.path.join(here, v.STATE_FILE))
        self.assertEqual(v.state_path(None), os.path.join(here, v.STATE_FILE))

    def test_progress_per_scene_in_id_order(self):
        lib = self.library()
        seen, logs = self.run_retag(lib, v.RetagState(self.path, 1000.0))
        self.assertEqual(seen, ["1", "3", "7", "12", "20"])
        prog = [m for lv, m in logs if lv == "p"]
        self.assertEqual(prog, ["0.0000", "0.2000", "0.4000", "0.6000", "0.8000", "1.0000"])
        for p in prog:
            self.assertTrue(0.0 <= float(p) <= 1.0)
        # a completed run leaves no state behind
        self.assertFalse(os.path.exists(self.path))

    def test_interrupted_run_resumes_after_last_scene(self):
        lib = self.library()
        now = time.time()
        seen, _ = self.run_retag(lib, v.RetagState(self.path, now - 60), fail_on="12")
        self.assertEqual(seen, ["1", "3", "7", "12"])
        self.assertEqual(self.saved(), {"started": now - 60, "last_id": 7})

        state = v.RetagState.load(self.path)
        self.assertEqual((state.last_id, state.started), (7, now - 60))
        seen, logs = self.run_retag(lib, state)
        self.assertEqual(seen, ["12", "20"])
        self.assertTrue(any("resuming after scene 7" in m for _, m in logs))
        self.assertIn(("p", "0.6000"), logs)
        self.assertFalse(os.path.exists(self.path))

    def test_stale_or_broken_state_is_ignored(self):
        now = 10 * 86400.0

        def load(data):
            with open(self.path, "w") as f:
                f.write(data if isinstance(data, str) else json.dumps(data))
            return v.RetagState.load(self.path, now)

        self.assertIsNone(v.RetagState.load(self.path, now))          # no file
        self.assertIsNotNone(load({"started": now - 6.9 * 86400, "last_id": 5}))
        self.assertIsNone(load({"started": now - 7 * 86400, "last_id": 5}))
        self.assertIsNone(load({"started": now + 60, "last_id": 5}))  # clock went back
        self.assertIsNone(load({"started": now, "last_id": "5"}))
        self.assertIsNone(load({"started": True, "last_id": 5}))
        self.assertIsNone(load({"last_id": 5}))
        self.assertIsNone(load([1, 2]))
        self.assertIsNone(load("{not json"))

    def test_unwritable_state_warns_once_and_carries_on(self):
        lib = self.library()
        state = v.RetagState(os.path.join(self.dir.name, "missing", "s.json"))
        seen, logs = self.run_retag(lib, state)
        self.assertEqual(len(seen), 5)
        self.assertEqual(sum(1 for lv, _ in logs if lv == "w"), 1)

    def run_main(self, lib, mode):
        payload = {"server_connection": {"PluginDir": self.dir.name}, "args": {"mode": mode}}
        seen = []

        def measure(c, sc):
            seen.append(sc["id"])
            return {v.DOME, v.SBS}, "why"

        with mock.patch.object(v, "Stash", return_value=lib), \
                mock.patch.object(v, "ensure_tags", return_value=IDS), \
                mock.patch.object(v, "load_config", return_value=cfg()), \
                mock.patch.object(v, "measure_projection", side_effect=measure), \
                mock.patch.object(v, "log"), mock.patch("builtins.print"), \
                mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
            v.main()
        return seen

    def test_retag_task_resumes_and_fresh_task_does_not(self):
        with open(self.path, "w") as f:
            json.dump({"started": time.time() - 3600, "last_id": 7}, f)
        self.assertEqual(self.run_main(self.library(), "retag"), ["12", "20"])
        self.assertFalse(os.path.exists(self.path))

        with open(self.path, "w") as f:
            json.dump({"started": time.time() - 3600, "last_id": 7}, f)
        self.assertEqual(self.run_main(self.library(), "retag_fresh"),
                         ["1", "3", "7", "12", "20"])
        self.assertFalse(os.path.exists(self.path))

    def test_untagged_and_clear_ignore_the_state(self):
        with open(self.path, "w") as f:
            json.dump({"started": time.time() - 3600, "last_id": 7}, f)
        self.assertEqual(self.run_main(self.library(), "untagged"),
                         ["1", "3", "7", "12", "20"])
        self.assertTrue(os.path.exists(self.path))
