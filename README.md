# VR Quality Tags

A [Stash](https://github.com/stashapp/stash) plugin that measures your VR files
and tags what they actually are: projection, fisheye lens, stereo layout, eye
order, the passthrough alpha matte, and a quality tier. It is the companion to
[stash-vr](https://github.com/skeel-x/stash-vr), whose video rules turn these
tags into the right player settings in HereSphere and DeoVR.

Most VR files carry no usable metadata: spherical metadata is rare, filenames
rarely say anything, and the container aspect cannot tell a 180 SBS file from a
360 mono one or from a fisheye. So the plugin decodes two frames per scene and
measures the picture, with filename markers and the SLR FOV watermark taking
precedence where they exist.

## Requirements

* Stash v0.24 or later (plugin sources); the plugin itself only needs Stash's
  plugin tasks and hooks.
* Python 3 as Stash's plugin interpreter. Standard library only: no numpy, no
  PIL, no pip installs.
* `ffmpeg` on the Stash host (default `/usr/bin/ffmpeg`).
* Optional: `tesseract` (default `/usr/bin/tesseract`) to read SLR's burned-in
  FOV watermark, which separates 190, 200 and 220 degree fisheye. Without it
  those scenes get plain `FISHEYE`.

## Install

**From the plugin source (recommended).** In Stash, open *Settings -> Plugins
-> Available plugins -> Add source* and enter

    https://skeel-x.github.io/vrQualityTags/main/index.yml

then install *VR Quality Tags* from that source.

**Manually.** Copy `vrQualityTags.py` and `vrQualityTags.yml` into
`<stash config>/plugins/vrQualityTags/` and click *Reload plugins*.

Then set the **path filter** (below) to the folder that holds your VR scenes and
run *Tag untagged VR scenes* once. New scenes are tagged on scan by the hook.

## What gets tagged

Every tag is a tag in stash-vr's default video rules, so every tag has an
effect in the headset. Tags outside this table are never added or removed.

| Detected | Tags |
|---|---|
| 180 equirect | `DOME` |
| 360 equirect | `SPHERE` |
| fisheye, FOV unknown | `FISHEYE` |
| fisheye 190 (Canon RF 5.2) | `FISHEYE` + `RF52` |
| fisheye 200 | `FISHEYE` + `MKX200` |
| fisheye 220 | `FISHEYE` + `MKX220`, or `VRCA220` when the filename or watermark says VRCA |
| stereo VR | `SBS` / `TB` |
| mono VR | `MONO` (with `DOME` or `SPHERE`) |
| right eye first | `RL` (with `SBS`) |
| flat 2D video | `FLAT` |
| flat stereoscopic 3D | `FLAT` + `SBS` / `FLAT` + `TB` |
| corner-packed alpha matte (passthrough) | `Alpha` |
| quality | `8K` / `7K` / `6K HBR`, each with the parent `HQ` |
| probed, not recognised | `VRP: Unresolved` |
| your opt-out | `VRP: Skip`: add it to a scene and the plugin never touches that scene |

`FLAT` means an ordinary 2D video; `MONO` is only used for mono VR. A scene
without any projection tag also plays flat in stash-vr, which is why files
outside the path filter are left alone unless they look like VR or their name
marks them as 3D (see [Outside the path filter](#outside-the-path-filter)).

`MONO`, `RL`, `MKX220` and `VRCA220` need the matching rules in stash-vr (they
are part of its default rules; an older configuration gets them with Setup's
"Add missing default rules").

The quality tag names and their parent are settings. The projection vocabulary
is fixed, because stash-vr matches it by name.

## How it decides

In order of authority:

1. **Filename markers** (case-insensitive).
   * VR layout, as a whole underscore-delimited segment so that title words do
     not count: `_LR_` `_SBS_` -> `SBS`, `_TB_` `_OU_` -> `TB`,
     `_RL_` -> `RL` + `SBS`, `_MONO_` `_2D_` -> mono, `_180` or `180x180` ->
     `DOME`, `_360` -> `SPHERE`. DeoVR's `3dh` / `3dv` words -> `SBS` / `TB`.
   * Lens, as any word: `FISHEYE190` `RF52` -> `RF52`, `FISHEYE200` `MKX200` ->
     `MKX200`, `FISHEYE220` `MKX220` -> `MKX220`, `VRCA220` (each implies
     `FISHEYE`). A separator before the number also works: `MKX-220`,
     `mkx 200`, `Fisheye_190`, `VRCA-220`.
   * `FISHEYE` on its own -> `FISHEYE` (lens left to the watermark).
   * Flat 3D: `HSBS`, `FSBS`, `LRF`, `Half-SBS`, `Full_SBS`, `HOU`, `Half-OU`,
     `TAB`, `TBF`, `Full-TB` -> `FLAT` + `SBS`/`TB`, anywhere. Outside the path
     filter the looser `3D`, `SBS` and `OU` count as well (`3D` alone means
     side by side).
   * `Passthrough`, `Pass-Through`, `Alpha`, `alphapacked` only make a scene an
     `Alpha` candidate; the pixels decide.
   * Markers that contradict each other cancel out.
2. **SLR FOV watermark.** SLR burns `SLR 190/200/220° FOV FISHEYE` into the
   top of the frame, centred across the dead space between the eyes (older
   releases: near the top of the left eye); both places are read. Nothing measurable separates those lenses, so where the text is
   readable it is the only authority. Up to eight frames are read with
   tesseract and two must agree. The OCR is skipped when it cannot help:
   * the filename already names the lens;
   * the scene does not end up `FISHEYE` (a filename screen marker such as
     `_180` beats the pixels, so `x_LR_180.mp4` is never read even when the
     pixels say fisheye);
   * the scene has the corner alpha matte and neither its file name nor its
     path mentions `SLR` or `SexLikeReal`: SLR's own passthrough releases carry
     the watermark, other studios' passthrough scenes never do.
3. **Frame measurement.** Two frames (40% and 60% into the file, nearest keyframe) are decoded to
   a 256 px thumbnail:
   * *stereo*: how well the two halves match (side by side, and top and
     bottom), searched for parallax; see [Stereo and 360](#stereo-and-360). A
     layout only counts if it leaves a plausible eye shape; when both layouts
     qualify, the better match wins.
   * *eye shape*: a square eye is 180, a 2:1 eye is 360 (but see below), a
     16:9, 16:10 or DCI 4K (1.9:1) eye is flat video. A full side-by-side flat
     3D file (7680x2160, aspect 3.2-3.8) is two 16:9 eyes, so it reads as
     `FLAT` + `SBS`, not as a 180 pair.
   * *360 seam*: a mono 2:1 frame is only `SPHERE` when its right edge
     continues into its left edge.
   * *disc or barrel*: within a square eye, a fisheye is a disc (as wide as
     high, black outside the inscribed circle) and a 180 equirect a barrel
     (full width, wider than high). Darkness alone cannot tell them apart, both
     leave the corners black; the shape of the lit region can.
   * *corner matte*: see below.
   * *packed-alpha guard*: a near-binary lower half is a matte, not a second
     eye; such a file gets `VRP: Unresolved` instead of a wrong `TB`.

## Stereo and 360

The two eyes of a stereo pair are not the same picture. Everything is shifted
sideways by its parallax, and in a 180 close-up the performer can be shifted by
several per cent of the eye width while the room behind barely moves. Matching
the halves pixel for pixel therefore reads many close-ups as mono, and a mono
2:1 frame looks exactly like a 360.

**Stereo match.** One eye is cut into 32 px tiles (on the 256 px thumbnail);
tiles without texture (standard deviation below 6: black borders, fades) are
skipped. Each remaining tile is correlated with the other eye at horizontal
shifts of up to 6% of the eye width, coarse to fine, and keeps its best match.
Parallax is horizontal in both layouts, so top/bottom pairs are searched
sideways as well. The match of a layout is the median over the tiles of both
frames, so one frame that is a fade or a title card cannot drag a pair down; a
frame with textured tiles in less than a quarter of the eye is not measured at
all. Side by side is accepted at a median of 0.55, top and bottom at 0.60.

**360 seam.** In a 360 equirect the last column and the first are neighbours
on the sphere. The seam ratio is the mean difference between those two
columns divided by the median difference between columns half the picture
apart: near 0 for a 360, around 1 when the edges are unrelated. Edges with
almost no texture (standard deviation below 5, such as black borders) give no
answer. Every frame that gives an answer must close the seam; the threshold is
0.25.

A mono 2:1 frame (no layout accepted) is then:

* `SPHERE` + `MONO` when the seam closes;
* otherwise `DOME` + `SBS` when the stereo match is at least 0.30: a 180 pair
  whose halves match poorly, which is far more common than a 360 without a
  seam;
* otherwise `VRP: Unresolved`: halves with nothing in common and no seam are
  neither (typically flat video in a 2:1 frame).

A 2:1 frame whose top and bottom halves match is a top/bottom 360 with each
eye squeezed to 4:1 (early 360 releases) when each eye's seam closes; it gets
`SPHERE` + `TB`.

## Outside the path filter

VR files do not always live under the path filter. A scene outside it is
measured like any VR scene when its primary file looks like VR from Stash's
metadata alone (setting `measureVrShapedOutside`, on by default):

* the frame is at least 3840 wide and its aspect (width / height) is within
  0.05 of 2.0 (a side-by-side 180 or a 360 file) or of 1.0 (a top-bottom 180
  file), or
* the file name carries a VR marker: a screen or lens marker from the list
  above (`_180`, `_360`, `FISHEYE`, `MKX200`, `RF52`, ...) or a VR word (`VR`,
  `VR180`, `VR360`, `180x180`, `3dh`, `3dv`).

A name with a flat 3D marker (`HSBS`, `Half-OU`, ...) never counts. Deciding
this needs no decoding: task runs ask Stash for scenes with a large frame or a
matching path and check each one's metadata.

Every other file outside the path filter is only checked for a flat 3D name
(setting `flat3dFilenameScan`); nothing there is decoded.

## Passthrough: the corner-packed alpha matte

Passthrough (AR) VR scenes are usually fisheye stereo with the alpha matte
packed into the corners the two discs leave free: solid saturated red
silhouettes of the performer, up to eight of them. HereSphere reads this with
its alpha-packed mask; stash-vr maps the `Alpha` tag to a passthrough
background. Plain fisheye scenes have black corners there.

A frame carries the matte when, over the pixels outside each eye's inscribed
circle,

* at least 0.6% are matte red (`R >= 96`, `G` and `B` at most `0.30 R`), and
* red plus black (`max(R,G,B) < 32`) make up at least 80% (the corners hold
  nothing else).

`Alpha` requires the matte in **both** frames. Where it is found, the corners
are blacked out before the projection is measured, and a square-eye stereo
file with a matte is treated as a fisheye.

The filename is deliberately not trusted for this: many passthrough files are
not named as such, and a file named "Passthrough" can be an ordinary 180 scene.

## How detection was calibrated

The shape rules (eye aspect, disc versus barrel, the packed-alpha guard) were
validated against a hand-checked set of scenes of every layout; the disc rule
is `bbox in [0.94, 1.06] and blk_out >= 0.72`.

The stereo and seam thresholds come from two frames each of about 140 scenes:
180 side-by-side pairs (among them several dozen close-ups that pixel-aligned
correlation had called mono), 360 top/bottom files, fisheye and passthrough
pairs, flat 2D and flat 3D files, and flat video in 2:1 frames. The eyes of
the 360 top/bottom files served as genuine mono 360 pictures.

| Group | Stereo match (median over tiles) | Seam ratio |
|---|---|---|
| 180 side-by-side pairs, close-ups included | 0.44 - 0.96 | 0.84 - 1.22 where the match is below 0.55 |
| fisheye and passthrough pairs | 0.62 - 1.00 | (not needed) |
| one eye of a 360 (mono 360 picture) | -0.01 - 0.46 | 0.00 - 0.13 |
| flat video in a 2:1 frame | 0.10 - 0.14 | 1.47 - 1.59 |

Pixel-aligned, the same 180 close-ups scored as low as 0.33, well inside the
mono range. The seam ratio of 180 pairs with a good stereo match reaches down
to about 0.35 (dark, similar walls at both outer edges), which is why the seam
only decides between mono readings and never overrides an accepted stereo
layout.

The corner-matte thresholds come from two frames each of several dozen
passthrough scenes from different studios, a random sample of plain fisheye
scenes, and hard negatives:

| Group | Red share per frame | Red + black |
|---|---|---|
| passthrough with a corner matte | 0.013 - 0.34 | >= 0.94 |
| plain fisheye | <= 0.0035 (a studio logo between the discs) | 0.83 - 1.0 |
| 180 scene with a red sheet reaching into the corners | 0.45 | 0.48 |
| 180 scene with magenta light in the corners | 0.00 | 0.40 |

The red threshold sits about 2x below the faintest real matte (performer far
from the camera) and 1.7x above the highest plain fisheye; the red + black
condition is what rejects red picture content in the corners. A compilation
whose clips mix formats is left without `Alpha` because both frames must agree.

## Tasks and hook

* **Tag untagged VR scenes**: measures every scene under the path filter that
  has no projection tag yet (`DOME SPHERE FISHEYE FLAT RF52 MKX200 MKX220
  VRCA220 CUBEMAP EAC VRP: Unresolved`), refreshes the quality tag of every
  scene there, measures VR-shaped files outside the path filter, and runs the
  flat 3D filename check elsewhere.
* **Re-measure and retag all VR scenes**: measures everything under the path
  filter (and the VR-shaped files outside it) again and replaces every tag the
  plugin manages. The run is resumable: scenes are visited in ascending id
  order and after each one the plugin records the scene id and the run's start
  time in `vrQualityTags.state.json` in the plugin directory. Running the task
  again after an interruption (a cancelled task, a Stash restart, an expired
  session) logs "resuming after scene N" and continues from there, as long as
  the interrupted run started less than 7 days ago. A run that completes
  deletes the file.
* **Restart the retag from the beginning**: the same retag, but it ignores any
  saved progress and starts from the first scene.
* **Remove all managed tags**: detaches the managed tags; the tags themselves
  are kept.
* **Tag on scan** (hook on scene create and update): the "untagged" logic for
  one scene. Updates that only changed tags are ignored, so the plugin's own
  writes do not trigger it again.

Task progress is reported to Stash after every scene. Writes are idempotent: a
scene whose tags are already right is not written to.
`VRP: Skip` makes every task and the hook leave a scene alone.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| Path filter | `/VR/` | only scenes whose file path contains this are measured. An empty value means the default; `/` matches every scene (then ordinary 2D videos get `FLAT`). |
| Read SLR FOV watermark | on | OCR the watermark to pick `RF52`/`MKX200`/`MKX220` (skipped where it cannot help, see "How it decides") |
| Re-measure already-tagged scenes | off | measure scenes that already have a projection tag on every run and hook |
| Minimum width (px) | 1920 | narrower files are not measured |
| Measure VR-shaped files outside the path filter | on | measure a scene outside the path filter when its file is a 2:1 or square frame at least 3840 wide or has a VR marker in its name |
| Tag flat 3D files outside the path filter | on | the flat 3D filename check |
| ffmpeg path, tesseract path | `/usr/bin/...` | |
| Parent tag, 8K/7K/6K tag names | `HQ`, `8K`, `7K`, `6K HBR` | quality tag names |
| 8K/7K/6K minimum width | 7680, 7000, 5760 | tier boundaries |
| 6K minimum bitrate (Mbit/s) | 40 | a 6K file below this is treated as an upscale |
| Stash API key (for long tasks) | empty | Stash ends a plugin task's session after an hour, so a full retag of a large library stops part way with `401 Unauthorized`. Paste your API key (*Settings -> Security*) and the plugin authenticates with it instead. |

Stash shows an untouched on/off setting as off; the plugin treats an untouched
setting as its default (on for the watermark, the VR-shaped check and the flat
3D check). Switch it
on and off once to store an explicit value.

## Quality tiers

| Tag | Condition |
|---|---|
| `8K` | width >= 7680 |
| `7K` | width 7000 - 7679 |
| `6K HBR` | width 5760 - 6999 and bitrate >= 40 Mbit/s |
| `HQ` | parent of the three, also applied directly |

Stash-box tags like `8K Available` describe what a studio sells, not your file;
these come from the file itself (the largest file of the scene). The bitrate
condition only applies to the 6K band, which is where upscales tend to hide.
`HQ` is applied directly too, because a VR bridge that forwards only directly
assigned tags would otherwise never see it.

## Limitations

* Only two frames are measured. A scene that changes format halfway (a
  compilation) is classified by those two frames.
* A mono 360 whose halves happen to match (0.55 or more) is read as a 180
  pair; the seam is only consulted when no stereo layout is accepted.
* The FOV of a fisheye cannot be measured. Without a filename marker or a
  readable SLR watermark it stays plain `FISHEYE`.
* Flat 3D outside the path filter is recognised by name only. A title word such
  as "3D" in an ordinary 2D video's name is read as side by side; add
  `VRP: Skip` to such a scene.
* The corner-matte test knows one packing: red silhouettes in the corners of
  fisheye stereo. Mattes packed elsewhere are not tagged `Alpha`.
* Measuring costs two decoded frames per scene (a few seconds for 8K HEVC on a
  network share) plus, for a fisheye whose lens is still open, up to eight
  small crops for the watermark.

## Development

    python3 -m unittest -v

runs the tests (standard library only). `python3 build.py` builds the plugin
source into `_site/main/` (`index.yml` plus a reproducible `vrQualityTags.zip`);
the workflow in `.github/workflows/publish.yml` runs the tests, builds, and
publishes `_site/` to GitHub Pages on every push to `main`, which is what the
source URL above points to. In the repository settings, Pages must be set to
deploy from GitHub Actions.

Licence: [GNU AGPL-3.0](LICENSE), the same licence as Stash.
