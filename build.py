#!/usr/bin/env python3
"""Build the Stash plugin source.

    python3 build.py [outdir]        (default: _site/main)

Writes <outdir>/vrQualityTags.zip and <outdir>/index.yml in the layout Stash
reads from a plugin source (Settings > Plugins > Available plugins). Published
through GitHub Pages, the source URL is <pages url>/main/index.yml.

The zip is reproducible: fixed timestamps and file order, so an unchanged
plugin gives an unchanged sha256.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

PLUGIN_ID = "vrQualityTags"
FILES = ("vrQualityTags.yml", "vrQualityTags.py", "README.md")
ROOT = os.path.dirname(os.path.abspath(__file__))


def manifest_field(text, key):
    m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
    if not m:
        raise SystemExit(f"{key} missing from {PLUGIN_ID}.yml")
    return m.group(1).strip().strip('"')


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    zpath = os.path.join(outdir, f"{PLUGIN_ID}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(os.path.join(ROOT, name), "rb") as f:
                z.writestr(info, f.read())
    with open(zpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(ROOT, f"{PLUGIN_ID}.yml"), encoding="utf-8") as f:
        yml = f.read()
    version = manifest_field(yml, "version")
    commit = git("log", "-n", "1", "--pretty=format:%h")
    if commit:
        version = f"{version}-{commit}"
    date = git("log", "-n", "1", "--date=format-local:%Y-%m-%d %H:%M:%S",
               "--pretty=format:%ad") or "1970-01-01 00:00:00"

    # json.dumps gives valid YAML double-quoted scalars
    index = "\n".join([
        f"- id: {PLUGIN_ID}",
        f"  name: {json.dumps(manifest_field(yml, 'name'))}",
        "  metadata:",
        f"    description: {json.dumps(manifest_field(yml, 'description'))}",
        f"  version: {json.dumps(version)}",
        f"  date: {json.dumps(date)}",
        f"  path: {PLUGIN_ID}.zip",
        f"  sha256: {sha}",
        "  requires: []",
        "",
    ])
    with open(os.path.join(outdir, "index.yml"), "w", encoding="utf-8") as f:
        f.write(index)
    print(index, end="")


if __name__ == "__main__":
    os.environ["TZ"] = "UTC0"
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "_site", "main"))
