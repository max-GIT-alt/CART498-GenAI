"""Rebuild a session gallery (and its export archive) from the command line."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "ComfyUI-Session"))

from session import build_export_zip, build_html  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"),
                        help="ComfyUI output directory")
    parser.add_argument("--session", default="session",
                        help="session folder name inside the output directory")
    parser.add_argument("--out", default=None,
                        help="write the .zip here instead of inside the session folder")
    parser.add_argument("--no-html", action="store_true",
                        help="only build the archive, do not rewrite session.html")
    args = parser.parse_args()

    base = os.path.join(os.path.abspath(args.output_dir), args.session)
    manifest = os.path.join(base, "session.json")
    if not os.path.isfile(manifest):
        raise SystemExit(f"no session manifest at {manifest}")
    with open(manifest, encoding="utf-8") as f:
        runs = json.load(f).get("runs", [])

    archive, written = build_export_zip(base, runs, path=args.out)
    print(f"wrote {archive} ({written} files, {os.path.getsize(archive) / 1e6:.1f} MB)")
    if not args.no_html:
        build_html(base, runs)
        print(f"wrote {os.path.join(base, 'session.html')}")


if __name__ == "__main__":
    main()
