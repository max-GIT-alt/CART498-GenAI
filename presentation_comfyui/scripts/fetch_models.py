"""Download the micro LCM checkpoint used by the live demo.

The full-flow SDXL and Qwen weights are not downloaded by this script. If you
already have those local models, symlink them into ``models/`` as described in
``models/README.md`` and the matrix workflows will use them.

    python scripts/fetch_models.py            # ~2.0 GiB micro model
    python scripts/fetch_models.py --verify   # also check SHA-256
    python scripts/fetch_models.py --list
"""
import argparse
import hashlib
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(ROOT, "models")
CHUNK = 8 << 20

MODELS = {
    "micro": {
        "file": "checkpoints/DreamShaper8_LCM.safetensors",
        "url": "https://huggingface.co/Lykon/dreamshaper-8-lcm/resolve/main/"
               "DreamShaper8_LCM.safetensors",
        "size": 2_133_804_992,
        "sha256": "a4f3e1526c5dc4fcbe342f5c410d83ae202c7a415fcefcbb92e0f93fcd0a87c3",
        "note": "DreamShaper 8 LCM, 4-step micro model",
    },
}
PROFILES = {"micro": ["micro"]}


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target(root, spec):
    return os.path.join(root, spec["file"])


def display(spec):
    return os.path.join("models", spec["file"])


def ready(path, spec):
    return os.path.isfile(path) and os.path.getsize(path) == spec["size"]


def download(spec, root, force=False, verify=False):
    dest = target(root, spec)
    if not force and ready(dest, spec):
        if verify:
            print(f"  verify   {display(spec)}")
            digest = sha256(dest)
            if digest != spec["sha256"]:
                raise SystemExit(
                    f"sha256 mismatch for {dest}\n  got      {digest}\n"
                    f"  expected {spec['sha256']}\n"
                    "Use --force to download the published file."
                )
        print(f"  ok       {display(spec)} ({human_size(spec['size'])})")
        return False

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = {"User-Agent": "comfyui-anchor-demo/1.0"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"  resume   {display(spec)} at {human_size(have)}")

    request = urllib.request.Request(spec["url"], headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"download failed ({exc.code}) for {spec['url']}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"download failed for {spec['url']}: {exc.reason}") from exc

    mode = "ab"
    if have and response.status != 206:
        have = 0
        mode = "wb"
    print(f"  get      {display(spec)} ({human_size(spec['size'])})")
    started = time.time()
    done = have
    next_report = have + max(1, spec["size"] // 20)
    with response, open(part, mode) as handle:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if done >= next_report:
                elapsed = max(0.01, time.time() - started)
                rate = (done - have) / elapsed
                print(f"    {done / spec['size'] * 100:5.1f}%  "
                      f"{human_size(done)}  {human_size(rate)}/s")
                next_report += max(1, spec["size"] // 20)

    size = os.path.getsize(part)
    if size != spec["size"]:
        raise SystemExit(
            f"size mismatch for {part}: got {size}, expected {spec['size']}; "
            "delete the .part file and retry"
        )
    os.replace(part, dest)
    if verify:
        print(f"  verify   {display(spec)}")
        digest = sha256(dest)
        if digest != spec["sha256"]:
            raise SystemExit(
                f"sha256 mismatch for {dest}\n  got      {digest}\n"
                f"  expected {spec['sha256']}"
            )
    print(f"  done     {display(spec)}")
    return True


def print_list():
    print("profiles:")
    for name, keys in PROFILES.items():
        total = sum(MODELS[key]["size"] for key in keys)
        print(f"  {name:6s} {human_size(total):>10s}  {len(keys)} files")
    print("\nmodels:")
    for key, spec in MODELS.items():
        print(f"  {key:12s} {human_size(spec['size']):>10s}  {spec['note']}")
        print(f"               {display(spec)}")
        print(f"               {spec['url']}")
    print("\nOptional SDXL/Qwen weights are not downloaded; see models/README.md.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="micro", choices=sorted(PROFILES),
                        help="model profile (only micro is packaged)")
    parser.add_argument("--model", action="append", choices=sorted(MODELS),
                        help="download one model by name; may be repeated")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"model root directory (default: {DEFAULT_ROOT})")
    parser.add_argument("--force", action="store_true", help="download again even if present")
    parser.add_argument("--verify", action="store_true", help="verify SHA-256")
    parser.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    parser.add_argument("--list", action="store_true", help="list the packaged model")
    args = parser.parse_args(argv)

    if args.list:
        print_list()
        return

    keys = args.model or PROFILES[args.profile]
    root = os.path.abspath(os.path.expanduser(args.root))
    total = sum(MODELS[key]["size"] for key in keys)
    print(f"models root: {root}")
    print(f"selected: {', '.join(keys)} ({human_size(total)})")

    for key in keys:
        spec = MODELS[key]
        print(f"\n[{key}] {spec['note']}")
        if args.dry_run:
            status = "present" if ready(target(root, spec), spec) else "missing"
            print(f"  {status:8s} {display(spec)}")
            continue
        download(spec, root, force=args.force, verify=args.verify)

    if args.dry_run:
        print("\ndry run: no files downloaded.")
    else:
        print("\nselected models are ready.")


if __name__ == "__main__":
    main()
