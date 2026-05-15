#!/usr/bin/env python3
"""Download the pretrained NoMaD checkpoint.

The upstream weights are hosted on Google Drive (see visualnav-transformer
README). Direct URL changes occasionally — verify in the upstream repo if
this 404s.
"""

import argparse
import os
import sys
from pathlib import Path

# Update this if the upstream URL changes. As of NoMaD release the file
# is published as `nomad.pth` from the visualnav-transformer release page.
DEFAULT_URL = "https://drive.google.com/uc?id=NOMAD_FILE_ID_HERE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=os.path.expanduser("~/visualnav-transformer/checkpoints/nomad.pth"),
        help="Destination path for nomad.pth",
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help="Direct download URL (Google Drive or otherwise)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"Checkpoint already present at {output} — skipping download.")
        return

    print(f"Downloading {args.url} -> {output}")
    if "drive.google.com" in args.url:
        try:
            import gdown
        except ImportError:
            print("gdown not installed. Install with: pip install gdown", file=sys.stderr)
            sys.exit(1)
        gdown.download(args.url, str(output), quiet=False)
    else:
        import urllib.request
        urllib.request.urlretrieve(args.url, str(output))

    if not output.exists() or output.stat().st_size < 1_000_000:
        print(f"Download failed or file too small: {output}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
