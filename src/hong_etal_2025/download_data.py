#!/usr/bin/env python3
"""Download the subset of the Hong et al. (2025) OSF dataset needed by these notebooks.

Downloads into  <repo_root>/data/hong_etal_2025/  (git-ignored),
preserving OSF folder structure.

Files fetched (always)
----------------------
- Organized data and model predictions/sub{N}/Thres_ellipses_sub{N}.csv
  Threshold-ellipse covariance matrices on the 7×7 reference grid.

- Calibration and transformation/Transformation matrices/*.csv
  Monitor calibration matrices needed for the 2DW ↔ RGB colour conversion.

Files fetched (with --fits)
---------------------------
- Organized data and model predictions/sub{N}/analyzed data files with class objects/
  Main fit and 120 bootstrap Wishart process fit objects (may be large).

Usage
-----
    python src/hong_etal_2025/download_data.py [--subjects 1 2 4 ...] [--fits]

No third-party packages required — stdlib only.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OSF_NODE = "k27js"
OSF_API = "https://api.osf.io/v2"
REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "hong_etal_2025"

DEFAULT_SUBJECTS = [1]  # sub1 = subject CH


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code} for {url}", file=sys.stderr)
        raise


def _download_file(download_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip  {dest.relative_to(REPO_ROOT)}  (already exists)")
        return
    print(f"  fetch {dest.relative_to(REPO_ROOT)}")
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)


def _list_folder(folder_url: str) -> list[Any]:
    """Return all items (data list) from a paginated OSF folder listing."""
    items: list[Any] = []
    url = folder_url
    while url:
        data = _get_json(url)
        items.extend(data.get("data", []))
        url = data.get("links", {}).get("next") or ""
    return items


def _find_subfolder(items: list[Any], name: str) -> str:
    """Return the files-listing URL for the named sub-folder, or ''."""
    for item in items:
        if (
            item.get("attributes", {}).get("name") == name
            and item.get("attributes", {}).get("kind") == "folder"
        ):
            return str(
                item.get("relationships", {})
                .get("files", {})
                .get("links", {})
                .get("related", {})
                .get("href", "")
            )
    return ""


def download_transformation_matrices(root_items: list[Any], data_dir: Path) -> None:
    cal_url = _find_subfolder(root_items, "Calibration and transformation")
    if not cal_url:
        print(
            "  warn: 'Calibration and transformation' folder not found", file=sys.stderr
        )
        return
    cal_items = _list_folder(cal_url)
    xform_url = _find_subfolder(cal_items, "Transformation matrices")
    if not xform_url:
        print("  warn: 'Transformation matrices' subfolder not found", file=sys.stderr)
        return
    dest_dir = data_dir / "Calibration and transformation" / "Transformation matrices"
    for item in _list_folder(xform_url):
        if item.get("attributes", {}).get("kind") == "file":
            name = item["attributes"]["name"]
            dl_url = item["links"].get("download", "")
            if dl_url:
                _download_file(dl_url, dest_dir / name)


def download_thres_ellipses(
    root_items: list[Any], data_dir: Path, subjects: list[Any]
) -> None:
    org_url = _find_subfolder(root_items, "Organized data and model predictions")
    if not org_url:
        print(
            "  warn: 'Organized data and model predictions' folder not found",
            file=sys.stderr,
        )
        return
    org_items = _list_folder(org_url)
    for sub_n in subjects:
        sub_name = f"sub{sub_n}"
        sub_url = _find_subfolder(org_items, sub_name)
        if not sub_url:
            print(f"  warn: folder '{sub_name}' not found in OSF", file=sys.stderr)
            continue
        sub_items = _list_folder(sub_url)
        target = f"Thres_ellipses_sub{sub_n}.csv"
        found = False
        for item in sub_items:
            if (
                item.get("attributes", {}).get("kind") == "file"
                and item["attributes"]["name"] == target
            ):
                dl_url = item["links"].get("download", "")
                dest = (
                    data_dir
                    / "Organized data and model predictions"
                    / sub_name
                    / target
                )
                _download_file(dl_url, dest)
                found = True
                break
        if not found:
            print(f"  warn: '{target}' not found under '{sub_name}'", file=sys.stderr)


def download_fit_pkls(
    root_items: list[Any], data_dir: Path, subjects: list[Any]
) -> None:
    """Download main-fit and bootstrap pkl files for each subject."""
    org_url = _find_subfolder(root_items, "Organized data and model predictions")
    if not org_url:
        print(
            "  warn: 'Organized data and model predictions' folder not found",
            file=sys.stderr,
        )
        return
    org_items = _list_folder(org_url)
    for sub_n in subjects:
        sub_name = f"sub{sub_n}"
        sub_url = _find_subfolder(org_items, sub_name)
        if not sub_url:
            print(f"  warn: folder '{sub_name}' not found in OSF", file=sys.stderr)
            continue
        sub_items = _list_folder(sub_url)
        fits_folder = "analyzed data files with class objects"
        fits_url = _find_subfolder(sub_items, fits_folder)
        if not fits_url:
            print(
                f"  warn: '{fits_folder}' subfolder not found for {sub_name}",
                file=sys.stderr,
            )
            continue
        dest_dir = (
            data_dir / "Organized data and model predictions" / sub_name / fits_folder
        )
        n = 0
        for item in _list_folder(fits_url):
            if item.get("attributes", {}).get("kind") == "file" and item["attributes"][
                "name"
            ].endswith(".pkl"):
                dl_url = item["links"].get("download", "")
                if dl_url:
                    _download_file(dl_url, dest_dir / item["attributes"]["name"])
                    n += 1
        print(f"  {sub_name}: {n} pkl file(s) processed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=DEFAULT_SUBJECTS,
        help="Subject numbers to download (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Destination directory (default: %(default)s)",
    )
    parser.add_argument(
        "--fits",
        action="store_true",
        default=False,
        help="Also download Wishart process fit pkl files (main + bootstrap)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching OSF node '{OSF_NODE}' → {data_dir}")
    root_url = f"{OSF_API}/nodes/{OSF_NODE}/files/osfstorage/"
    root_items = _list_folder(root_url)

    print("\n--- Transformation matrices ---")
    download_transformation_matrices(root_items, data_dir)

    print("\n--- Threshold ellipses ---")
    download_thres_ellipses(root_items, data_dir, args.subjects)

    if args.fits:
        print("\n--- Wishart process fit pkl files ---")
        download_fit_pkls(root_items, data_dir, args.subjects)

    print("\nDone.")


if __name__ == "__main__":
    main()
