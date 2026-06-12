#!/usr/bin/env python3
"""Download the subset of the Hong et al. (2025) OSF dataset needed by these notebooks.

Downloads into the directory specified by data_dir in local_config.json,
or <repo_root>/data/hong_etal_2025/ if no config is found (git-ignored),
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
    python analysis/upenn/hong_etal_2025/download_data.py \
        [--subjects 1 2 4 ...] [--fits]

No third-party packages required — stdlib only (certifi used if available).
"""

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OSF_API = "https://api.osf.io/v2"
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent

# Known system CA bundle locations (covers macOS, Debian/Ubuntu, RHEL, OpenSUSE)
_CA_PATHS = [
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/usr/local/etc/openssl/cert.pem",
]


def _make_ssl_context() -> ssl.SSLContext:
    """Return an SSL context that works on macOS Python.org installs and Linux.

    Python.org Python on macOS ships with its own OpenSSL that does not use the
    system keychain, so the default context fails with CERTIFICATE_VERIFY_FAILED.
    We work around this by preferring certifi's CA bundle (if installed) and
    falling back to known system CA file locations before accepting the default.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for ca_file in _CA_PATHS:
        if Path(ca_file).exists():
            try:
                return ssl.create_default_context(cafile=ca_file)
            except ssl.SSLError:
                continue
    return ssl.create_default_context()


_SSL_CTX = _make_ssl_context()


def _load_config() -> dict[str, Any]:
    cfg_path = _SCRIPT_DIR / "local_config.json"
    if not cfg_path.exists():
        sys.exit(
            f"No local_config.json found at {cfg_path}.\n"
            "Copy local_config.json.template → local_config.json and fill in paths."
        )
    with open(cfg_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _resolve(path_str: str, label: str) -> Path:
    """Resolve an absolute or repo-relative path; fail clearly if empty."""
    if not path_str:
        sys.exit(
            f"{label} must be set in local_config.json (see template). "
            "Relative paths are resolved from the repo root."
        )
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _osf_node(cfg: dict[str, Any]) -> str:
    """Return the OSF node ID from data_repo in config, falling back to the default."""
    repo = str(cfg.get("data_repo", ""))
    if repo:
        # Accept full URL (https://osf.io/k27js) or bare node ID (k27js)
        return repo.rstrip("/").split("/")[-1]
    return "k27js"


DEFAULT_SUBJECTS = [1]  # sub1 = subject CH


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code} for {url}", file=sys.stderr)
        raise


def _download_file(download_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip  {dest.relative_to(_REPO_ROOT)}  (already exists)")
        return
    print(f"  fetch {dest.relative_to(_REPO_ROOT)}")
    req = urllib.request.Request(download_url)
    with (
        urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp,
        open(dest, "wb") as fh,
    ):
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
        default=None,
        help="Override data_dir from local_config.json (for Colab or scripted use)",
    )
    parser.add_argument(
        "--fits",
        action="store_true",
        default=False,
        help="Also download Wishart process fit pkl files (main + bootstrap)",
    )
    args = parser.parse_args()

    cfg = _load_config()
    data_dir: Path = (
        args.data_dir
        if args.data_dir is not None
        else _resolve(cfg.get("data_dir", ""), "data_dir")
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    osf_node = _osf_node(cfg)
    print(f"Fetching OSF node '{osf_node}' → {data_dir}")
    root_url = f"{OSF_API}/nodes/{osf_node}/files/osfstorage/"
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
