"""
Fetches the Kaggle "Credit Card Fraud Detection" dataset (mlg-ulb/creditcardfraud)
into data/raw/creditcard.csv.

Requires Kaggle API credentials (either ~/.kaggle/kaggle.json or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables). If credentials are not
configured, or the download fails for any reason, this script prints
instructions and exits non-zero rather than silently substituting fake data --
use scripts/generate_synthetic_data.py explicitly if you want a runnable
demo without Kaggle access.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
TARGET_CSV = RAW_DIR / "creditcard.csv"
DATASET = "mlg-ulb/creditcardfraud"


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_CSV.exists():
        print(f"Already present: {TARGET_CSV}")
        return 0

    if not have_credentials():
        print(
            "Kaggle credentials not found.\n"
            "Set them up with one of:\n"
            "  1) Place your Kaggle API token at ~/.kaggle/kaggle.json\n"
            "     (Kaggle account -> Settings -> API -> Create New Token)\n"
            "  2) export KAGGLE_USERNAME=... KAGGLE_KEY=...\n\n"
            "No credentials available in this environment, so falling back is\n"
            "NOT automatic. To run the full pipeline right now without Kaggle\n"
            "access, generate a schema-compatible synthetic dataset instead:\n"
            "  python scripts/generate_synthetic_data.py\n"
        )
        return 1

    try:
        import kaggle  # noqa: WPS433 (import inside function is intentional)
    except ImportError:
        print("The 'kaggle' package is required. Install with: pip install kaggle")
        return 1

    print(f"Downloading {DATASET} via Kaggle API ...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=str(RAW_DIR), quiet=False)

    zip_path = RAW_DIR / "creditcardfraud.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    if not TARGET_CSV.exists():
        print("Download finished but creditcard.csv was not found in the archive.")
        return 1

    print(f"Saved dataset to {TARGET_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
