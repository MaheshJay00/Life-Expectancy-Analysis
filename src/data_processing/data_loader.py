from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"


def download_world_bank_indicator(
    indicator_code: str,
    folder_name: str,
    save_dir: str | Path = RAW_DATA_DIR,
) -> dict[str, Path]:
    """
    Download a World Bank indicator ZIP file and extract all contents
    into a dedicated folder under raw_data.

    Returns a dictionary containing the paths of extracted CSV files,
    categorized as data, metadata, or other.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_dir = save_dir / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    url = (
        f"https://api.worldbank.org/v2/en/indicator/"
        f"{indicator_code}?downloadformat=csv"
    )

    try:
        with urlopen(url, timeout=60) as response:
            zip_bytes = response.read()

    except HTTPError as exc:
        raise RuntimeError(
            f"World Bank download failed for {indicator_code}: "
            f"HTTP {exc.code}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Could not reach World Bank API for {indicator_code}"
        ) from exc

    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        raise ValueError(
            f"Downloaded content for {indicator_code} is not a valid ZIP file"
        )

    extracted_files: dict[str, Path] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Extract EVERYTHING from the ZIP.
        zf.extractall(target_dir)

        # Record extracted CSV files for downstream processing.
        for member in zf.infolist():
            if member.is_dir():
                continue

            extracted_path = target_dir / member.filename
            filename_lower = Path(member.filename).name.lower()

            if not filename_lower.endswith(".csv"):
                continue

            if "metadata_country" in filename_lower:
                file_key = "metadata"
            elif filename_lower.startswith("api_"):
                file_key = "data"
            else:
                file_key = "other_csv"

            # If there is already a file under the same key,
            # don't silently overwrite the reference.
            if file_key in extracted_files:
                raise ValueError(
                    f"Multiple files classified as '{file_key}' "
                    f"for indicator {indicator_code}"
                )

            extracted_files[file_key] = extracted_path

    if "data" not in extracted_files:
        raise ValueError(
            f"No main API data CSV found for indicator {indicator_code}"
        )

    return extracted_files


def download_world_bank_indicators(
    indicator_map: dict[str, str],
    save_dir: str | Path = RAW_DATA_DIR,
) -> dict[str, dict[str, Path]]:
    """
    Download multiple World Bank indicators.

    indicator_map maps:
        World Bank indicator code -> folder name
    """
    results: dict[str, dict[str, Path]] = {}

    for indicator_code, folder_name in indicator_map.items():
        results[indicator_code] = download_world_bank_indicator(
            indicator_code=indicator_code,
            folder_name=folder_name,
            save_dir=save_dir,
        )

    return results