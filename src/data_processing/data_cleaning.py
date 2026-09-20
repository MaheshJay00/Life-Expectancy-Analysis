from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed_data"

WORLD_BANK_HEADER_ROWS = 4

REQUIRED_DATA_COLUMNS = {
    "Country Name",
    "Country Code",
    "Indicator Name",
    "Indicator Code",
}

METADATA_COLUMN_RENAME = {
    "Country Code": "country_code",
    "Region": "region",
    "IncomeGroup": "income_group",
    "SpecialNotes": "special_notes",
    "TableName": "table_name",
}

FINAL_COLUMNS = [
    "country_name",
    "country_code",
    "region",
    "income_group",
    "special_notes",
    "table_name",
    "indicator_name",
    "indicator_code",
    "year",
    "value",
]


def clean_world_bank_dataset(
    data_path: str | Path,
    metadata_path: str | Path,
) -> pd.DataFrame:
    """
    Read one World Bank API CSV and its matching country metadata CSV.

    The API data is converted from wide year columns into long format
    and country metadata is merged using country_code.

    Missing indicator values are retained as NaN.
    """
    data_path = Path(data_path)
    metadata_path = Path(metadata_path)

    # ---------------------------------------------------------
    # Load main World Bank dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        data_path,
        skiprows=WORLD_BANK_HEADER_ROWS,
    )

    missing_columns = REQUIRED_DATA_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {data_path}: "
            f"{sorted(missing_columns)}"
        )

    df = df.rename(
        columns={
            "Country Name": "country_name",
            "Country Code": "country_code",
            "Indicator Name": "indicator_name",
            "Indicator Code": "indicator_code",
        }
    )

    # ---------------------------------------------------------
    # Identify year columns
    # ---------------------------------------------------------

    year_cols = [
        column
        for column in df.columns
        if str(column).isdigit()
    ]

    if not year_cols:
        raise ValueError(
            f"No year columns found in {data_path}"
        )

    # ---------------------------------------------------------
    # Wide -> long
    # ---------------------------------------------------------

    df_long = df.melt(
        id_vars=[
            "country_name",
            "country_code",
            "indicator_name",
            "indicator_code",
        ],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    df_long["year"] = pd.to_numeric(
        df_long["year"],
        errors="raise",
    ).astype(int)

    df_long["value"] = pd.to_numeric(
        df_long["value"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Load country metadata
    # ---------------------------------------------------------

    meta = pd.read_csv(metadata_path)

    if "Country Code" not in meta.columns:
        raise ValueError(
            f"Missing 'Country Code' in metadata file: "
            f"{metadata_path}"
        )

    meta = meta.rename(columns=METADATA_COLUMN_RENAME)

    metadata_columns = [
        "country_code",
        "region",
        "income_group",
        "special_notes",
        "table_name",
    ]

    # Add missing optional metadata columns as NA.
    for column in metadata_columns:
        if column not in meta.columns:
            meta[column] = pd.NA

    meta = meta[metadata_columns]

    # ---------------------------------------------------------
    # Validate metadata
    # ---------------------------------------------------------

    if meta["country_code"].duplicated().any():
        duplicates = (
            meta.loc[
                meta["country_code"].duplicated(keep=False),
                "country_code",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate country codes found in metadata: {duplicates}"
        )

    # ---------------------------------------------------------
    # Merge indicator data with country metadata
    # ---------------------------------------------------------

    df_clean = df_long.merge(
        meta,
        on="country_code",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Final column ordering
    # ---------------------------------------------------------

    df_clean = df_clean[FINAL_COLUMNS]

    return df_clean


def process_world_bank_folder(
    folder: str | Path,
    output_dir: str | Path = PROCESSED_DATA_DIR,
) -> Path:
    """
    Process one World Bank indicator folder.

    The folder must contain exactly one API CSV and exactly one
    country metadata CSV.
    """
    folder = Path(folder)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    api_files = sorted(folder.glob("API_*.csv"))
    metadata_files = sorted(
        folder.glob("Metadata_Country_*.csv")
    )

    if len(api_files) != 1:
        raise ValueError(
            f"Expected exactly one API CSV in {folder}, "
            f"found {len(api_files)}"
        )

    if len(metadata_files) != 1:
        raise ValueError(
            f"Expected exactly one metadata CSV in {folder}, "
            f"found {len(metadata_files)}"
        )

    cleaned = clean_world_bank_dataset(
        data_path=api_files[0],
        metadata_path=metadata_files[0],
    )

    output_path = (
        output_dir / f"processed_{folder.name}.csv"
    )

    cleaned.to_csv(
        output_path,
        index=False,
    )

    return output_path


def process_all_world_bank_data(
    raw_data_dir: str | Path = RAW_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
) -> list[Path]:
    """
    Process every World Bank indicator folder under raw_data.

    Each valid indicator folder produces one cleaned CSV
    in processed_data.
    """
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {raw_data_dir}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files: list[Path] = []

    folders = sorted(
        path
        for path in raw_data_dir.iterdir()
        if path.is_dir()
    )

    if not folders:
        raise ValueError(
            f"No indicator folders found in {raw_data_dir}"
        )

    for folder in folders:
        api_files = sorted(folder.glob("API_*.csv"))
        metadata_files = sorted(
            folder.glob("Metadata_Country_*.csv")
        )

        # Ignore folders that aren't World Bank indicator folders.
        if not api_files and not metadata_files:
            continue

        output_path = process_world_bank_folder(
            folder=folder,
            output_dir=output_dir,
        )

        saved_files.append(output_path)

    return saved_files


if __name__ == "__main__":
    files = process_all_world_bank_data()

    for path in files:
        print(f"Processed: {path}")