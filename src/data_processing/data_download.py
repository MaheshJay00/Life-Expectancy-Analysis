from data_loader import download_world_bank_indicators


INDICATORS = {
    "SP.DYN.LE00.IN": "life_expectancy_total",
    "SP.DYN.LE00.MA.IN": "life_expectancy_male",
    "SP.DYN.LE00.FE.IN": "life_expectancy_female",
    "SP.DYN.CDRT.IN": "death_rate",
    "SP.DYN.TFRT.IN": "fertility_rate",
}


def main() -> None:
    results = download_world_bank_indicators(INDICATORS)

    for indicator_code, files in results.items():
        print(f"\n{indicator_code}:")

        for file_type, path in files.items():
            print(f"  {file_type}: {path}")


if __name__ == "__main__":
    main()