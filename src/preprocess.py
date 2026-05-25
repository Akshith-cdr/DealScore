"""
Preprocess the raw Indian used-car dataset for DealScore.

Expected raw file:
    data/raw/used_cars_india.csv

Output:
    data/processed/cars_cleaned.csv
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "used_cars_india.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "cars_cleaned.csv"

CURRENT_YEAR = 2026

COLUMN_ALIASES = {
    "brand": ["brand", "make", "company"],
    "model": ["model", "car_model", "name", "car_name"],
    "year": ["year", "manufacturing_year", "registration_year", "year_of_manufacture"],
    "fuel_type": ["fuel_type", "fuel"],
    "transmission": ["transmission"],
    "owner_type": ["owner_type", "owner", "owner_number"],
    "km_driven": ["km_driven", "kms_driven", "driven_kms", "kilometers_driven", "kilometres_driven"],
    "seller_type": ["seller_type", "selling_type", "seller"],
    "mileage": ["mileage", "mileage_kmpl"],
    "engine": ["engine", "engine_cc"],
    "seats": ["seats"],
    "price": ["price", "selling_price", "listed_price", "present_price", "car_price_in_rupees"],
}

REQUIRED_COLUMNS = [
    "brand",
    "model",
    "year",
    "fuel_type",
    "transmission",
    "owner_type",
    "km_driven",
    "seller_type",
    "mileage",
    "engine",
    "seats",
    "price",
]

DEFAULT_NUMERIC_VALUES = {
    "mileage": 18.0,
    "engine": 1200.0,
    "seats": 5.0,
}

CAR_MODEL_SPECS = {
    "800": {"mileage": 16.1, "engine": 796, "seats": 4},
    "alto 800": {"mileage": 22.7, "engine": 796, "seats": 5},
    "alto k10": {"mileage": 24.1, "engine": 998, "seats": 5},
    "amaze": {"mileage": 18.6, "engine": 1199, "seats": 5},
    "baleno": {"mileage": 21.4, "engine": 1197, "seats": 5},
    "brio": {"mileage": 18.5, "engine": 1198, "seats": 5},
    "camry": {"mileage": 12.9, "engine": 2494, "seats": 5},
    "ciaz": {"mileage": 20.7, "engine": 1373, "seats": 5},
    "city": {"mileage": 17.4, "engine": 1497, "seats": 5},
    "corolla": {"mileage": 14.5, "engine": 1798, "seats": 5},
    "corolla altis": {"mileage": 14.3, "engine": 1798, "seats": 5},
    "creta": {"mileage": 16.8, "engine": 1497, "seats": 5},
    "dzire": {"mileage": 22.0, "engine": 1197, "seats": 5},
    "elantra": {"mileage": 14.6, "engine": 1999, "seats": 5},
    "eon": {"mileage": 21.1, "engine": 814, "seats": 5},
    "ertiga": {"mileage": 19.0, "engine": 1462, "seats": 7},
    "etios cross": {"mileage": 18.2, "engine": 1197, "seats": 5},
    "etios g": {"mileage": 16.8, "engine": 1197, "seats": 5},
    "etios gd": {"mileage": 23.6, "engine": 1364, "seats": 5},
    "etios liva": {"mileage": 18.2, "engine": 1197, "seats": 5},
    "fortuner": {"mileage": 12.6, "engine": 2755, "seats": 7},
    "grand i10": {"mileage": 18.9, "engine": 1197, "seats": 5},
    "i10": {"mileage": 19.8, "engine": 1086, "seats": 5},
    "i20": {"mileage": 18.6, "engine": 1197, "seats": 5},
    "ignis": {"mileage": 20.9, "engine": 1197, "seats": 5},
    "innova": {"mileage": 12.99, "engine": 2494, "seats": 7},
    "jazz": {"mileage": 18.7, "engine": 1199, "seats": 5},
    "land cruiser": {"mileage": 11.0, "engine": 4461, "seats": 7},
    "omni": {"mileage": 16.8, "engine": 796, "seats": 8},
    "ritz": {"mileage": 18.5, "engine": 1197, "seats": 5},
    "s cross": {"mileage": 18.5, "engine": 1248, "seats": 5},
    "swift": {"mileage": 21.2, "engine": 1197, "seats": 5},
    "sx4": {"mileage": 16.5, "engine": 1586, "seats": 5},
    "verna": {"mileage": 17.7, "engine": 1497, "seats": 5},
    "vitara brezza": {"mileage": 17.0, "engine": 1462, "seats": 5},
    "wagon r": {"mileage": 21.5, "engine": 998, "seats": 5},
    "xcent": {"mileage": 19.1, "engine": 1197, "seats": 5},
    "3 series": {"mileage": 16.1, "engine": 1995, "seats": 5},
    "5 series": {"mileage": 15.6, "engine": 1995, "seats": 5},
    "718": {"mileage": 9.0, "engine": 1988, "seats": 2},
    "a4": {"mileage": 17.4, "engine": 1984, "seats": 5},
    "compass": {"mileage": 14.9, "engine": 1956, "seats": 5},
    "ecosport": {"mileage": 15.9, "engine": 1497, "seats": 5},
    "eeco": {"mileage": 16.1, "engine": 1196, "seats": 5},
    "eqc": {"mileage": 0.0, "engine": 0, "seats": 5},
    "kwid": {"mileage": 22.0, "engine": 999, "seats": 5},
    "optra": {"mileage": 12.7, "engine": 1991, "seats": 5},
    "safari": {"mileage": 14.1, "engine": 1956, "seats": 7},
    "seltos": {"mileage": 16.5, "engine": 1497, "seats": 5},
}

KNOWN_BRANDS = [
    "Mercedes-Benz",
    "Maruti Suzuki",
    "Land Rover",
    "Chevrolet",
    "Volkswagen",
    "Mahindra",
    "Hyundai",
    "Porsche",
    "Renault",
    "Toyota",
    "Datsun",
    "Jaguar",
    "Nissan",
    "Skoda",
    "Volvo",
    "Bentley",
    "Citroen",
    "Isuzu",
    "Audi",
    "BMW",
    "Fiat",
    "Ford",
    "Honda",
    "Jeep",
    "Kia",
    "Mini",
    "Tata",
    "MG",
]

BRAND_DISPLAY_NAMES = {
    "maruti suzuki": "Maruti",
    "mercedes-benz": "Mercedes-Benz",
    "bmw": "BMW",
    "mg": "MG",
}

MODEL_KEYWORDS = [
    "Corolla Altis",
    "Vitara Brezza",
    "Land Cruiser",
    "Grand i10",
    "Elite i20",
    "Alto 800",
    "Alto K10",
    "Etios Liva",
    "S-Presso",
    "XUV500",
    "TUV300",
    "CR-V",
    "7 Series",
    "E Class",
    "C Class",
    "S Class",
    "3 Series",
    "5 Series",
    "EcoSport",
    "Wagon R",
    "S Cross",
    "Compass",
    "Fortuner",
    "Safari",
    "Seltos",
    "Baleno",
    "Creta",
    "Ertiga",
    "Innova",
    "Swift",
    "Verna",
    "Amaze",
    "Ciaz",
    "City",
    "Dzire",
    "Jazz",
    "Kwid",
    "Xcent",
    "Eeco",
    "Optra",
    "i20",
    "i10",
    "Eon",
    "EQC",
    "718",
]

BRAND_DEFAULT_SPECS = {
    "Audi": {"mileage": 14.5, "engine": 1984, "seats": 5},
    "BMW": {"mileage": 15.0, "engine": 1995, "seats": 5},
    "Chevrolet": {"mileage": 15.0, "engine": 1199, "seats": 5},
    "Datsun": {"mileage": 20.6, "engine": 999, "seats": 5},
    "Fiat": {"mileage": 16.3, "engine": 1248, "seats": 5},
    "Ford": {"mileage": 16.0, "engine": 1497, "seats": 5},
    "Honda": {"mileage": 17.8, "engine": 1497, "seats": 5},
    "Hyundai": {"mileage": 18.3, "engine": 1197, "seats": 5},
    "Isuzu": {"mileage": 13.8, "engine": 1898, "seats": 5},
    "Jaguar": {"mileage": 13.0, "engine": 1999, "seats": 5},
    "Jeep": {"mileage": 14.9, "engine": 1956, "seats": 5},
    "Kia": {"mileage": 16.5, "engine": 1497, "seats": 5},
    "Land Rover": {"mileage": 11.8, "engine": 2179, "seats": 5},
    "Mahindra": {"mileage": 15.1, "engine": 2179, "seats": 7},
    "Maruti": {"mileage": 20.8, "engine": 1197, "seats": 5},
    "Mercedes-Benz": {"mileage": 13.8, "engine": 1991, "seats": 5},
    "MG": {"mileage": 14.2, "engine": 1451, "seats": 5},
    "Mini": {"mileage": 16.3, "engine": 1998, "seats": 4},
    "Nissan": {"mileage": 18.0, "engine": 1498, "seats": 5},
    "Porsche": {"mileage": 9.0, "engine": 1988, "seats": 2},
    "Renault": {"mileage": 20.0, "engine": 999, "seats": 5},
    "Skoda": {"mileage": 15.6, "engine": 1498, "seats": 5},
    "Tata": {"mileage": 17.0, "engine": 1199, "seats": 5},
    "Toyota": {"mileage": 14.5, "engine": 1496, "seats": 5},
    "Volkswagen": {"mileage": 17.2, "engine": 1197, "seats": 5},
    "Volvo": {"mileage": 12.4, "engine": 1969, "seats": 5},
    "Bentley": {"mileage": 8.6, "engine": 3993, "seats": 4},
    "Citroen": {"mileage": 18.6, "engine": 1199, "seats": 5},
}

CAR_MODEL_TO_BRAND = {
    "800": "Maruti",
    "alto 800": "Maruti",
    "alto k10": "Maruti",
    "baleno": "Maruti",
    "ciaz": "Maruti",
    "dzire": "Maruti",
    "ertiga": "Maruti",
    "ignis": "Maruti",
    "omni": "Maruti",
    "ritz": "Maruti",
    "s cross": "Maruti",
    "swift": "Maruti",
    "sx4": "Maruti",
    "vitara brezza": "Maruti",
    "wagon r": "Maruti",
    "camry": "Toyota",
    "corolla": "Toyota",
    "corolla altis": "Toyota",
    "etios cross": "Toyota",
    "etios g": "Toyota",
    "etios gd": "Toyota",
    "etios liva": "Toyota",
    "fortuner": "Toyota",
    "innova": "Toyota",
    "land cruiser": "Toyota",
    "amaze": "Honda",
    "brio": "Honda",
    "city": "Honda",
    "jazz": "Honda",
    "creta": "Hyundai",
    "elantra": "Hyundai",
    "eon": "Hyundai",
    "grand i10": "Hyundai",
    "i10": "Hyundai",
    "i20": "Hyundai",
    "verna": "Hyundai",
    "xcent": "Hyundai",
}

TWO_WHEELER_BRANDS = (
    "bajaj",
    "hero",
    "hero honda",
    "hyosung",
    "ktm",
    "royal enfield",
    "tvs",
    "um",
    "yamaha",
)

TWO_WHEELER_MODEL_KEYWORDS = (
    "access",
    "activa",
    "apache",
    "avenger",
    "bullet",
    "cb hornet",
    "cb shine",
    "cb trigger",
    "cb twister",
    "cbr",
    "cbz",
    "classic",
    "ct 100",
    "discover",
    "dominar",
    "dream yuga",
    "duke",
    "extreme",
    "fazer",
    "fz",
    "glamour",
    "gt250r",
    "hunk",
    "ignitor",
    "jupyter",
    "karizma",
    "mojo",
    "passion",
    "pulsar",
    "rc200",
    "rc390",
    "renegade",
    "splender",
    "sport",
    "super splendor",
    "thunder",
    "unicorn",
    "wego",
    "xtreme",
)


def normalize_column_name(column: str) -> str:
    """Convert a column name into snake_case for easier matching."""
    column = column.strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common Kaggle column variants to the DealScore schema."""
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]

    rename_map: dict[str, str] = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias = normalize_column_name(alias)
            if alias in df.columns:
                rename_map[alias] = standard_name
                break

    df = df.rename(columns=rename_map)
    return df


def extract_first_number(value) -> float:
    """Extract the first numeric value from strings like '18.2 kmpl' or '1197 CC'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"(\d+\.?\d*)", str(value).replace(",", ""))
    return float(match.group(1)) if match else np.nan


def parse_price(value) -> float:
    """Parse Indian listing prices like 'Rs. 4.45 Lakh' or 'Rs. 1.2 Crore'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).replace(",", "").lower()
    number = extract_first_number(text)
    if pd.isna(number):
        return np.nan
    if "crore" in text or " cr" in text:
        return number * 10_000_000
    if "lakh" in text or " lac" in text:
        return number * 100_000
    if "thousand" in text:
        return number * 1_000
    return number


def normalize_vehicle_name(value) -> str:
    """Normalize vehicle names for matching against known car and bike names."""
    value = str(value).strip().lower()
    value = re.sub(r"\[[^\]]+\]", "", value)
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_brand_and_model(vehicle_name: str) -> tuple[str, str]:
    """Split full listing names into a readable brand and compact model."""
    name = str(vehicle_name).strip()
    normalized_name = normalize_vehicle_name(name)

    brand = normalized_name.split()[0].title() if normalized_name else "Unknown"
    remaining = normalized_name
    for candidate in sorted(KNOWN_BRANDS, key=len, reverse=True):
        normalized_brand = normalize_vehicle_name(candidate)
        if normalized_name.startswith(normalized_brand):
            brand = BRAND_DISPLAY_NAMES.get(normalized_brand, candidate)
            remaining = normalized_name[len(normalized_brand):].strip()
            break

    model = remaining.split()[0].title() if remaining else name.title()
    for keyword in sorted(MODEL_KEYWORDS, key=len, reverse=True):
        normalized_keyword = normalize_vehicle_name(keyword)
        if remaining.startswith(normalized_keyword):
            model = keyword
            break

    return brand, model


def infer_brand_and_model(df: pd.DataFrame) -> pd.DataFrame:
    """Split a combined car name into brand and model when separate columns are missing."""
    df = df.copy()
    if "model" in df.columns and "brand" not in df.columns:
        full_listing_names = df["model"].astype(str)
        parsed_values = full_listing_names.apply(parse_brand_and_model)
        df["brand"] = parsed_values.apply(lambda value: value[0])
        df["model"] = parsed_values.apply(lambda value: value[1])
        if "transmission" not in df.columns:
            df["transmission"] = full_listing_names.apply(infer_transmission)
    elif "brand" in df.columns and "model" not in df.columns:
        df["model"] = df["brand"].astype(str)

    if "brand" in df.columns and "model" in df.columns:
        df["brand"] = df["brand"].astype(str).str.strip().str.title()
        df["model"] = df["model"].astype(str).str.strip().str.title()
    return df


def remove_two_wheelers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bike and scooter rows from mixed CarDekho-style datasets."""
    df = df.copy()
    normalized_model = df["model"].apply(normalize_vehicle_name)

    starts_with_bike_brand = normalized_model.apply(
        lambda name: any(name.startswith(brand) for brand in TWO_WHEELER_BRANDS)
    )
    contains_bike_keyword = normalized_model.apply(
        lambda name: any(keyword in name for keyword in TWO_WHEELER_MODEL_KEYWORDS)
    )

    car_mask = ~(starts_with_bike_brand | contains_bike_keyword)
    removed_count = int((~car_mask).sum())
    if removed_count:
        print(f"Removed two-wheeler rows: {removed_count}")
    return df.loc[car_mask].copy()


def correct_car_brands(df: pd.DataFrame) -> pd.DataFrame:
    """Map dataset-specific car model names to their real car brands."""
    df = df.copy()
    normalized_model = df["model"].apply(normalize_vehicle_name)
    mapped_brands = normalized_model.map(CAR_MODEL_TO_BRAND)
    df.loc[mapped_brands.notna(), "brand"] = mapped_brands[mapped_brands.notna()]
    return df


def enrich_missing_car_specs(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing generic specs with practical model-level car values."""
    df = df.copy()
    normalized_model = df["model"].apply(normalize_vehicle_name)

    for row_index, model_name in normalized_model.items():
        brand = str(df.at[row_index, "brand"])
        fuel_type = normalize_vehicle_name(df.at[row_index, "fuel_type"])
        specs = CAR_MODEL_SPECS.get(model_name, BRAND_DEFAULT_SPECS.get(brand, DEFAULT_NUMERIC_VALUES)).copy()
        if "diesel" in fuel_type and specs["engine"] < 1400:
            specs["engine"] = 1498
            specs["mileage"] = max(specs["mileage"], 19.0)
        if "electric" in fuel_type:
            specs["engine"] = 0
            specs["mileage"] = 0.0
        for column, value in specs.items():
            if column not in df.columns or pd.isna(df.at[row_index, column]):
                df.at[row_index, column] = value
                continue
            if column in DEFAULT_NUMERIC_VALUES and df.at[row_index, column] == DEFAULT_NUMERIC_VALUES[column]:
                df.at[row_index, column] = value
    return df


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill and standardize text columns before model training."""
    df = df.copy()
    categorical_columns = ["brand", "model", "fuel_type", "transmission", "owner_type", "seller_type"]
    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
                .fillna("Unknown")
                .str.title()
            )
    if "brand" in df.columns:
        df["brand"] = df["brand"].replace({"Bmw": "BMW", "Mg": "MG"})
    if "model" in df.columns:
        df["model"] = df["model"].replace(
            {
                "Eqc": "EQC",
                "Cr-V": "CR-V",
                "Xuv500": "XUV500",
                "Tuv300": "TUV300",
            }
        )
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed text/numeric columns into usable numeric values."""
    df = df.copy()
    numeric_columns = ["year", "km_driven", "mileage", "engine", "seats", "price"]
    for column in numeric_columns:
        if column in df.columns:
            if column == "price":
                df[column] = df[column].apply(parse_price)
            else:
                df[column] = df[column].apply(extract_first_number)

    # Remove unrealistic rows that usually come from parsing errors or outliers.
    if "year" in df.columns:
        df = df[df["year"].between(1990, CURRENT_YEAR)]
    if "price" in df.columns:
        df = df[df["price"] > 0]
        if df["price"].median() < 1000:
            df["price"] = df["price"] * 100000
    if "km_driven" in df.columns:
        df = df[df["km_driven"].between(0, 500_000)]

    return df


def add_missing_supported_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create supported optional columns when a Kaggle file does not include them."""
    df = df.copy()
    for column, default_value in DEFAULT_NUMERIC_VALUES.items():
        if column not in df.columns:
            df[column] = default_value
    if "seller_type" not in df.columns:
        df["seller_type"] = "Unknown"
    if "owner_type" not in df.columns:
        df["owner_type"] = "Unknown"
    if "transmission" not in df.columns:
        df["transmission"] = "Unknown"
    return df


def infer_transmission(model_name: str) -> str:
    """Infer transmission from common automatic keywords in listing names."""
    name = normalize_vehicle_name(model_name)
    automatic_keywords = (" automatic ", " amt ", " at ", " cvt ", " dct ", " ivt ")
    padded_name = f" {name} "
    if any(keyword in padded_name for keyword in automatic_keywords):
        return "Automatic"
    return "Manual"


def infer_owner_type(year: float) -> str:
    """Estimate owner type from car age when the source dataset does not provide it."""
    car_age = CURRENT_YEAR - int(year)
    if car_age <= 4:
        return "First Owner"
    if car_age <= 8:
        return "Second Owner"
    if car_age <= 12:
        return "Third Owner"
    return "Fourth & Above Owner"


def infer_missing_market_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing market fields with valid, explainable values."""
    df = df.copy()

    if "transmission" in df.columns:
        missing_transmission = df["transmission"].isna() | (df["transmission"].astype(str).str.lower() == "unknown")
        df.loc[missing_transmission, "transmission"] = df.loc[missing_transmission, "model"].apply(infer_transmission)
        premium_auto_brands = {
            "Audi",
            "BMW",
            "Jaguar",
            "Land Rover",
            "Mercedes-Benz",
            "Mini",
            "Porsche",
            "Volvo",
        }
        electric_fuel = df["fuel_type"].astype(str).str.lower().str.contains("electric", na=False)
        premium_brand = df["brand"].isin(premium_auto_brands)
        df.loc[electric_fuel | premium_brand, "transmission"] = "Automatic"

    if "owner_type" in df.columns:
        missing_owner = df["owner_type"].isna() | (df["owner_type"].astype(str).str.lower() == "unknown")
        df.loc[missing_owner, "owner_type"] = df.loc[missing_owner, "year"].apply(infer_owner_type)

    if "seller_type" in df.columns:
        missing_seller = df["seller_type"].isna() | (df["seller_type"].astype(str).str.lower() == "unknown")
        df.loc[missing_seller, "seller_type"] = "Dealer"

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using simple beginner-friendly rules."""
    df = df.copy()

    numeric_columns = ["year", "km_driven", "mileage", "engine", "seats", "price"]
    for column in numeric_columns:
        if column in df.columns:
            median_value = df[column].median()
            if pd.isna(median_value):
                median_value = DEFAULT_NUMERIC_VALUES.get(column, 0)
            df[column] = df[column].fillna(median_value)

    categorical_columns = ["brand", "model", "fuel_type", "transmission", "owner_type", "seller_type"]
    for column in categorical_columns:
        if column in df.columns:
            mode_value = df[column].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful supervised-learning features from raw car attributes."""
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["engine"] = df["engine"].round().astype(int)
    df["seats"] = df["seats"].round().astype(int)
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)
    df["engine_per_seat"] = df["engine"] / df["seats"].replace(0, np.nan)
    df["engine_per_seat"] = df["engine_per_seat"].fillna(df["engine_per_seat"].median())
    return df


def round_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Round selected output columns for a cleaner processed CSV."""
    df = df.copy()
    for column in ["price", "mileage", "km_per_year", "engine_per_seat"]:
        if column in df.columns:
            df[column] = df[column].round(2)
    for column in ["year", "engine", "seats", "car_age"]:
        if column in df.columns:
            df[column] = df[column].astype(int)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if an important column is not available."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns after standardization: "
            f"{missing_columns}. Check your CSV column names in data/raw/used_cars_india.csv."
        )


def preprocess_data(raw_path: Path = RAW_DATA_PATH, output_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Run the full preprocessing pipeline and save the cleaned dataset."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Place your Kaggle CSV there and rerun this script."
        )

    df = pd.read_csv(raw_path)
    df = standardize_columns(df)
    df = infer_brand_and_model(df)
    df = add_missing_supported_columns(df)
    validate_columns(df)

    df = df[REQUIRED_COLUMNS].copy()
    df = df.drop_duplicates()
    df = remove_two_wheelers(df)
    df = correct_car_brands(df)
    df = enrich_missing_car_specs(df)
    df = infer_missing_market_fields(df)
    df = clean_categorical_columns(df)
    df = clean_numeric_columns(df)
    df = fill_missing_values(df)
    df = engineer_features(df)
    df = round_output_columns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    cleaned_df = preprocess_data()
    print(f"Cleaned dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {cleaned_df.shape[0]}, Columns: {cleaned_df.shape[1]}")
