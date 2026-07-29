"""
Script to generate a synthetic housing dataset for training.

This creates a sample housing.csv that matches the expected schema:
id, square_footage, bedrooms, bathrooms, year_built, lot_size,
distance_to_city_center, school_rating, price
"""

import csv
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

output_path = DATA_DIR / "housing.csv"

n_samples = 100

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "square_footage", "bedrooms", "bathrooms",
        "year_built", "lot_size", "distance_to_city_center",
        "school_rating", "price"
    ])

    for i in range(1, n_samples + 1):
        square_footage = random.randint(800, 4000)
        bedrooms = random.randint(1, 6)
        bathrooms = round(random.uniform(1, 4), 1)
        year_built = random.randint(1920, 2024)
        lot_size = random.randint(2000, 15000)
        distance_to_city = round(random.uniform(0.5, 15.0), 1)
        school_rating = round(random.uniform(1, 10), 1)

        base_price = (
            square_footage * 150
            + bedrooms * 5000
            + bathrooms * 8000
            + (year_built - 1920) * 200
            + lot_size * 0.5
            - distance_to_city * 3000
            + school_rating * 8000
        )
        noise = random.gauss(0, 20000)
        price = max(50000, round(base_price + noise, 0))

        writer.writerow([
            i, square_footage, bedrooms, bathrooms,
            year_built, lot_size, distance_to_city,
            school_rating, int(price)
        ])

print(f"Generated {n_samples} samples to {output_path}")
