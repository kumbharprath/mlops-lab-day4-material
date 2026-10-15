"""Generate reproducible synthetic food delivery examples."""
from pathlib import Path

import numpy as np
import pandas as pd


def create_dataset(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(1, 15, n)
    raining = rng.integers(0, 2, n)
    heavy_traffic = rng.integers(0, 2, n)
    score = -3.0 + 0.2 * distance + 0.9 * raining + 1.2 * heavy_traffic
    probability = 1 / (1 + np.exp(-score))
    late_delivery = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "distance_km": distance.round(2),
            "raining": raining,
            "heavy_traffic": heavy_traffic,
            "late_delivery": late_delivery,
        }
    )


def main():
    output = Path(__file__).resolve().parent.parent / "data" / "deliveries.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    create_dataset().to_csv(output, index=False)
    print(f"Dataset created: {output}")


if __name__ == "__main__":
    main()
