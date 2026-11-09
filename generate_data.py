# -*- coding: utf-8 -*-
"""
Генерация данных варианта 1 для ЛР №1 по методам оптимизации.
Задача: план производства продукции на станках, максимум прибыли.
Минимальный масштаб варианта: 5 продуктов, 4 станка.

Seed: 10000 + 100*Q + N = 10131 (Q=1, N=31).
"""

from pathlib import Path
import csv
import json
import numpy as np

SEED = 10131
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

PRODUCTS = ["Продукт A", "Продукт B", "Продукт C", "Продукт D", "Продукт E"]
BASE_PROFIT = np.array([120.0, 90.0, 150.0, 70.0, 110.0])
BASE_DEMAND = np.array([40.0, 60.0, 30.0, 80.0, 50.0])
BASE_TIMES = np.array([
    [2.0, 1.0, 3.0, 0.5, 1.5],
    [1.0, 2.0, 1.0, 1.5, 0.5],
    [3.0, 1.0, 2.0, 1.0, 2.5],
    [1.5, 2.5, 1.5, 2.0, 1.0],
], dtype=float)
MACHINES = ["Станок M1", "Станок M2", "Станок M3", "Станок M4"]
BASE_CAPACITY = np.array([260.0, 240.0, 300.0, 280.0])


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    profit = np.round(BASE_PROFIT * rng.normal(1.0, 0.03, len(PRODUCTS)), 2)
    times = np.round(BASE_TIMES * rng.normal(1.0, 0.05, BASE_TIMES.shape), 2)

    with open(DATA / "products.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "product", "profit_per_unit", "max_demand",
                    "time_m1", "time_m2", "time_m3", "time_m4"])
        for j, name in enumerate(PRODUCTS):
            w.writerow([j + 1, name, float(profit[j]), float(BASE_DEMAND[j]),
                        *[float(times[m, j]) for m in range(4)]])

    with open(DATA / "machines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["machine_id", "machine", "capacity_hours"])
        for m, name in enumerate(MACHINES):
            w.writerow([m + 1, name, float(BASE_CAPACITY[m])])

    with open(DATA / "params.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "units": {
                "x": "единиц продукции",
                "profit": "тыс. руб. на единицу",
                "time": "часов на единицу",
                "capacity": "часов на период",
                "demand": "единиц",
            },
            "note": "Данные синтетические: экспертные значения с небольшим шумом.",
        }, f, ensure_ascii=False, indent=2)

    print(f"Данные сохранены в {DATA}")
    print("Прибыль:", profit)
    print("Трудоёмкости:\n", times)


if __name__ == "__main__":
    main()
