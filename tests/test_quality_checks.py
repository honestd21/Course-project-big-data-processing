"""Мини-тесты правил R1-R5 из quality_checks.py.

Каждый тест берёт маленькую таблицу, где вручную известно, какие строки нарушают правило,
и сравнивает маску и метрики правила с этим ответом. Реальные данные не нужны.

Запуск: python3 test_quality_checks.py   (или pytest, если установлен)
"""
import numpy as np
import pandas as pd

import quality_checks as qc

COLS = ["Area", "Item", "Year", qc.TARGET, qc.RAIN, qc.PEST, qc.TEMP]


def table(rows):
    return pd.DataFrame(rows, columns=COLS)


def mask_of(m):
    return list(m.astype(bool))


# ---------------------------------------------------------------- R1
def test_r1_key_unique():
    df = table([
        ["A", "Wheat", 2000, 10, 500, 1, 10.0],   # ключ повторён, температура разная
        ["A", "Wheat", 2000, 10, 500, 1, 14.0],
        ["B", "Rice", 2001, 20, 600, 2, 15.0],    # полный дубль
        ["B", "Rice", 2001, 20, 600, 2, 15.0],
        ["C", "Rice", 2002, 30, 700, 3, 16.0],    # уникальная строка
    ])
    mask, m = qc.rule_key_unique(df)
    assert mask_of(mask) == [True, True, True, True, False]
    assert m["rows_sharing_key"] == 4
    assert m["extra_rows"] == 2
    assert m["unique_keys"] == 3
    assert m["keys_temp_differs"] == 1
    assert m["keys_yield_differs"] == 0
    assert m["median_temp_gap"] == 4.0
    assert m["full_duplicates"] == 1


def test_r1_same_key_different_yield():
    df = table([
        ["A", "Wheat", 2000, 10, 500, 1, 10.0],
        ["A", "Wheat", 2000, 99, 500, 1, 10.0],
    ])
    _, m = qc.rule_key_unique(df)
    assert m["keys_yield_differs"] == 1
    assert m["keys_temp_differs"] == 0
    assert m["full_duplicates"] == 0


# ---------------------------------------------------------------- R2
def test_r2_temp_consistent():
    df = table([
        ["A", "Wheat", 2000, 10, 500, 1, 10.0],   # (A, 2000): температура одна
        ["A", "Rice", 2000, 11, 500, 1, 10.0],
        ["A", "Wheat", 2001, 12, 500, 1, 11.0],   # (A, 2001): две температуры
        ["A", "Rice", 2001, 13, 500, 1, 12.0],
        ["B", "Wheat", 2000, 14, 400, 1, 5.0],    # (B, 2000): одна строка
    ])
    mask, m = qc.rule_temp_consistent(df)
    assert mask_of(mask) == [False, False, True, True, False]
    assert m["area_year_total"] == 3
    assert m["area_year_multi_temp"] == 1


# ---------------------------------------------------------------- R3
def test_r3_outliers_inside_item():
    wheat = [10, 11, 12, 13, 14, 100]             # 100 - выброс среди пшеницы
    rice = [100, 101, 102, 103, 104]              # то же 100 среди риса нормально
    rows = [["A", "Wheat", 2000 + i, y, 500, i + 1, 10.0] for i, y in enumerate(wheat)]
    rows += [["A", "Rice", 2000 + i, y, 500, i + 7, 10.0] for i, y in enumerate(rice)]
    mask, m = qc.rule_outliers_skew(table(rows))
    assert mask_of(mask) == [False] * 5 + [True] + [False] * 5
    assert m["outlier_rows"] == 1
    assert m["outlier_share_Wheat"] == round(1 / 6 * 100, 1)
    assert m["outlier_share_Rice"] == 0.0


def test_r3_skew_gives_warn_without_outliers():
    # урожайность одинакова (выбросов нет), пестициды сильно скошены
    pest = [1] * 9 + [100]
    df = table([["A", "Wheat", 2000 + i, 10, 500, p, 10.0] for i, p in enumerate(pest)])
    res = qc.Result("outliers_skew", "warning", *qc.rule_outliers_skew(df))
    assert res.n_rows == 0
    assert res.metrics["skewed_cols"] == 1
    assert res.status == "WARN"


# ---------------------------------------------------------------- R4
def test_r4_missing():
    df = pd.DataFrame({
        "Area": ["A", None, "A", "A", "A", "A", "A"],
        "Item": ["Wheat", "Wheat", "  ", "Wheat", "Wheat", "Wheat", "Wheat"],
        "Year": [2000, 2001, 2002, 2003, 2004, 2005, 2006],
        qc.TARGET: [10, 10, 10, "..", 10, 10, 0],           # «..» не число
        qc.RAIN: [500, 500, 500, 500, np.nan, 500, 500],    # пропуск
        qc.PEST: [1, 1, 1, 1, 1, 1, 0],                     # ноль - не пропуск
        qc.TEMP: [10.0] * 7,
    })
    mask, m = qc.rule_missing(df)
    assert mask_of(mask) == [False, True, True, True, True, False, False]
    assert m["rows_with_missing"] == 4


# ---------------------------------------------------------------- R5
def test_r5_constancy():
    df = table([
        ["A", "Wheat", 2000, 10, 500, 1, 10.0],   # у A осадки постоянны
        ["A", "Wheat", 2001, 11, 500, 1, 10.0],
        ["B", "Wheat", 2000, 12, 400, 1, 10.0],   # у B осадки меняются
        ["B", "Wheat", 2001, 13, 450, 1, 10.0],
    ])
    mask, m = qc.rule_constancy(df)
    assert mask_of(mask) == [True, True, False, False]
    assert m == {"areas": 2, "areas_rain_const": 1, "areas_rain_varies": 1}


def test_rules_work_without_rain_column():
    # после полной очистки столбца осадков нет: R4 не падает, R5 не применимо (нарушений нет)
    df = clean_table().drop(columns=qc.RAIN)
    results = qc.run_all(df)
    assert results["R5"].n_rows == 0 and results["R5"].status == "OK"
    assert results["R4"].n_rows == 0


# ---------------------------------------------------------------- общие
def clean_table():
    rows = []
    for area, shift in [("A", 0), ("B", 10)]:
        for i in range(5):
            rows.append([area, "Wheat", 2000 + i, 10 + i, 400 + 10 * i + shift,
                         1 + i + shift, 10.0 + i])
    return table(rows)


def test_clean_table_passes_everything():
    results = qc.run_all(clean_table())
    assert set(results) == {"R1", "R2", "R3", "R4", "R5"}
    assert {r.status for r in results.values()} == {"OK"}


def test_rules_do_not_change_data():
    df = clean_table()
    before = df.copy()
    qc.run_all(df)
    pd.testing.assert_frame_equal(df, before)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\nТестов: {len(tests)}, упало: {failed}")
    raise SystemExit(1 if failed else 0)
