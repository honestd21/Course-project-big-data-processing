"""Этап 4. Порча очищенной таблицы и оценка полноты и ложных срабатываний правил R1-R5.

Для каждого правила своя порча на одном уровне (FRAC = 5% единиц порчи, seed фиксирован):
  R4  в 5% строк один столбец заменяется на пропуск, «..» или пустое название
  R2  у 5% пар (Area, Year), где есть минимум 2 строки, температура одной строки сдвигается на 3 °C
  R1  5% строк копируются (полные дубликаты)
  R3  у 5% строк урожайность ставится за границу IQR (в 2-4 IQR от квартиля своей культуры)
  R5  осадки: после очистки столбца нет, поэтому к таблице добавляются синтетические осадки,
      меняющиеся по годам; у 5% стран их делаем константой

Каждая функция порчи берёт таблицу и возвращает (base, bad, truth):
  base  - таблица до порчи (для R5 с добавленным столбцом осадков)
  bad   - испорченная таблица
  truth - True у строк, которые правило должно пометить (для R1 и R2 это все строки группы)
Порча делается на копии, исходная таблица не меняется. Каждое правило проверяется на своей порче.

Оценка (evaluate):
  полнота              доля строк из truth, которые правило пометило
  ложные срабатывания  помеченные строки вне truth и вне срабатываний на base: так срабатывания
                       правил на исходной таблице (например, новые выбросы R3 после очистки)
                       не считаются ложными

Запуск: python3 corruption.py data/yield_clean.csv
  сначала мини-тесты порчи, потом оценка; результат в data/corruption_eval.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quality_checks import (KEY, NUMERIC, PEST, RAIN, RULES, TARGET, TEMP, load_data,
                            missing_mask, outlier_mask, rule_temp_consistent)

FRAC = 0.05
SEED = 42
IQR_MARGIN = (2.0, 4.0)   # насколько далеко за границу IQR ставится испорченная урожайность
TEMP_SHIFT = 3.0          # на сколько градусов сдвигается температура


def _pick(n, frac, rng, pool=None):
    """Позиции для порчи: frac * n штук (минимум 1) из pool (по умолчанию из range(n))."""
    pool = np.arange(n) if pool is None else pool
    return rng.choice(pool, size=min(len(pool), max(1, round(frac * n))), replace=False)


def _truth(index, labels):
    t = pd.Series(False, index=index)
    t.loc[labels] = True
    return t


# ---------------------------------------------------------------- порча
def corrupt_missing(df, rng, frac=FRAC):
    out = df.copy()
    pos = _pick(len(df), frac, rng)
    cols = [c for c in NUMERIC if c in df.columns] + ["Area", "Item"]
    chosen = rng.choice(cols, size=len(pos))
    is_nan = rng.random(len(pos)) < 0.5            # иначе «..» (числа) или пустая строка (названия)
    for c in set(chosen):
        out[c] = out[c].astype(object)
        sel = chosen == c
        rows = out.index[pos[sel]]
        out.loc[rows[is_nan[sel]], c] = np.nan
        out.loc[rows[~is_nan[sel]], c] = ".." if c in NUMERIC else ""
    return df, out, _truth(out.index, out.index[pos])


def corrupt_temperature(df, rng, frac=FRAC):
    out = df.copy()
    sizes = out.groupby(["Area", "Year"]).size()
    elig = sizes[sizes >= 2].index                 # в паре с одной строкой конфликт не создать
    chosen = elig[_pick(len(elig), frac, rng)]
    in_pair = pd.MultiIndex.from_frame(out[["Area", "Year"]]).isin(chosen)
    truth = pd.Series(in_pair, index=out.index)
    first = out[truth].drop_duplicates(["Area", "Year"]).index
    out.loc[first, TEMP] = out.loc[first, TEMP] + rng.choice([-1.0, 1.0], size=len(first)) * TEMP_SHIFT
    return df, out, truth


def corrupt_duplicates(df, rng, frac=FRAC):
    src = df.iloc[_pick(len(df), frac, rng)]
    copies = src.copy()
    copies.index = range(int(df.index.max()) + 1, int(df.index.max()) + 1 + len(copies))
    out = pd.concat([df, copies])
    return df, out, _truth(out.index, src.index.append(copies.index))


def corrupt_outliers(df, rng, frac=FRAC):
    out = df.copy()
    out[TARGET] = out[TARGET].astype(float)
    g = out[TARGET].groupby(out["Item"])
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    pool = np.flatnonzero(~outlier_mask(out).to_numpy())   # строки, которые уже выбросы, не берём
    pos = _pick(len(out), frac, rng, pool)
    mult = rng.uniform(*IQR_MARGIN, size=len(pos))
    hi = q3.iloc[pos].to_numpy() + mult * iqr.iloc[pos].to_numpy()
    lo = q1.iloc[pos].to_numpy() - mult * iqr.iloc[pos].to_numpy()
    use_hi = (rng.random(len(pos)) < 0.5) | (lo <= 0)      # урожайность не бывает отрицательной
    out.loc[out.index[pos], TARGET] = np.where(use_hi, hi, lo)
    return df, out, _truth(out.index, out.index[pos])


def corrupt_constancy(df, rng, frac=FRAC):
    base = df.copy()
    ay = base[["Area", "Year"]].drop_duplicates()
    level = pd.Series(rng.uniform(200, 2500, base["Area"].nunique()), index=base["Area"].unique())
    ay[RAIN] = level[ay["Area"]].to_numpy() * np.exp(rng.normal(0, 0.15, len(ay)))
    rain = ay.set_index(["Area", "Year"])[RAIN]
    base[RAIN] = rain.reindex(pd.MultiIndex.from_frame(base[["Area", "Year"]])).to_numpy()
    out = base.copy()
    areas = base["Area"].unique()
    chosen = areas[_pick(len(areas), frac, rng)]
    for a in chosen:
        m = out["Area"] == a
        out.loc[m, RAIN] = out.loc[m, RAIN].iloc[0]
    return base, out, out["Area"].isin(chosen)


CORRUPTIONS = {"R1": corrupt_duplicates, "R2": corrupt_temperature, "R3": corrupt_outliers,
               "R4": corrupt_missing, "R5": corrupt_constancy}


# ---------------------------------------------------------------- оценка
def evaluate(rid, base, bad, truth):
    """Полнота и ложные срабатывания одного правила (только его, не run_all: на порче R4
    остаются нечисловые значения, на которых другие правила падают)."""
    rule = RULES[rid][2]
    flag = rule(bad)[0]
    base_flag = rule(base)[0].reindex(bad.index, fill_value=False)
    n_true, found = int(truth.sum()), int((flag & truth).sum())
    return {"правило": rid, "строк": len(bad), "внесено": n_true, "найдено": found,
            "полнота, %": round(100 * found / n_true, 1) if n_true else float("nan"),
            "ложных срабатываний": int((flag & ~truth & ~base_flag).sum()),
            "срабатываний до порчи": int(base_flag.sum())}


def run_evaluation(df, seed=SEED, frac=FRAC):
    rows = []
    for rid, corrupt in CORRUPTIONS.items():
        rng = np.random.default_rng([seed, int(rid[1:])])
        base, bad, truth = corrupt(df, rng, frac)
        rows.append(evaluate(rid, base, bad, truth))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- мини-тесты
def toy(n_area=6, n_year=5):
    """Чистая таблица с известным устройством: у каждой пары (Area, Year) 3 строки и одна температура."""
    rng = np.random.default_rng(0)
    areas = [f"A{i}" for i in range(n_area)]
    df = pd.DataFrame([(a, i, y) for a in areas for i in ("Maize", "Rice", "Wheat")
                       for y in range(2000, 2000 + n_year)], columns=KEY)
    df[TARGET] = 100 + rng.normal(0, 5, len(df))
    df[PEST] = rng.uniform(1, 50, len(df))
    df[TEMP] = 10 + df["Area"].str[1:].astype(int) + (df["Year"] - 2000) * 0.1
    return df


def _same(a, b):
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def _run(corrupt, df=None, seed=1, frac=0.1):
    df = toy() if df is None else df
    return corrupt(df, np.random.default_rng(seed), frac)


def test_missing():
    base, bad, truth = _run(corrupt_missing)
    assert truth.sum() == 9
    assert (missing_mask(bad) == truth).all()          # помечены ровно испорченные строки
    _same(bad[~truth], base[~truth])


def test_duplicates():
    base, bad, truth = _run(corrupt_duplicates)
    assert len(bad) == len(base) + 9 and truth.sum() == 18
    assert (bad.duplicated(KEY, keep=False) == truth).all()
    assert len(bad.drop_duplicates()) == len(base)     # копии полные
    _same(bad.iloc[:len(base)], base)


def test_temperature():
    base, bad, truth = _run(corrupt_temperature)
    assert truth.sum() == 9                            # 3 пары по 3 строки
    assert (rule_temp_consistent(bad)[0] == truth).all()
    assert (bad[TEMP] != base[TEMP]).sum() == 3        # по одной строке на пару
    _same(bad[~truth], base[~truth])


def test_outliers():
    base, bad, truth = _run(corrupt_outliers)
    assert truth.sum() == 9
    g = base[TARGET].groupby(base["Item"])
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    new = bad.loc[truth, TARGET]
    assert ((new > q3[truth] + 2 * iqr[truth] - 1e-9) | (new < q1[truth] - 2 * iqr[truth] + 1e-9)).all()
    _same(bad[~truth], base[~truth])


def test_constancy():
    base, bad, truth = _run(corrupt_constancy)
    assert (base.groupby("Area")[RAIN].nunique() > 1).all()   # до порчи осадки меняются
    n = bad.groupby("Area")[RAIN].nunique()
    chosen = bad.loc[truth, "Area"].unique()
    assert len(chosen) == 1 and (n[chosen] == 1).all()
    _same(bad[~truth], base[~truth])


def test_input_not_modified():
    df = toy()
    before = df.copy()
    for corrupt in CORRUPTIONS.values():
        corrupt(df, np.random.default_rng(1), 0.1)
        _same(df, before)


def test_reproducible():
    for corrupt in CORRUPTIONS.values():
        _, a, ta = _run(corrupt)
        _, b, tb = _run(corrupt)
        _same(a, b)
        assert (ta == tb).all()


def test_evaluate_arithmetic():
    base, bad, truth = _run(corrupt_missing)
    r = evaluate("R4", base, bad, truth)
    assert r["полнота, %"] == 100 and r["ложных срабатываний"] == 0 and r["срабатываний до порчи"] == 0
    extra = truth.copy()
    extra[extra[~extra].index[0]] = True                # лишняя «внесённая» строка, которую правило не пометит
    r = evaluate("R4", base, bad, extra)
    assert r["внесено"] == 10 and r["найдено"] == 9 and r["ложных срабатываний"] == 0
    fewer = truth.copy()
    fewer[fewer[fewer].index[0]] = False               # одну помеченную строку не засчитали как внесённую
    r = evaluate("R4", base, bad, fewer)
    assert r["внесено"] == 8 and r["ложных срабатываний"] == 1


def run_tests():
    tests = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    failed = 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                         # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"Тестов: {len(tests)}, упало: {failed}")
    return failed == 0


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/yield_clean.csv")
    print("Мини-тесты порчи:")
    if not run_tests():
        sys.exit(1)
    df = load_data(src)
    table = run_evaluation(df)
    pd.set_option("display.width", 200)
    print(f"\nОценка правил на {src.name}: порча {FRAC:.0%}, seed {SEED}")
    print(table.to_string(index=False))
    print("\nR5 проверялся на синтетических осадках (в очищенной таблице столбца нет).")
    table.to_csv(src.with_name("corruption_eval.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {src.parent}/corruption_eval.csv")
