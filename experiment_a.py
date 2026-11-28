"""Эксперимент A. Влияние дефектов данных на качество моделей (задача 6).

Состояния данных (из cleaning.py):
  исходные            yield_df.csv как есть
  без дублей          только шаг R1 (осадки остаются)
  полная очистка      шаги R4, R2, R1, R3, R5 (столбца осадков нет)

Модели: Ridge, RandomForest, HistGradientBoosting. Целевая переменная и пестициды проходят
через log1p внутри пайплайна (TransformedTargetRegressor, FunctionTransformer), предобработка
обучается только на обучающей части. Признаки: Item, Year, пестициды, температура, осадки
(если столбец есть). Area в признаки не входит: страны в тесте не видны модели.

Протокол: разбиение по странам (страны случайно делятся на --folds групп, как в GroupKFold,
но с перемешиванием по seed), несколько seed. Разбиение стран одинаково для всех состояний.
Тестовая выборка общая для всех состояний: строки полной очистки в тестовых странах
(осадки добавлены по стране из исходных данных: внутри страны они константа).
Метрики считаются в исходной шкале (hg/ha). Среднее по фолдам внутри seed, затем среднее ± std по seed.

Запуск: python3 experiment_a.py data/yield_df.csv [--seeds 3] [--folds 5]
Сохраняет в папку данных: experiment_a_results.csv (все фолды), experiment_a_summary.csv,
experiment_a_predictions.csv (предсказания для первого seed, для анализа ошибок).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cleaning import build_states
from quality_checks import NUMERIC, PEST, RAIN, TARGET, TEMP, load_data

SEED = 42
METRICS = ["MAE", "RMSE", "R2"]
FEATURES = ["Item", "Year", PEST, TEMP, RAIN]
MODELS = {
    "Ridge": lambda seed: Ridge(alpha=1.0),
    "RandomForest": lambda seed: RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                       n_jobs=-1, random_state=seed),
    "HistGradientBoosting": lambda seed: HistGradientBoostingRegressor(random_state=seed),
}


def prepare(df):
    d = df.copy()
    cols = [c for c in NUMERIC if c in d.columns]
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")
    return d.dropna(subset=[TARGET])


def make_states(df):
    states, _ = build_states(df)
    return {name: prepare(d) for name, d in states.items()}


def make_pipeline(model, features):
    impute_scale = lambda: [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    num = [c for c in ["Year", TEMP, RAIN] if c in features]
    pre = ColumnTransformer([
        ("pest", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), [PEST]),
        ("num", Pipeline(impute_scale()), num),
        ("item", OneHotEncoder(handle_unknown="ignore"), ["Item"]),
    ])
    return TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("model", model)]),
                                      func=np.log1p, inverse_func=np.expm1)


def country_folds(countries, n_folds, seed):
    return np.array_split(np.random.default_rng(seed).permutation(countries), n_folds)


def scores(y, pred):
    return {"MAE": mean_absolute_error(y, pred), "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": r2_score(y, pred)}


def run(df, n_seeds=3, n_folds=5):
    states = make_states(df)
    clean = states["полная очистка"].copy()
    # осадки константа страны (дефект 2), поэтому для теста берём значение страны из исходных данных
    clean[RAIN] = clean["Area"].map(prepare(df).groupby("Area")[RAIN].median())
    countries = np.array(sorted(df["Area"].unique()))
    seeds = [SEED + i for i in range(n_seeds)]
    rows, preds = [], []
    t0 = time.time()
    for seed in seeds:
        for k, test_c in enumerate(country_folds(countries, n_folds, seed)):
            clean_test = clean[clean["Area"].isin(test_c)]
            for sname, d in states.items():
                train = d[~d["Area"].isin(test_c)]
                feats = [c for c in FEATURES if c in d.columns]
                for mname, make in MODELS.items():
                    est = make_pipeline(make(seed), feats).fit(train[feats], train[TARGET])
                    p = est.predict(clean_test[feats])
                    rows.append({"состояние": sname, "модель": mname,
                                 "seed": seed, "фолд": k, **scores(clean_test[TARGET], p)})
                    if seed == seeds[0]:
                        preds.append(clean_test[["Area", "Item", "Year"]].assign(
                            y=clean_test[TARGET].to_numpy(), pred=p, состояние=sname, модель=mname))
            print(f"seed {seed}, фолд {k + 1}/{n_folds} готов, {time.time() - t0:.0f} с", flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize(res):
    """Среднее по фолдам внутри seed, затем среднее и std по seed."""
    keys = ["состояние", "модель"]
    per_seed = res.groupby(keys + ["seed"])[METRICS].mean()
    g = per_seed.groupby(level=keys)
    mean, std = g.mean(), g.std()
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    return out.reset_index()


def show(summary):
    fmt = {"MAE": 0, "RMSE": 0, "R2": 3}
    view = summary[["состояние", "модель"]].copy()
    for m, nd in fmt.items():
        view[m] = [f"{a:.{nd}f}" + ("" if np.isnan(b) else f" ± {b:.{nd}f}")
                   for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    print("\nРезультаты эксперимента A:")
    print(view.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/yield_df.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    src = Path(a.path)
    res, preds = run(load_data(src), a.seeds, a.folds)
    summary = summarize(res)
    pd.set_option("display.width", 200)
    show(summary)
    res.to_csv(src.with_name("experiment_a_results.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(src.with_name("experiment_a_summary.csv"), index=False, encoding="utf-8-sig")
    preds.to_csv(src.with_name("experiment_a_predictions.csv"), index=False, encoding="utf-8-sig")
    print(f"\nСохранено в {src.parent}/: experiment_a_results.csv, experiment_a_summary.csv, "
          "experiment_a_predictions.csv")
