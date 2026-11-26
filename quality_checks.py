"""Этап 2. Пять автоматических правил проверки качества yield_df.csv.

Правило получает таблицу, ничего в ней не меняет и возвращает:
  mask    - True у строк-нарушителей (понадобится для оценки полноты обнаружения)
  metrics - числа для сверки с этапом 1

  R1 key_unique       ключ (Area, Item, Year) не повторяется            (дефект 1)
  R2 temp_consistent  одна температура на пару (Area, Year)             (дефект 3)
  R3 outliers_skew    выбросы IQR внутри культуры; асимметрия столбцов  (дефекты 4 и 5)
  R4 missing          нет пропусков и нечисловых значений
  R5 constancy        осадки не постоянны внутри страны                 (дефект 2)
                      (если столбца осадков нет, правило не применимо: нарушений 0)

Асимметрия считается по столбцу целиком, строк-нарушителей у неё нет. Она даёт только
предупреждение и в оценку полноты обнаружения (задача 5) не входит.

Запуск: python3 quality_checks.py data/yield_df.csv
"""
import sys
from dataclasses import dataclass

import pandas as pd

KEY = ["Area", "Item", "Year"]
TARGET = "hg/ha_yield"
RAIN = "average_rain_fall_mm_per_year"
PEST = "pesticides_tonnes"
TEMP = "avg_temp"
NUMERIC = ["Year", TARGET, RAIN, PEST, TEMP]

IQR_K = 1.5
SKEW_COLS = [TARGET, PEST]   # столбцы, для которых проверяем асимметрию (дефекты 4 и 5)
SKEW_LIMIT = 1.0             # |асимметрия| выше порога даёт предупреждение


def load_data(path):
    return pd.read_csv(path).drop(columns="Unnamed: 0", errors="ignore")


# ---------------------------------------------------------------- маски
def outlier_mask(df):
    """Выбросы урожайности по правилу IQR внутри своей культуры."""
    g = df[TARGET].groupby(df["Item"])
    q1 = g.transform(lambda s: s.quantile(0.25))
    q3 = g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return (df[TARGET] < q1 - IQR_K * iqr) | (df[TARGET] > q3 + IQR_K * iqr)


def missing_mask(df):
    """Пропуск, пустое название или значение, не читаемое как число (например «..»)."""
    bad = pd.Series(False, index=df.index)
    for col in ["Area", "Item"]:
        bad |= df[col].isna() | (df[col].astype("string").str.strip() == "")
    for col in NUMERIC:
        if col in df.columns:   # после полной очистки столбца осадков уже нет (шаг R5)
            bad |= pd.to_numeric(df[col], errors="coerce").isna()
    return bad


# ---------------------------------------------------------------- правила
def rule_key_unique(df):
    mask = df.duplicated(KEY, keep=False)
    g = df[mask].groupby(KEY).agg(t_min=(TEMP, "min"), t_max=(TEMP, "max"),
                                  n_yield=(TARGET, "nunique"))
    gap = (g.t_max - g.t_min)[g.t_max > g.t_min]
    extra = int(df.duplicated(KEY).sum())
    return mask, {
        "rows_sharing_key": int(mask.sum()),
        "share_sharing_key": round(mask.mean() * 100, 1),
        "unique_keys": len(df) - extra,
        "extra_rows": extra,
        "keys_temp_differs": len(gap),
        "keys_yield_differs": int((g.n_yield > 1).sum()),
        "median_temp_gap": round(float(gap.median()), 2) if len(gap) else 0.0,
        "full_duplicates": int(df.duplicated().sum()),
    }


def rule_temp_consistent(df):
    g = df.groupby(["Area", "Year"])[TEMP]
    mask = (g.transform("max") - g.transform("min")) > 1e-9
    return mask, {"area_year_total": g.ngroups,
                  "area_year_multi_temp": int(((g.max() - g.min()) > 1e-9).sum())}


def rule_outliers_skew(df):
    """Строки-нарушители: выбросы IQR. Асимметрия идёт в метрики и в предупреждение."""
    out = outlier_mask(df)
    share = (out.groupby(df["Item"]).mean() * 100).round(1)
    skew = {col: float(pd.to_numeric(df[col], errors="coerce").skew()) for col in SKEW_COLS}
    metrics = {"outlier_rows": int(out.sum()),
               "skew_yield": round(skew[TARGET], 2),
               "skew_pest": round(skew[PEST], 2),
               "skewed_cols": sum(abs(v) > SKEW_LIMIT for v in skew.values())}
    metrics.update({f"outlier_share_{item}": v for item, v in share.items()})
    return out, metrics


def rule_missing(df):
    mask = missing_mask(df)
    return mask, {"rows_with_missing": int(mask.sum())}


def rule_constancy(df):
    if RAIN not in df.columns:   # столбец удалён очисткой: правило не применимо, нарушений нет
        return pd.Series(False, index=df.index), {"areas": df["Area"].nunique(),
                                                  "areas_rain_const": 0, "areas_rain_varies": 0}
    n = df.groupby("Area")[RAIN].nunique()
    const = n[n == 1].index
    return df["Area"].isin(const), {"areas": len(n), "areas_rain_const": len(const),
                                    "areas_rain_varies": len(n) - len(const)}


# id: (имя, серьёзность, функция)
RULES = {
    "R1": ("key_unique", "error", rule_key_unique),
    "R2": ("temp_consistent", "error", rule_temp_consistent),
    "R3": ("outliers_skew", "warning", rule_outliers_skew),
    "R4": ("missing", "error", rule_missing),
    "R5": ("constancy", "error", rule_constancy),
}


@dataclass
class Result:
    name: str
    severity: str
    mask: pd.Series
    metrics: dict

    @property
    def n_rows(self):
        return int(self.mask.sum())

    @property
    def status(self):
        # строк-нарушителей нет и асимметрии нет -> OK
        if self.n_rows == 0 and not self.metrics.get("skewed_cols", 0):
            return "OK"
        return "FAIL" if self.severity == "error" else "WARN"


def run_all(df):
    results = {}
    for rid, (name, severity, fn) in RULES.items():
        mask, metrics = fn(df)
        results[rid] = Result(name, severity, mask.astype(bool), metrics)
    return results


def summary_table(results):
    return pd.DataFrame([{"правило": k, "имя": r.name, "статус": r.status,
                          "строк-нарушителей": r.n_rows,
                          "доля, %": round(r.n_rows / len(r.mask) * 100, 1)}
                         for k, r in results.items()])


# ---------------------------------------------------------------- сверка с этапом 1
# (правило, показатель, значение из блокнота)
STAGE1 = [
    ("R1", "rows_sharing_key", 18949), ("R1", "share_sharing_key", 67.1),
    ("R1", "unique_keys", 13130), ("R1", "extra_rows", 15112),
    ("R1", "keys_temp_differs", 3690), ("R1", "keys_yield_differs", 0),
    ("R1", "median_temp_gap", 4.49), ("R1", "full_duplicates", 2310),
    ("R2", "area_year_total", 2250), ("R2", "area_year_multi_temp", 575),
    ("R3", "skew_yield", 1.79), ("R3", "skew_pest", 3.49),
    ("R3", "outlier_share_Wheat", 8.6), ("R3", "outlier_share_Plantains and others", 7.6),
    ("R3", "outlier_share_Maize", 5.0),
    ("R4", "rows_with_missing", 0),
    ("R5", "areas_rain_const", 101), ("R5", "areas_rain_varies", 0),
]


def verify_stage1(results):
    rows = [{"правило": rid, "показатель": key, "этап 1": want,
             "найдено": results[rid].metrics[key],
             "совпало": abs(results[rid].metrics[key] - want) < 0.05}
            for rid, key, want in STAGE1]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/yield_df.csv"
    df = load_data(path)
    results = run_all(df)
    pd.set_option("display.width", 200)
    print(f"Строк: {len(df)}\n")
    print(summary_table(results).to_string(index=False))
    check = verify_stage1(results)
    print("\nСверка с этапом 1:")
    print(check.to_string(index=False))
    ok = bool(check["совпало"].all())
    print("\nИТОГ:", "всё совпало с этапом 1" if ok else "ЕСТЬ РАСХОЖДЕНИЯ")
    sys.exit(0 if ok else 1)
