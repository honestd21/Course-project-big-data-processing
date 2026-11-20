"""Этап 2. Шесть автоматических правил проверки качества yield_df.csv.

Правило получает таблицу, ничего в ней не меняет и возвращает:
  mask    - True у строк-нарушителей (понадобится для оценки полноты обнаружения)
  metrics - числа для сверки с этапом 1

  R1 key_unique       ключ (Area, Item, Year) не повторяется
  R2 temp_consistent  одна температура на пару (Area, Year)
  R3 ranges           значения в допустимых диапазонах; выбросы IQR внутри культуры
  R4 missing          нет пропусков и нечисловых значений
  R5 constancy        осадки не постоянны внутри страны
  R6 names            названия записаны единообразно

Запуск: python quality_checks.py data/yield_df.csv
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
ITEMS = {"Cassava", "Maize", "Plantains and others", "Potatoes", "Rice, paddy",
         "Sorghum", "Soybeans", "Sweet potatoes", "Wheat", "Yams"}

# Допустимые диапазоны (min, max). Широкие: ловим невозможное, а не редкое.
RANGES = {"Year": (1990, 2013), TARGET: (1, 1_000_000), RAIN: (0, 5000),
          PEST: (0, 1_000_000), TEMP: (-50, 50)}
IQR_K = 1.5


def load_data(path):
    return pd.read_csv(path).drop(columns="Unnamed: 0", errors="ignore")


def norm_name(s):
    """Название без лишних пробелов и в нижнем регистре."""
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()


# ---------------------------------------------------------------- маски
def hard_range_mask(df):
    """Невозможные значения (пропуски сюда не входят, это R4)."""
    bad = pd.Series(False, index=df.index)
    for col, (lo, hi) in RANGES.items():
        x = pd.to_numeric(df[col], errors="coerce")
        bad |= x.notna() & ~x.between(lo, hi)
    return bad


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
        "median_temp_gap": round(float(gap.median()), 2),
        "full_duplicates": int(df.duplicated().sum()),
    }


def rule_temp_consistent(df):
    g = df.groupby(["Area", "Year"])[TEMP]
    mask = (g.transform("max") - g.transform("min")) > 1e-9
    return mask, {"area_year_total": g.ngroups,
                  "area_year_multi_temp": int(((g.max() - g.min()) > 1e-9).sum())}


def rule_ranges(df):
    hard, out = hard_range_mask(df), outlier_mask(df)
    share = (out.groupby(df["Item"]).mean() * 100).round(1)
    metrics = {"hard_range_rows": int(hard.sum()), "outlier_rows": int(out.sum()),
               "skew_yield": round(float(df[TARGET].skew()), 2),
               "skew_pest": round(float(df[PEST].skew()), 2)}
    metrics.update({f"outlier_share_{item}": v for item, v in share.items()})
    return hard | out, metrics


def rule_missing(df):
    mask = missing_mask(df)
    return mask, {"rows_with_missing": int(mask.sum())}


def rule_constancy(df):
    n = df.groupby("Area")[RAIN].nunique()
    const = n[n == 1].index
    return df["Area"].isin(const), {"areas": len(n), "areas_rain_const": len(const),
                                    "areas_rain_varies": len(n) - len(const)}


def rule_names(df):
    """Нарушение: лишние пробелы, редкое написание при нескольких (регистр),
    культура вне списка ITEMS."""
    mask = ~df["Item"].isin(ITEMS) & df["Item"].notna()
    for col in ["Area", "Item"]:
        raw = df[col].astype("string")
        dominant = raw.groupby(norm_name(raw)).transform(lambda s: s.mode().iloc[0])
        spaces = raw != raw.str.strip().str.replace(r"\s+", " ", regex=True)
        mask |= (raw.notna() & ((raw != dominant) | spaces)).fillna(False)
    return mask, {"rows_bad_names": int(mask.sum())}


# id: (имя, серьёзность, функция)
RULES = {
    "R1": ("key_unique", "error", rule_key_unique),
    "R2": ("temp_consistent", "error", rule_temp_consistent),
    "R3": ("ranges", "warning", rule_ranges),
    "R4": ("missing", "error", rule_missing),
    "R5": ("constancy", "error", rule_constancy),
    "R6": ("names", "error", rule_names),
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
        if self.n_rows == 0:
            return "OK"
        return "FAIL" if self.severity == "error" else "WARN"


def run_all(df):
    results = {}
    for rid, (name, severity, fn) in RULES.items():
        mask, metrics = fn(df)
        if rid == "R3" and metrics["hard_range_rows"]:
            severity = "error"          # невозможные значения это уже ошибка, а не выбросы
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
    ("R3", "hard_range_rows", 0), ("R3", "skew_yield", 1.79), ("R3", "skew_pest", 3.49),
    ("R3", "outlier_share_Wheat", 8.6), ("R3", "outlier_share_Plantains and others", 7.6),
    ("R3", "outlier_share_Maize", 5.0),
    ("R4", "rows_with_missing", 0),
    ("R5", "areas_rain_const", 101), ("R5", "areas_rain_varies", 0),
    ("R6", "rows_bad_names", 0),
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
