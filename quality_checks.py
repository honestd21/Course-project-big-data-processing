"""Этап 2. Автоматические правила проверки качества данных для yield_df.csv.

Каждый дефект, измеренный в analysis.ipynb (этап 1), оформлен как правило.
Правило ничего не исправляет (очистка это cleaning.py). Оно только:
  * считает число нарушений и сопутствующие метрики;
  * возвращает булеву маску строк-нарушителей (нужна на этапе 3, чтобы считать
    полноту обнаружения и долю ложных срабатываний на искусственно испорченных данных).

Шесть правил:
  R1  key_unique            уникальность ключа (Area, Item, Year)
  R2  temp_consistent       одна температура на пару (Area, Year)
  R3  ranges                допустимые диапазоны + выбросы по IQR внутри культуры
  R4  missing               пропуски и нечисловые значения в числовых столбцах
  R5  constancy             осадки не должны быть константой внутри страны
  R6  names_consistent      согласованность названий стран и культур

Запуск:
    python3 quality_checks.py data/yield_df.csv
Код возврата 0, если все числа совпали с этапом 1, иначе 1.

Использование из кода:
    from quality_checks import load_data, run_all, summary_table
    results = run_all(load_data("data/yield_df.csv"))
    print(summary_table(results))
    results["R1"].mask        # строки, делящие ключ
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Конфигурация: имена столбцов и пороги
# --------------------------------------------------------------------------- #
KEY = ["Area", "Item", "Year"]
TARGET = "hg/ha_yield"
RAIN = "average_rain_fall_mm_per_year"
PEST = "pesticides_tonnes"
TEMP = "avg_temp"
NUMERIC = ["Year", TARGET, RAIN, PEST, TEMP]
REQUIRED = ["Area", "Item", *NUMERIC]

# Эталонный список культур (описание набора, раздел 2).
ITEMS = {
    "Cassava", "Maize", "Plantains and others", "Potatoes", "Rice, paddy",
    "Sorghum", "Soybeans", "Sweet potatoes", "Wheat", "Yams",
}

# Жёсткие физические диапазоны: (минимум, максимум, минимум включается?).
# Границы широкие: цель правила поймать невозможные значения, а не редкие.
RANGES = {
    "Year": (1990, 2013, True),        # период итоговой таблицы
    TARGET: (0, 1_000_000, False),     # hg/ha; 1 000 000 = 100 т/га, урожайность строго > 0
    RAIN: (0, 5_000, True),            # мм/год, среднее по стране
    PEST: (0, 1_000_000, True),        # тонны действующего вещества
    TEMP: (-50, 50, True),             # °C, среднегодовая
}

IQR_K = 1.5          # коэффициент правила IQR
SKEW_WARN = 1.0      # |асимметрия| выше порога даёт предупреждение
TEMP_TOL = 1e-9      # разрыв температуры внутри (страна, год) больше порога считаем конфликтом
# Столбцы, которые в таблице страна x год обязаны меняться по годам.
MUST_VARY_WITHIN_AREA = [RAIN]


# --------------------------------------------------------------------------- #
# Результат правила
# --------------------------------------------------------------------------- #
@dataclass
class RuleResult:
    rule_id: str
    name: str
    severity: str                      # "error" или "warning"
    n_rows: int                        # число строк-нарушителей
    n_total: int                       # размер таблицы
    mask: pd.Series                    # True у строк-нарушителей (индекс как у df)
    metrics: dict = field(default_factory=dict)
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.n_rows == 0

    @property
    def share(self) -> float:
        return self.n_rows / self.n_total if self.n_total else 0.0

    @property
    def status(self) -> str:
        if self.passed:
            return "OK"
        return "FAIL" if self.severity == "error" else "WARN"


def _result(df, rule_id, name, severity, mask, metrics, message):
    mask = mask.reindex(df.index, fill_value=False).astype(bool)
    return RuleResult(rule_id, name, severity, int(mask.sum()), len(df), mask, metrics, message)


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #
def load_data(path) -> pd.DataFrame:
    """Читает таблицу как есть и убирает служебный индекс `Unnamed: 0`.

    Типы не приводятся: нечисловые значения должен найти R4, а не тихо
    превратить в NaN при чтении.
    """
    df = pd.read_csv(path)
    return df.drop(columns="Unnamed: 0", errors="ignore")


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"В таблице нет обязательных столбцов: {missing}")


def _num(s: pd.Series) -> pd.Series:
    """Числовое представление столбца; нечисловое становится NaN (это ловит R4)."""
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------- #
# R1. Уникальность ключа (дефект 1)
# --------------------------------------------------------------------------- #
def check_key_unique(df: pd.DataFrame) -> RuleResult:
    """Одна запись об урожайности (Area, Item, Year) не должна встречаться дважды."""
    mask = df.duplicated(KEY, keep=False)               # все строки, делящие ключ
    extra = df.duplicated(KEY, keep="first")            # лишние копии (кроме первой)

    g = df[mask].groupby(KEY).agg(
        n_yield=(TARGET, "nunique"),
        n_temp=(TEMP, "nunique"),
        temp_gap=(TEMP, lambda s: _num(s).max() - _num(s).min()),
    )
    conf_temp = g[g["n_temp"] > 1]

    metrics = {
        "rows_sharing_key": int(mask.sum()),
        "share_sharing_key": round(mask.mean() * 100, 1),
        "unique_keys": int(len(df) - extra.sum()),
        "extra_rows": int(extra.sum()),
        "dup_keys": int(len(g)),
        "keys_temp_differs": int(len(conf_temp)),
        "keys_yield_differs": int((g["n_yield"] > 1).sum()),
        "median_temp_gap": round(float(conf_temp["temp_gap"].median()), 2) if len(conf_temp) else 0.0,
        "full_duplicates": int(df.duplicated().sum()),
    }
    msg = (f"{metrics['rows_sharing_key']} строк делят ключ, лишних {metrics['extra_rows']}; "
           f"полных дубликатов {metrics['full_duplicates']}")
    result = _result(df, "R1", "key_unique", "error", mask, metrics, msg)
    result.extra_mask = extra  # маска лишних копий: пригодится для cleaning.py
    return result


# --------------------------------------------------------------------------- #
# R2. Согласованность температуры (дефект 5)
# --------------------------------------------------------------------------- #
def check_temp_consistent(df: pd.DataFrame) -> RuleResult:
    """Температура зависит только от страны и года, но не от культуры."""
    t = _num(df[TEMP])
    gap = t.groupby([df["Area"], df["Year"]]).transform(lambda s: s.max() - s.min())
    mask = gap > TEMP_TOL

    pair_gap = (t.groupby([df["Area"], df["Year"]]).agg(lambda s: s.max() - s.min()))
    bad_pairs = pair_gap[pair_gap > TEMP_TOL]

    metrics = {
        "area_year_total": int(len(pair_gap)),
        "area_year_multi_temp": int(len(bad_pairs)),
        "share_area_year": round(len(bad_pairs) / len(pair_gap) * 100, 1) if len(pair_gap) else 0.0,
        "median_pair_gap": round(float(bad_pairs.median()), 2) if len(bad_pairs) else 0.0,
        "max_pair_gap": round(float(bad_pairs.max()), 2) if len(bad_pairs) else 0.0,
    }
    msg = f"{metrics['area_year_multi_temp']} из {metrics['area_year_total']} пар (страна, год) с разной температурой"
    return _result(df, "R2", "temp_consistent", "error", mask, metrics, msg)


# --------------------------------------------------------------------------- #
# R3. Диапазоны, выбросы внутри культуры, асимметрия (дефекты 6 и 7)
# --------------------------------------------------------------------------- #
def _iqr_mask(s: pd.Series, k: float = IQR_K) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def check_ranges(df: pd.DataFrame) -> RuleResult:
    """Две части.

    1. Жёсткие диапазоны (severity error): невозможные значения. На реальных данных их нет.
    2. Выбросы по IQR внутри культуры (severity warning): настоящие, но редкие значения.
       Они не ошибка данных, их судьбу решает cleaning.py и эксперименты.
    Асимметрия считается как метрика и даёт предупреждение при |skew| > SKEW_WARN.
    """
    hard = pd.Series(False, index=df.index)
    metrics = {}
    for col, (lo, hi, lo_incl) in RANGES.items():
        x = _num(df[col])
        below = (x < lo) if lo_incl else (x <= lo)
        bad = below | (x > hi)                       # NaN даёт False: пропуски это R4
        metrics[f"out_of_range_{col}"] = int(bad.sum())
        hard |= bad
    metrics["hard_range_rows"] = int(hard.sum())

    y = _num(df[TARGET])
    outl = y.groupby(df["Item"]).transform(lambda s: _iqr_mask(s)).astype(bool)
    share = (outl.groupby(df["Item"]).mean() * 100).round(1)
    metrics["outlier_rows_iqr"] = int(outl.sum())
    metrics["outlier_share_by_item"] = share.sort_values(ascending=False).to_dict()
    metrics["outlier_wheat"] = float(share.get("Wheat", np.nan))
    metrics["outlier_plantains"] = float(share.get("Plantains and others", np.nan))
    metrics["outlier_maize"] = float(share.get("Maize", np.nan))
    metrics["outlier_share_whole_table"] = round(float(_iqr_mask(y).mean() * 100), 1)

    metrics["skew_yield"] = round(float(y.skew()), 2)
    metrics["skew_pest"] = round(float(_num(df[PEST]).skew()), 2)
    metrics["skew_yield_log1p"] = round(float(np.log1p(y.clip(lower=0)).skew()), 2)
    metrics["skew_pest_log1p"] = round(float(np.log1p(_num(df[PEST]).clip(lower=0)).skew()), 2)
    skewed = [c for c, k in ((TARGET, metrics["skew_yield"]), (PEST, metrics["skew_pest"]))
              if abs(k) > SKEW_WARN]
    metrics["skewed_columns"] = skewed

    # Итоговая серьёзность: ошибка, если есть невозможные значения, иначе предупреждение.
    severity = "error" if hard.any() else "warning"
    mask = hard | outl
    metrics["mask_hard"] = hard
    metrics["mask_outliers"] = outl
    msg = (f"вне диапазонов {metrics['hard_range_rows']}; выбросы IQR внутри культуры {metrics['outlier_rows_iqr']}; "
           f"скошены: {', '.join(skewed) or 'нет'}")
    return _result(df, "R3", "ranges", severity, mask, metrics, msg)


# --------------------------------------------------------------------------- #
# R4. Пропуски и нечисловые значения
# --------------------------------------------------------------------------- #
def check_missing(df: pd.DataFrame) -> RuleResult:
    """Пропуск = NaN, пустая строка/пробелы в текстовых столбцах, либо значение,
    которое не читается как число в числовых (например «..» из исходных файлов FAO)."""
    mask = pd.Series(False, index=df.index)
    per_col = {}
    for col in ["Area", "Item"]:
        s = df[col]
        bad = s.isna() | s.astype("string").str.strip().eq("").fillna(False)
        per_col[col] = int(bad.sum())
        mask |= bad
    for col in NUMERIC:
        bad = _num(df[col]).isna()                    # NaN и нечисловое
        per_col[col] = int(bad.sum())
        mask |= bad

    metrics = {
        "missing": int(sum(per_col.values())),
        "missing_by_column": per_col,
        "rows_with_missing": int(mask.sum()),
    }
    msg = f"ячеек с пропуском или нечисловым значением: {metrics['missing']}, строк: {metrics['rows_with_missing']}"
    return _result(df, "R4", "missing", "error", mask, metrics, msg)


# --------------------------------------------------------------------------- #
# R5. Константность (дефект 3)
# --------------------------------------------------------------------------- #
def check_constancy(df: pd.DataFrame) -> RuleResult:
    """Признак, который должен меняться по годам, не должен быть константой внутри страны.

    Маска: строки стран, у которых столбец принимает ровно одно значение (при числе лет > 1).
    Для осадков на реальных данных это все строки: осадки здесь идентификатор страны.
    """
    n_years = df.groupby("Area")["Year"].nunique()
    mask = pd.Series(False, index=df.index)
    metrics = {"areas": int(df["Area"].nunique())}
    for col in MUST_VARY_WITHIN_AREA:
        nun = df.groupby("Area")[col].nunique()
        const_areas = nun[(nun == 1) & (n_years.reindex(nun.index) > 1)].index
        metrics[f"const_areas_{col}"] = int(len(const_areas))
        mask |= df["Area"].isin(const_areas)
    metrics["areas_rain_varies"] = int((df.groupby("Area")[RAIN].nunique() > 1).sum())
    metrics["areas_rain_const"] = metrics["areas"] - metrics["areas_rain_varies"]

    # Для сравнения (не нарушение): сколько разных значений внутри страны у остальных признаков.
    metrics["mean_unique_per_area"] = {
        c: round(float(df.groupby("Area")[c].nunique().mean()), 1) for c in (RAIN, TEMP, PEST)
    }
    # Глобальные константы: столбец с одним значением на всю таблицу.
    metrics["global_constant_columns"] = [c for c in REQUIRED if df[c].nunique(dropna=True) <= 1]

    msg = (f"осадки константны у {metrics['areas_rain_const']} из {metrics['areas']} стран; "
           f"глобальные константы: {metrics['global_constant_columns'] or 'нет'}")
    severity = "error"
    if metrics["global_constant_columns"]:
        mask = pd.Series(True, index=df.index)
    return _result(df, "R5", "constancy", severity, mask, metrics, msg)


# --------------------------------------------------------------------------- #
# R6. Согласованность названий
# --------------------------------------------------------------------------- #
def _norm(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.normalize("NFKC").str.strip()
              .str.replace(r"\s+", " ", regex=True).str.casefold())


def check_names_consistent(df: pd.DataFrame) -> RuleResult:
    """Одно и то же название записано одним способом.

    Нарушение: (а) пробелы по краям или двойные пробелы; (б) разные варианты
    регистра/пробелов одного названия; (в) культура вне эталонного списка.
    """
    mask = pd.Series(False, index=df.index)
    metrics = {}
    for col in ["Area", "Item"]:
        raw = df[col].astype("string")
        norm = _norm(df[col])
        ws = raw.notna() & (raw != raw.str.strip().str.replace(r"\s+", " ", regex=True))
        # Если у одного названия несколько написаний, нарушителем считается редкое:
        # строки с самым частым написанием в группе не помечаются.
        variants = raw.groupby(norm).nunique()
        multi = variants[variants > 1].index
        dominant = raw.groupby(norm).transform(lambda s: s.mode().iloc[0] if s.notna().any() else pd.NA)
        split = norm.isin(multi) & norm.notna() & (raw != dominant)
        metrics[f"{col}_whitespace_rows"] = int(ws.sum())
        metrics[f"{col}_variant_groups"] = int(len(multi))
        mask |= ws.fillna(False) | split.fillna(False)

    unknown = ~df["Item"].isin(ITEMS) & df["Item"].notna()
    metrics["item_unknown_rows"] = int(unknown.sum())
    metrics["item_unknown_values"] = sorted(map(str, df.loc[unknown, "Item"].unique()))[:20]
    mask |= unknown

    metrics["n_areas_raw"] = int(df["Area"].nunique())
    metrics["n_areas_normalized"] = int(_norm(df["Area"]).nunique())
    metrics["n_items_raw"] = int(df["Item"].nunique())
    msg = (f"строк с неоднородными названиями: {int(mask.sum())}; "
           f"стран {metrics['n_areas_raw']} (после нормализации {metrics['n_areas_normalized']}), "
           f"культур {metrics['n_items_raw']}")
    return _result(df, "R6", "names_consistent", "error", mask, metrics, msg)


# --------------------------------------------------------------------------- #
# Запуск всех правил и отчёт
# --------------------------------------------------------------------------- #
RULES = {
    "R1": check_key_unique,
    "R2": check_temp_consistent,
    "R3": check_ranges,
    "R4": check_missing,
    "R5": check_constancy,
    "R6": check_names_consistent,
}


def run_all(df: pd.DataFrame, rules: dict | None = None) -> dict[str, RuleResult]:
    _check_columns(df)
    return {rid: fn(df) for rid, fn in (rules or RULES).items()}


def summary_table(results: dict[str, RuleResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {"правило": r.rule_id, "имя": r.name, "серьёзность": r.severity, "статус": r.status,
         "строк-нарушителей": r.n_rows, "доля, %": round(r.share * 100, 1), "суть": r.message}
        for r in results.values()
    ])


def any_row_flagged(results: dict[str, RuleResult], rules=None) -> pd.Series:
    """Строка помечена хотя бы одним правилом (для подсчёта полноты и ложных срабатываний)."""
    masks = [r.mask for k, r in results.items() if rules is None or k in rules]
    return pd.concat(masks, axis=1).any(axis=1)


# --------------------------------------------------------------------------- #
# Сверка с этапом 1
# --------------------------------------------------------------------------- #
# Числа из analysis.ipynb («Сводка: ожидали / получили»). Для двух расхождений
# с dataset_description_1.md берём измеренные значения, а не значения описания.
STAGE1 = {
    "rows_sharing_key":     ("R1", "rows_sharing_key", 18949),
    "share_sharing_key":    ("R1", "share_sharing_key", 67.1),
    "unique_keys":          ("R1", "unique_keys", 13130),
    "extra_rows":           ("R1", "extra_rows", 15112),
    "keys_temp_differs":    ("R1", "keys_temp_differs", 3690),
    "keys_yield_differs":   ("R1", "keys_yield_differs", 0),
    "median_temp_gap":      ("R1", "median_temp_gap", 4.49),
    "full_duplicates":      ("R1", "full_duplicates", 2310),
    "area_year_total":      ("R2", "area_year_total", 2250),
    "area_year_multi_temp": ("R2", "area_year_multi_temp", 575),
    "hard_range_rows":      ("R3", "hard_range_rows", 0),
    "skew_yield":           ("R3", "skew_yield", 1.79),
    "skew_pest":            ("R3", "skew_pest", 3.49),
    "outlier_wheat":        ("R3", "outlier_wheat", 8.6),
    "outlier_plantains":    ("R3", "outlier_plantains", 7.6),
    "outlier_maize":        ("R3", "outlier_maize", 5.0),
    "missing":              ("R4", "missing", 0),
    "areas_rain_const":     ("R5", "areas_rain_const", 101),
    "areas_rain_varies":    ("R5", "areas_rain_varies", 0),
    "names_variant_rows":   ("R6", None, 0),   # число строк-нарушителей R6
}


def verify_against_stage1(results: dict[str, RuleResult], tol: float = 0.05) -> pd.DataFrame:
    rows = []
    for name, (rid, key, expected) in STAGE1.items():
        got = results[rid].n_rows if key is None else results[rid].metrics[key]
        rows.append({"показатель": name, "правило": rid, "этап 1": expected, "правило нашло": got,
                     "совпало": bool(abs(got - expected) < tol)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    path = Path(argv[0]) if argv else Path("data/yield_df.csv")
    df = load_data(path)
    results = run_all(df)

    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print(f"Файл: {path}  |  строк: {len(df)}\n")
    print(summary_table(results).to_string(index=False))
    check = verify_against_stage1(results)
    print("\nСверка с этапом 1:")
    print(check.to_string(index=False))
    ok = bool(check["совпало"].all())
    print("\nИТОГ:", "правила находят ровно то, что измерено на этапе 1" if ok
          else "ЕСТЬ РАСХОЖДЕНИЯ с этапом 1")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
