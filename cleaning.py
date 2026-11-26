"""Этап 3. Очистка yield_df.csv по правилам из quality_checks.py, с логом.

Шаги (порядок важен):
  R4  удалить строки с пропусками и нечисловыми значениями
  R2  температура внутри (Area, Year) = медиана (до R1, иначе копии ключа останутся разными)
  R1  оставить одну строку на ключ (Area, Item, Year)
  R3  удалить выбросы урожайности по IQR внутри культуры (после R1, чтобы копии строк
      не влияли на квартили); один проход, поэтому после очистки R3 может сработать снова
  R5  удалить столбец осадков (константа внутри страны, исправить значения нечем)

Пропуски не заполняем (способ заполнения это фактор эксперимента). Асимметрия (часть R3)
в очистку не входит: у неё нет строк-нарушителей, log1p делается внутри Pipeline.

Три состояния данных:
  исходные        load_data(path)
  без дублей      clean(df, only_dedup=True)   только шаг R1 (столбец осадков остаётся)
  полная очистка  clean(df)                    шаги R4, R2, R1, R3, R5

Запуск: python3 cleaning.py data/yield_df.csv
Сохраняет data/yield_dedup.csv, data/yield_clean.csv, data/cleaning_log.csv
"""
import sys
from pathlib import Path

import pandas as pd

from quality_checks import (KEY, NUMERIC, RAIN, TEMP, load_data, missing_mask, outlier_mask,
                            run_all)


def clean(df, only_dedup=False):
    """Возвращает (очищенная таблица, лог шагов). Исходная таблица не меняется."""
    log, cur = [], df

    def step(rule, action, new, changed=0):
        nonlocal cur
        log.append({"правило": rule, "действие": action, "строк до": len(cur),
                    "строк после": len(new), "изменено значений": changed})
        cur = new

    if not only_dedup:
        # R4: пропуски удаляем, заполнять не будем (способ заполнения это фактор эксперимента)
        out = cur[~missing_mask(cur)].copy()
        out[NUMERIC] = out[NUMERIC].apply(pd.to_numeric)
        step("R4", "удаление строк с пропусками", out)

        # R2: температура = медиана по (страна, год)
        med = cur.groupby(["Area", "Year"])[TEMP].transform("median")
        changed = int(((cur[TEMP] - med).abs() > 1e-9).sum())
        step("R2", "температура = медиана по (страна, год)", cur.assign(**{TEMP: med}), changed)

    # R1: одна строка на ключ (остаётся первая)
    step("R1", "удаление копий ключа", cur.drop_duplicates(KEY))

    if not only_dedup:
        # R3: выбросы IQR внутри культуры
        step("R3", "удаление выбросов урожайности (IQR внутри культуры)", cur[~outlier_mask(cur)])

        # R5: осадки постоянны внутри страны, столбец удаляем
        step("R5", "удаление столбца осадков", cur.drop(columns=RAIN))
    return cur.reset_index(drop=True), pd.DataFrame(log)


def build_states(df):
    """Три состояния данных: исходные, без дублей, полная очистка (+ лог полной очистки)."""
    full, log = clean(df)
    return {"исходные": df, "без дублей": clean(df, only_dedup=True)[0],
            "полная очистка": full}, log


def states_table(states):
    """Число строк и число нарушений каждого правила в каждом состоянии."""
    res = {name: run_all(d) for name, d in states.items()}
    rows = [{"показатель": "строк", **{n: len(d) for n, d in states.items()}}]
    for rid, r in next(iter(res.values())).items():
        rows.append({"показатель": f"{rid} {r.name}", **{n: x[rid].n_rows for n, x in res.items()}})
    return pd.DataFrame(rows)


def expected_removed(df):
    """Сколько строк должно уйти на каждом шаге, считается по исходной таблице напрямую."""
    ok = df[~missing_mask(df)].copy()
    ok[NUMERIC] = ok[NUMERIC].apply(pd.to_numeric)
    dedup = ok.drop_duplicates(KEY)
    return {"R4": int(missing_mask(df).sum()),
            "R1": len(ok) - len(dedup),
            "R3": int(outlier_mask(dedup).sum())}


def removal_table(df, log):
    """Сравнение: сколько строк надо было убрать и сколько убрал каждый шаг очистки."""
    exp = expected_removed(df)
    actual = {r["правило"]: r["строк до"] - r["строк после"] for r in log.to_dict("records")}
    rows = [{"правило": r, "надо убрать": exp[r], "убрано": actual[r],
             "совпало": exp[r] == actual[r]} for r in exp]
    return pd.DataFrame(rows)


def check_clean(df, cleaned, log):
    """Прямые проверки полной очистки, не через правила (поэтому они независимы от них)."""
    results = run_all(cleaned)
    keys_before = df[~missing_mask(df)].drop_duplicates(KEY)[KEY]
    removal = removal_table(df, log)
    exp_total = int(removal["надо убрать"].sum())
    checks = {
        "ключ (Area, Item, Year) уникален": not cleaned.duplicated(KEY).any(),
        "одна температура на (страна, год)":
            bool((cleaned.groupby(["Area", "Year"])[TEMP].nunique() == 1).all()),
        "пропусков и нечисловых значений нет": not missing_mask(cleaned).any(),
        "столбца осадков нет": RAIN not in cleaned.columns,
        "строк не больше, чем уникальных ключей среди строк без пропусков":
            len(cleaned) <= len(keys_before),
        "R1, R2, R4, R5: 0 нарушений": all(results[r].n_rows == 0 for r in ("R1", "R2", "R4", "R5")),
        "по каждому правилу убрано ровно столько строк, сколько надо": bool(removal["совпало"].all()),
        "итого убрано = сумма того, что надо было убрать": len(df) - len(cleaned) == exp_total,
    }
    return pd.DataFrame({"проверка": list(checks), "выполнено": list(checks.values())})


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/yield_df.csv")
    df = load_data(src)
    states, log = build_states(df)

    pd.set_option("display.width", 200)
    print("Лог полной очистки:")
    print(log.to_string(index=False))
    print("\nЧисло нарушений по состояниям данных:")
    print(states_table(states).to_string(index=False))
    print("\nСколько строк надо было убрать и сколько убрано:")
    print(removal_table(df, log).to_string(index=False))
    check = check_clean(df, states["полная очистка"], log)
    print("\nПроверка полной очистки:")
    print(check.to_string(index=False))

    states["без дублей"].to_csv(src.with_name("yield_dedup.csv"), index=False)
    states["полная очистка"].to_csv(src.with_name("yield_clean.csv"), index=False)
    log.to_csv(src.with_name("cleaning_log.csv"), index=False, encoding="utf-8-sig")
    print(f"\nСохранено в {src.parent}/: yield_dedup.csv, yield_clean.csv, cleaning_log.csv")
    sys.exit(0 if bool(check["выполнено"].all()) else 1)
