"""Этап 3. Очистка yield_df.csv по правилам из quality_checks.py, с логом.

Шаги (порядок важен):
  R6  единое написание названий
  R4  удалить строки с пропусками и нечисловыми значениями
  R3  удалить строки с невозможными значениями
  R2  температура внутри (Area, Year) = медиана (до R1, иначе копии ключа останутся разными)
  R1  оставить одну строку на ключ (Area, Item, Year)

Намеренно НЕ делаем: не удаляем выбросы IQR и не заполняем пропуски (это факторы
экспериментов), не трогаем константные осадки (это свойство источника, R5).

Три состояния данных:
  исходные        load_data(path)
  без дублей      clean(df, only_dedup=True)   только шаг R1
  полная очистка  clean(df)

Запуск: python cleaning.py data/yield_df.csv
Сохраняет data/yield_dedup.csv, data/yield_clean.csv, data/cleaning_log.csv
"""
import sys
from pathlib import Path

import pandas as pd

from quality_checks import (ITEMS, KEY, NUMERIC, TEMP, hard_range_mask, load_data,
                            missing_mask, norm_name, run_all)


def clean(df, only_dedup=False):
    """Возвращает (очищенная таблица, лог шагов)."""
    log, cur = [], df

    def step(rule, action, new, changed=0):
        nonlocal cur
        log.append({"правило": rule, "действие": action, "строк до": len(cur),
                    "строк после": len(new), "изменено значений": changed})
        cur = new

    if not only_dedup:
        # R6: культуры по эталонному списку, страны по самому частому написанию
        out = cur.copy()
        canon = {i.casefold(): i for i in ITEMS}
        item = norm_name(out["Item"]).map(canon).fillna(out["Item"])
        area = out["Area"].groupby(norm_name(out["Area"])).transform(lambda s: s.mode().iloc[0])
        changed = int((item != out["Item"]).sum() + (area != out["Area"]).sum())
        out["Item"], out["Area"] = item, area
        step("R6", "единое написание названий", out, changed)

        # R4: пропуски удаляем, заполнять не будем (способ заполнения это фактор эксперимента)
        out = cur[~missing_mask(cur)].copy()
        out[NUMERIC] = out[NUMERIC].apply(pd.to_numeric)
        step("R4", "удаление строк с пропусками", out)

        # R3: невозможные значения
        step("R3", "удаление строк вне диапазонов", cur[~hard_range_mask(cur)])

        # R2: температура = медиана по (страна, год)
        med = cur.groupby(["Area", "Year"])[TEMP].transform("median")
        changed = int(((cur[TEMP] - med).abs() > 1e-9).sum())
        step("R2", "температура = медиана по (страна, год)", cur.assign(**{TEMP: med}), changed)

    # R1: одна строка на ключ (остаётся первая)
    step("R1", "удаление копий ключа", cur.drop_duplicates(KEY))
    return cur, pd.DataFrame(log)


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


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/yield_df.csv")
    df = load_data(src)
    states, log = build_states(df)

    pd.set_option("display.width", 200)
    print("Лог полной очистки:")
    print(log.to_string(index=False))
    print("\nЧисло нарушений по состояниям данных:")
    print(states_table(states).to_string(index=False))

    states["без дублей"].to_csv(src.with_name("yield_dedup.csv"), index=False)
    states["полная очистка"].to_csv(src.with_name("yield_clean.csv"), index=False)
    log.to_csv(src.with_name("cleaning_log.csv"), index=False, encoding="utf-8-sig")
    print(f"\nСохранено в {src.parent}/: yield_dedup.csv, yield_clean.csv, cleaning_log.csv")
