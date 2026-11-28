"""Построение графиков результатов, анализ ошибок и важности признаков (разделы 2.3, 2.4.5, 2.4.6).

Использует результаты работы модулей quality_checks.py, cleaning.py, corruption.py и experiment_a.py:
  yield_df.csv, yield_dedup.csv, yield_clean.csv
  corruption_eval.csv
  experiment_a_summary.csv, experiment_a_predictions.csv

Сохраняет графики (fig3..fig9) в папку figures/ и таблицы анализа ошибок
(error_by_item.csv, error_by_area.csv, feature_importance.csv) в папку данных.

Запуск: python3 analysis_plots.py data/yield_df.csv [--fig-dir figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from quality_checks import PEST, TARGET, TEMP, load_data, run_all

SEED = 42
C_RAW = "#C0392B"
C_DEDUP = "#D68910"
C_CLEAN = "#2E86C1"
C_GREEN = "#27AE60"
C_GRAY = "#7F8C8D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_ridge_pipeline():
    pre = ColumnTransformer([
        ("pest", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("sc", StandardScaler()),
        ]), [PEST]),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), ["Year", TEMP]),
        ("item", OneHotEncoder(handle_unknown="ignore"), ["Item"]),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))]),
        func=np.log1p, inverse_func=np.expm1,
    )


def plot_violations(data_dir, fig_dir):
    states = {
        "исходные": load_data(data_dir / "yield_df.csv"),
        "без дублей": load_data(data_dir / "yield_dedup.csv"),
        "полная очистка": load_data(data_dir / "yield_clean.csv"),
    }
    res = {k: run_all(v) for k, v in states.items()}
    rids = ["R1", "R2", "R3", "R4", "R5"]
    labels = [f"{r}\n{res['исходные'][r].name}" for r in rids]
    v_raw = [res["исходные"][r].n_rows for r in rids]
    v_dedup = [res["без дублей"][r].n_rows for r in rids]
    v_clean = [res["полная очистка"][r].n_rows for r in rids]

    x = np.arange(len(rids))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    b1 = ax.bar(x - w, v_raw, w, label=f"Исходные ({len(states['исходные']):,} строк)".replace(",", " "), color=C_RAW, alpha=0.88)
    b2 = ax.bar(x, v_dedup, w, label=f"Без дублей ({len(states['без дублей']):,} строк)".replace(",", " "), color=C_DEDUP, alpha=0.88)
    b3 = ax.bar(x + w, v_clean, w, label=f"Полная очистка ({len(states['полная очистка']):,} строк)".replace(",", " "), color=C_CLEAN, alpha=0.88)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h):,}".replace(",", " "),
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Число строк-нарушителей")
    ax.set_ylim(0, max(v_raw) * 1.15)
    ax.legend(frameon=False, loc="upper left")
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_violations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_corruption(data_dir, fig_dir):
    corr_df = pd.read_csv(data_dir / "corruption_eval.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax = axes[0]
    bars = ax.bar(corr_df["правило"], corr_df["полнота, %"], color=C_GREEN, width=0.52, alpha=0.88)
    for bar, n_in in zip(bars, corr_df["внесено"]):
        ax.annotate(f"100%\n({n_in}/{n_in})",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 124)
    ax.set_ylabel("Полнота обнаружения, %")
    ax.set_title("Полнота обнаружения внесённых дефектов")
    ax.xaxis.grid(False)

    ax = axes[1]
    xr = np.arange(len(corr_df))
    w2 = 0.35
    b_fp = ax.bar(xr - w2 / 2, corr_df["ложных срабатываний"], w2, label="Ложные срабатывания", color=C_RAW, alpha=0.88)
    b_pre = ax.bar(xr + w2 / 2, corr_df["срабатываний до порчи"], w2, label="Срабатывания до порчи", color=C_GRAY, alpha=0.85)
    for bars in (b_fp, b_pre):
        for bar in bars:
            h = int(bar.get_height())
            ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xr)
    ax.set_xticklabels(corr_df["правило"])
    ax.set_ylim(0, 265)
    ax.set_ylabel("Число строк")
    ax.set_title("Ложные срабатывания и исходный фон")
    ax.legend(frameon=False, loc="upper left")
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_corruption_eval.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_metrics(data_dir, fig_dir):
    summ = pd.read_csv(data_dir / "experiment_a_summary.csv")
    if "тест" in summ.columns:
        summ = summ[summ["тест"] == "чистая"].copy()
    models_order = ["Ridge", "RandomForest", "HistGradientBoosting"]
    states_order = ["исходные", "без дублей", "полная очистка"]
    state_labels = ["Исходные", "Без дублей", "Полная очистка"]
    state_colors = [C_RAW, C_DEDUP, C_CLEAN]
    xm = np.arange(len(models_order))
    wm = 0.27

    # Figure 5: MAE (Y-axis from 0)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for i, (st, lbl, col) in enumerate(zip(states_order, state_labels, state_colors)):
        part = summ[summ["состояние"] == st].set_index("модель").loc[models_order]
        bars = ax.bar(xm + (i - 1) * wm, part["MAE_mean"], wm, yerr=part["MAE_std"],
                      capsize=4, label=lbl, color=col, alpha=0.88, error_kw={"elinewidth": 1.1})
        for idx_b, (bar, val) in enumerate(zip(bars, part["MAE_mean"])):
            ax.annotate(f"{int(round(val)):,}".replace(",", " "),
                        xy=(bar.get_x() + bar.get_width() / 2, val + part["MAE_std"].iloc[idx_b]),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xm)
    ax.set_xticklabels(models_order)
    ax.set_ylabel("MAE, hg/ha")
    ax.set_ylim(0, 36000)
    ax.legend(frameon=False, loc="upper right", ncol=3)
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_mae_by_state.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: RMSE and R2 (no MAE duplication, Y-axes from 0)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    cfg = [
        ("RMSE", "RMSE, hg/ha", (0, 60000), 0),
        ("R2", "R²", (0, 0.80), 3),
    ]
    short_models = ["Ridge", "RandomForest", "HGB"]
    for ax, (m, ylabel, ylim, nd) in zip(axes, cfg):
        for i, (st, lbl, col) in enumerate(zip(states_order, state_labels, state_colors)):
            part = summ[summ["состояние"] == st].set_index("модель").loc[models_order]
            bars = ax.bar(xm + (i - 1) * wm, part[f"{m}_mean"], wm, yerr=part[f"{m}_std"],
                          capsize=3, label=lbl, color=col, alpha=0.88, error_kw={"elinewidth": 1.0})
            for idx_b, (bar, val) in enumerate(zip(bars, part[f"{m}_mean"])):
                txt = f"{int(round(val)):,}".replace(",", " ") if nd == 0 else f"{val:.3f}".replace(".", ",")
                ax.annotate(txt,
                            xy=(bar.get_x() + bar.get_width() / 2, val + part[f"{m}_std"].iloc[idx_b]),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(xm)
        ax.set_xticklabels(short_models)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(m if m != "R2" else "R²")
        ax.xaxis.grid(False)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_all_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_error_analysis(data_dir, fig_dir):
    preds = pd.read_csv(data_dir / "experiment_a_predictions.csv")
    best = preds[(preds["состояние"] == "полная очистка") & (preds["модель"] == "Ridge")].copy()
    best["err"] = best["y"] - best["pred"]
    best["abs_err"] = best["err"].abs()

    by_area = best.groupby("Area").agg(
        n=("y", "size"),
        mean_y=("y", "mean"),
        MAE=("abs_err", "mean"),
        RMSE=("err", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        bias=("err", "mean"),
    ).sort_values("MAE", ascending=False)
    by_area.to_csv(data_dir / "error_by_area.csv", encoding="utf-8-sig")

    by_item = best.groupby("Item").agg(
        n=("y", "size"),
        mean_y=("y", "mean"),
        MAE=("abs_err", "mean"),
        RMSE=("err", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        bias=("err", "mean"),
    ).sort_values("MAE", ascending=False)
    by_item["rel_MAE_%"] = by_item["MAE"] / by_item["mean_y"] * 100
    by_item.to_csv(data_dir / "error_by_item.csv", encoding="utf-8-sig")

    # Figure 7: Top 15 countries by MAE
    top15 = by_area.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.barh(top15.index, top15["MAE"], color=C_CLEAN, alpha=0.88)
    for bar, val in zip(bars, top15["MAE"]):
        ax.annotate(f"{int(round(val)):,}".replace(",", " "),
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
    ax.set_xlabel("MAE, hg/ha")
    ax.set_xlim(0, 122000)
    ax.set_title("15 стран с наибольшей средней абсолютной ошибкой (Ridge, полная очистка)")
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_errors_by_area.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 8: Actual vs Predicted and Residuals vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    ax.scatter(best["pred"], best["y"], s=6, alpha=0.22, color=C_CLEAN, edgecolors="none")
    lim = max(best["y"].max(), best["pred"].max()) * 1.03
    ax.plot([0, lim], [0, lim], "--", color=C_RAW, linewidth=1.2, label="y = ŷ")
    ax.set_xlabel("Прогноз урожайности ŷ, hg/ha")
    ax.set_ylabel("Фактическая урожайность y, hg/ha")
    ax.set_title("Факт и прогноз")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.scatter(best["pred"], best["err"], s=6, alpha=0.22, color=C_CLEAN, edgecolors="none")
    ax.axhline(0, color=C_RAW, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Прогноз урожайности ŷ, hg/ha")
    ax.set_ylabel("Остаток (y − ŷ), hg/ha")
    ax.set_title("Остатки модели от прогноза")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_residuals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 9: Permutation importance + Ridge coefficients
    clean = load_data(data_dir / "yield_clean.csv")
    feats = ["Item", "Year", PEST, TEMP]
    feat_ru = {"Item": "Культура (Item)", PEST: "Пестициды (log1p)", TEMP: "Температура", "Year": "Год"}
    countries = np.array(sorted(clean["Area"].unique()))
    folds = np.array_split(np.random.default_rng(SEED).permutation(countries), 5)
    perm_rows = []
    for test_c in folds:
        tr, te = clean[~clean["Area"].isin(test_c)], clean[clean["Area"].isin(test_c)]
        est = make_ridge_pipeline().fit(tr[feats], tr[TARGET])
        pi = permutation_importance(est, te[feats], te[TARGET],
                                    scoring="neg_mean_absolute_error", n_repeats=10, random_state=SEED)
        perm_rows.append(pd.Series(pi.importances_mean, index=feats))
    perm_df = pd.DataFrame(perm_rows)
    fi_summary = pd.DataFrame({"mean_dMAE": perm_df.mean(), "std_dMAE": perm_df.std()}).sort_values("mean_dMAE", ascending=False)
    fi_summary.to_csv(data_dir / "feature_importance.csv", encoding="utf-8-sig")

    pipe = make_ridge_pipeline().fit(clean[feats], clean[TARGET])
    fnames = pipe.regressor_.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(pipe.regressor_.named_steps["model"].coef_, index=fnames)
    coefs.index = [
        n.replace("item__Item_", "")
         .replace(f"pest__{PEST}", "Пестициды (log1p)")
         .replace(f"num__{TEMP}", "Температура")
         .replace("num__Year", "Год")
        for n in coefs.index
    ]
    coefs_sorted = coefs.reindex(coefs.abs().sort_values().index)
    p_mean = perm_df.mean().sort_values()
    p_std = perm_df.std().reindex(p_mean.index)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    bars = ax.barh([feat_ru[k] for k in p_mean.index], p_mean.values, xerr=p_std.values,
                   capsize=3, color=C_CLEAN, alpha=0.88)
    for bar, val, sd in zip(bars, p_mean.values, p_std.values):
        ax.annotate(f"+{int(round(val)):,}".replace(",", " "),
                    xy=(val + sd, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
    ax.set_xlabel("Прирост MAE при перестановке, hg/ha")
    ax.set_xlim(0, 46500)
    ax.set_title("Перестановочная важность (5 фолдов)")
    ax.yaxis.grid(False)

    ax = axes[1]
    colors_c = [C_CLEAN if v >= 0 else C_RAW for v in coefs_sorted.values]
    ax.barh(coefs_sorted.index, coefs_sorted.values, color=colors_c, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент Ridge (шкала log1p(y))")
    ax.set_title("Коэффициенты гребневой регрессии")
    ax.yaxis.grid(False)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig9_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Ridge (полная очистка, seed={SEED}): MAE={mean_absolute_error(best['y'], best['pred']):.1f}, "
          f"RMSE={np.sqrt(mean_squared_error(best['y'], best['pred'])):.1f}, "
          f"R2={r2_score(best['y'], best['pred']):.3f}")
    print(f"Остатки: среднее={best['err'].mean():.1f}, медиана={best['err'].median():.1f}, "
          f"доля y>pred={(best['err'] > 0).mean()*100:.1f}%, "
          f"r(y, err)={best['y'].corr(best['err']):.3f}, r(pred, |err|)={best['pred'].corr(best['abs_err']):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/yield_df.csv")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()
    data_dir = Path(args.path).parent
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_violations(data_dir, fig_dir)
    plot_corruption(data_dir, fig_dir)
    plot_experiment_metrics(data_dir, fig_dir)
    run_error_analysis(data_dir, fig_dir)
    print(f"Графики сохранены в {fig_dir}/, таблицы ошибок — в {data_dir}/")
