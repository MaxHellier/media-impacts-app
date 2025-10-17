# app.py — Media Impacts: Quick ITS (all-in-one web app)
# Features:
# - Data sources: Google Trends (0–100 index) or Wikipedia pageviews (counts)
# - Auto-fallback from Trends -> Wikipedia if rate-limited
# - Caching for speed
# - Plain-English summary with significance flags
# - Core charts + Performance Visuals Pack (counterfactual, excess, cumulative, event study, decay half-life, placebo check)
# - Date comparison table (e.g., theatrical vs streaming vs YouTube)
# - Interest-by-region (Trends)
# - Batch mode (upload a CSV)
# - Downloads: CSVs, ZIP bundle, PowerPoint

import io, time, zipfile, datetime as dt
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

# --------------------------- UI basics ---------------------------
st.set_page_config(page_title="🎬 Media Impacts — Quick ITS", layout="centered")
st.title("🎬 Media Impacts — Quick ITS")
st.caption("Enter a film and outcome, pick a date, press **Run analysis**. View results, compare dates, and download a report.")

# --------------------------- Data fetchers ---------------------------
def fetch_trends_weekly(queries: List[str], geo: str, timeframe: str = "today 5-y",
                        hl: str = "en-US", tz: int = 0, tries: int = 6) -> pd.Series:
    """Google Trends weekly series (W-MON), averaged across up to 5 queries. Manual retry to avoid urllib3 issues."""
    from pytrends.request import TrendReq
    from pytrends import exceptions as pex

    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise ValueError("Please provide at least one query.")
    queries = queries[:5]  # Google supports ≤5

    py = TrendReq(
        hl=hl, tz=tz, retries=0, backoff_factor=0, timeout=(10, 30),
        requests_args={"headers": {"User-Agent": "media-impacts-app/0.1"}}
    )

    wait = 5
    for i in range(tries):
        try:
            py.build_payload(queries, timeframe=timeframe, geo=geo)
            df = py.interest_over_time()
            if df.empty:
                raise RuntimeError("Google Trends returned no data; try a different GEO or timeframe.")
            if "isPartial" in df.columns: df = df.drop(columns=["isPartial"])
            s = df[queries].mean(axis=1).resample("W-MON").mean()
            return s.rename("value")
        except Exception as e:
            msg = str(e)
            if isinstance(e, pex.TooManyRequestsError) or "429" in msg:
                st.info(f"Rate limited (429). Retrying in {wait}s…")
                time.sleep(wait); wait = min(wait*2, 60); continue
            if "code 400" in msg and timeframe != "today 5-y":
                timeframe = "today 5-y"  # safer default
                continue
            raise

def wiki_weekly_pageviews(title: str, start: str, end: str, lang: str = "en") -> pd.Series:
    """Wikipedia weekly pageviews (sum of daily), W-MON index, with polite UA."""
    import requests
    from urllib.parse import quote

    session = requests.Session()
    session.headers.update({"User-Agent": "media-impacts-app/0.1 (contact: you@example.com)"})

    # Validate page exists
    chk = session.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}", timeout=15)
    chk.raise_for_status()

    r = session.get(
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/user/{quote(title, safe='')}/daily/{start}/{end}",
        timeout=30
    )
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise RuntimeError(f"No pageview data for “{title}”.")
    idx = [pd.to_datetime(it["timestamp"][:8]) for it in items]
    vals = [it["views"] for it in items]
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name=title).resample("W-MON").sum()

# --------------------------- Modeling ---------------------------
def quick_its(outcome: pd.Series, film: pd.Series, intervention_date: str):
    """Segmented regression: level + slope change w/ HAC errors. Returns model, df, figs, metrics."""
    df = pd.concat(
        {"outcome": outcome.rename("outcome"), "film_interest": film.rename("film_interest")},
        axis=1
    ).dropna()
    if len(df) < 12:
        raise RuntimeError("Not enough weekly data after alignment. Try a wider timeframe or different source.")

    df["time"] = np.arange(len(df))
    t0 = pd.to_datetime(intervention_date)
    t0_monday = (t0 - pd.offsets.Week(weekday=0))
    df["post"] = (df.index >= t0_monday).astype(int)
    df["time_post"] = df["time"] * df["post"]
    df["film_lag1"] = df["film_interest"].shift(1).fillna(method="bfill")

    X = sm.add_constant(df[["time", "post", "time_post", "film_lag1"]])
    y = df["outcome"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    # Figures (core)
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["outcome"], label="Outcome interest")
    plt.axvline(t0_monday, linestyle="--", label=f"Intervention: {t0.date()}")
    plt.plot(df.index, model.predict(X), label="Fitted (segmented regression)")
    plt.legend(); plt.title("Outcome vs intervention — quick ITS"); plt.tight_layout()

    fig2 = plt.figure(figsize=(10, 2.8))
    plt.plot(df.index, df["film_interest"], label="Film interest")
    plt.axvline(t0_monday, linestyle="--")
    plt.title("Film interest"); plt.tight_layout()

    # Metrics
    pre_mean = df.loc[df.index < t0_monday, "outcome"].mean()
    level = float(model.params.get("post", np.nan))
    slope = float(model.params.get("time_post", np.nan))
    ci = model.conf_int()
    level_lo, level_hi = [float(x) for x in ci.loc["post"]] if "post" in ci.index else (np.nan, np.nan)
    slope_lo, slope_hi = [float(x) for x in ci.loc["time_post"]] if "time_post" in ci.index else (np.nan, np.nan)

    metrics = {
        "level_change": level,
        "level_change_pct_of_pre": float(100 * level / pre_mean) if pre_mean else np.nan,
        "level_ci_lo": level_lo,
        "level_ci_hi": level_hi,
        "slope_change_per_week": slope,
        "slope_ci_lo": slope_lo,
        "slope_ci_hi": slope_hi,
        "t0_monday": t0_monday,
        "pre_mean": float(pre_mean) if pre_mean else np.nan,
    }
    return model, df, (fig1, fig2), metrics

def trends_interest_by_region(query: str, geo: str = "", timeframe: str = "today 5-y") -> pd.DataFrame:
    """Table of attention by region (Google Trends)."""
    from pytrends.request import TrendReq
    py = TrendReq(hl="en-US", tz=0, retries=0, backoff_factor=0,
                  requests_args={"headers":{"User-Agent":"media-impacts-app/0.1"}})
    py.build_payload([query], timeframe=timeframe, geo=geo)
    df = py.interest_by_region(resolution="COUNTRY" if geo=="" else "REGION",
                               inc_low_vol=True, inc_geo_code=True)
    df = df.reset_index()
    name_col = "geoName" if "geoName" in df.columns else df.columns[0]
    out = df.rename(columns={name_col:"region"})
    return out.sort_values(query, ascending=False).reset_index(drop=True)

def compare_interventions(dates: list, outcome: pd.Series, film: pd.Series) -> pd.DataFrame:
    rows = []
    for d in dates:
        model, dfX, _, m = quick_its(outcome, film, d)
        rows.append({
            "intervention": d,
            "level_change": m["level_change"],
            "level_%_pre": m["level_change_pct_of_pre"],
            "level_p": float(model.pvalues.get("post", np.nan)),
            "slope_change": m["slope_change_per_week"],
            "slope_p": float(model.pvalues.get("time_post", np.nan)),
        })
    return pd.DataFrame(rows).sort_values("level_%_pre", ascending=False)

def make_ppt(film, outcome, intervention, figs, metrics) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()

    # Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Media Impacts — Quick ITS"
    slide.placeholders[1].text = f"Film: {film}\nOutcome: {outcome}\nIntervention: {intervention}"

    # Numbers
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Results"
    body = slide.shapes.placeholders[1].text_frame
    body.text = (f"Immediate level change: {metrics['level_change']:.0f} "
                 f"({metrics['level_change_pct_of_pre']:.1f}% of pre)\n"
                 f"Slope change per week: {metrics['slope_change_per_week']:.3f}\n"
                 f"95% CIs — level: {metrics['level_ci_lo']:.0f}…{metrics['level_ci_hi']:.0f}; "
                 f"slope: {metrics['slope_ci_lo']:.3f}…{metrics['slope_ci_hi']:.3f}")

    # Figures
    for fig in figs:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        img = io.BytesIO()
        fig.savefig(img, format="png", dpi=200, bbox_inches="tight")
        img.seek(0)
        slide.shapes.add_picture(img, Inches(0.5), Inches(1.0), width=Inches(8.5))
    buf = io.BytesIO(); prs.save(buf); buf.seek(0)
    return buf.getvalue()

# --------------------------- Caching wrappers ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_trends(queries, geo, timeframe):
    return fetch_trends_weekly(queries, geo=geo, timeframe=timeframe)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wiki(title, start, end, lang="en"):
    return wiki_weekly_pageviews(title, start, end, lang=lang)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_regions(query, geo, timeframe):
    return trends_interest_by_region(query, geo=geo, timeframe=timeframe)

# --------------------------- Sidebar inputs ---------------------------
with st.sidebar:
    st.header("Inputs")

    data_source = st.radio(
        "Data source",
        ["Google Trends (index 0–100)", "Wikipedia pageviews (counts)"],
        help="Trends can rate-limit; Wikipedia is instant.",
        index=0
    )
    auto_fallback = st.checkbox("Auto-fallback to Wikipedia if Trends fails", value=True)

    film_input = st.text_input(
        "Film (title or Trends terms)",
        value="Eating Our Way to Extinction",
        help="Trends: comma-separated variants (max 5). Wikipedia: exact page title."
    )
    outcome_input = st.text_input(
        "Outcome (Trends term or Wikipedia page)",
        value="plant-based food" if "Google" in data_source else "Plant-based diet"
    )

    col1, col2 = st.columns(2)
    with col1:
        geo = st.selectbox("Region (Trends)", ["", "US", "GB", "TH"], index=1, help="'' = worldwide")
    with col2:
        timeframe = st.selectbox("Timeframe (Trends)", ["today 5-y", "today 12-m", "today 3-m"], index=0)

    intervention = st.date_input("Intervention date (release window)", dt.date(2021, 9, 30))

    st.markdown("---")
    st.caption("Wikipedia-only options")
    start_daily = st.text_input("Start YYYYMMDD", "20190101")
    end_daily   = st.text_input("End YYYYMMDD",   dt.date.today().strftime("%Y%m%d"))
    lang        = st.text_input("Wikipedia language code", "en")

    st.markdown("---")
    multi_dates = st.text_input("Compare dates (comma-separated)", "2021-09-30, 2022-07-01")
    show_regions = st.checkbox("Show interest by region (Trends)", value=False)

    st.markdown("---")
    st.caption("Batch mode CSV: film,outcome,intervention,source,geo")
    batch_file = st.file_uploader("Upload batch CSV", type=["csv"])

    run = st.button("▶️ Run analysis", use_container_width=True)

# --------------------------- Main run ---------------------------
if run:
    try:
        with st.spinner("Fetching series…"):
            if "Google" in data_source:
                film_qs = [q.strip() for q in film_input.split(",")]
                outcome_qs = [q.strip() for q in outcome_input.split(",")]
                try:
                    film_s = cached_trends(film_qs, geo=geo, timeframe=timeframe).rename("film_interest")
                    out_s  = cached_trends(outcome_qs, geo=geo, timeframe=timeframe).rename("outcome")
                except Exception as e:
                    if auto_fallback:
                        st.warning(f"Google Trends failed ({e}). Falling back to Wikipedia pageviews.")
                        film_s = cached_wiki(film_input, start_daily, end_daily, lang=lang).rename("film_interest")
                        out_s  = cached_wiki(outcome_input, start_daily, end_daily, lang=lang).rename("outcome")
                    else:
                        raise
            else:
                film_s = cached_wiki(film_input, start_daily, end_daily, lang=lang).rename("film_interest")
                out_s  = cached_wiki(outcome_input, start_daily, end_daily, lang=lang).rename("outcome")

        with st.spinner("Running ITS…"):
            model, df, figs, metrics = quick_its(out_s, film_s, intervention.isoformat())

        # ---- Plain-English summary ----
        sig_level = 0.05
        p_post     = float(model.pvalues.get("post", 1.0))
        p_timepost = float(model.pvalues.get("time_post", 1.0))
        flag = (lambda p: "✅" if p < sig_level else "⚠️")

        st.subheader("Results (plain English)")
        st.write(
            f"{flag(p_post)} Around **{intervention}**, the outcome shifted by "
            f"**{metrics['level_change']:.0f}** "
            f"({metrics['level_change_pct_of_pre']:.1f}% of the pre-release average; "
            f"95% CI {metrics['level_ci_lo']:.0f}…{metrics['level_ci_hi']:.0f})."
        )
        st.write(
            f"{flag(p_timepost)} After that date, the weekly trend changed by "
            f"**{metrics['slope_change_per_week']:.3f}** per week "
            f"(95% CI {metrics['slope_ci_lo']:.3f}…{metrics['slope_ci_hi']:.3f})."
        )
        st.caption("Green check = statistically significant at p<0.05; warning = not significant.")

        # ---- Core charts ----
        st.pyplot(figs[0]); st.pyplot(figs[1])

        # ---- Coeff table ----
        coef = (model.params.to_frame("coef")
                .join(model.bse.to_frame("stderr"))
                .join(model.pvalues.to_frame("pval")))
        st.subheader("Model coefficients")
        st.dataframe(coef.style.format({"coef": "{:.4f}", "stderr": "{:.4f}", "pval": "{:.4f}"}))

        # ---- Downloads (CSV/ZIP/PPT) ----
        merged = df.copy()
        merged.index.name = "week_start"
        csv_coef = coef.to_csv().encode()
        csv_data = merged.to_csv().encode()
        st.download_button("⬇️ Download coefficients (CSV)", csv_coef, file_name="its_coefficients.csv")
        st.download_button("⬇️ Download weekly series (CSV)", csv_data, file_name="weekly_series.csv")

        # ZIP bundle (figures + CSVs + summary)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, fig in enumerate(figs, start=1):
                fbuf = io.BytesIO()
                fig.savefig(fbuf, format="png", dpi=200, bbox_inches="tight")
                z.writestr(f"figure_{i}.png", fbuf.getvalue())
            z.writestr("its_coefficients.csv", csv_coef)
            z.writestr("weekly_series.csv", csv_data)
            summary = (
                f"Film: {film_input}\nOutcome: {outcome_input}\n"
                f"Intervention: {intervention}\n\n"
                f"Immediate level change: {metrics['level_change']:.2f} "
                f"({metrics['level_change_pct_of_pre']:.1f}% of pre)\n"
                f"Slope change per week: {metrics['slope_change_per_week']:.3f}\n"
                f"95% CIs — level: {metrics['level_ci_lo']:.2f}..{metrics['level_ci_hi']:.2f}, "
                f"slope: {metrics['slope_ci_lo']:.3f}..{metrics['slope_ci_hi']:.3f}\n"
                f"Source: {'Google Trends' if 'Google' in data_source else 'Wikipedia'}"
            ).encode()
            z.writestr("summary.txt", summary)
        st.download_button("⬇️ Download report bundle (ZIP)", buf.getvalue(), file_name="its_report_bundle.zip")

        # PowerPoint
        def _make_ppt():
            return make_ppt(film_input, outcome_input, intervention, figs, metrics)
        st.download_button("⬇️ Download PowerPoint (.pptx)", _make_ppt(), file_name="its_report.pptx")

        # ---- Performance Visuals Pack ----
        st.subheader("Performance visuals")
        # Rebuild design matrices
        design = sm.add_constant(df[["time","post","time_post","film_lag1"]])
        pred    = model.predict(design)
        design_cf = design.copy()
        design_cf["post"] = 0
        design_cf["time_post"] = 0
        pred_cf = model.predict(design_cf)  # counterfactual
        after_mask = df.index >= metrics["t0_monday"]
        excess = (df["outcome"] - pred_cf).where(after_mask, 0.0)

        # 1) Actual vs Counterfactual (shaded excess)
        fig = plt.figure(figsize=(10,4))
        plt.plot(df.index, df["outcome"], label="Actual outcome")
        plt.plot(df.index, pred_cf, label="Counterfactual (no intervention)")
        plt.fill_between(df.index[after_mask], pred_cf[after_mask], df["outcome"][after_mask], alpha=0.25)
        plt.axvline(metrics["t0_monday"], linestyle="--", label="Intervention")
        plt.title("Actual vs Counterfactual"); plt.legend(); plt.tight_layout()
        st.pyplot(fig)

        # 2) Weekly excess bars
        fig = plt.figure(figsize=(10,3))
        plt.bar(df.index[after_mask], excess[after_mask])
        plt.axvline(metrics["t0_monday"], linestyle="--")
        plt.title("Weekly Excess vs Counterfactual"); plt.tight_layout()
        st.pyplot(fig)

        # 3) Cumulative excess
        cum_excess = excess.cumsum()
        fig = plt.figure(figsize=(10,3))
        plt.plot(df.index, cum_excess)
        plt.axvline(metrics["t0_monday"], linestyle="--")
        plt.title("Cumulative Excess Interest"); plt.tight_layout()
        st.pyplot(fig)

        # 4) Event-study (±26 weeks)
        weeks_from = ((df.index - metrics["t0_monday"]).days // 7).astype(int)
        window = weeks_from.between(-26, 26)
        es = df.loc[window, ["outcome"]].copy()
        es["k"] = weeks_from[window]
        es_avg = es.groupby("k")["outcome"].mean()
        fig = plt.figure(figsize=(10,3))
        plt.plot(es_avg.index, es_avg.values)
        plt.axvline(0, linestyle="--")
        plt.title("Event-time average (weeks relative to intervention)")
        plt.xlabel("Weeks from intervention"); plt.tight_layout()
        st.pyplot(fig)

        # 5) Decay fit & half-life (fit simple exponential to positive excess after event)
        from scipy.optimize import curve_fit
        def _exp_decay(t, A, k, c):  # A*e^(-k t)+c
            return A * np.exp(-k * t) + c
        pos = after_mask & (excess > 0)
        if pos.sum() >= 6:
            y = excess[pos].values
            t = np.arange(len(y))
            try:
                A0, k0, c0 = max(y), 0.1, 0.0
                (A, k, c), _ = curve_fit(_exp_decay, t, y, p0=[A0, k0, c0], maxfev=20000)
                hl = (np.log(2)/k) if k > 0 else np.nan
                fig = plt.figure(figsize=(10,3))
                plt.plot(df.index[pos], y, label="Observed excess")
                plt.plot(df.index[pos], _exp_decay(t, A, k, c), label=f"Decay fit (half-life ≈ {hl:.1f} wks)")
                plt.axvline(metrics["t0_monday"], linestyle="--")
                plt.title("Excess decay after intervention"); plt.legend(); plt.tight_layout()
                st.pyplot(fig)
            except Exception as _e:
                st.info(f"Decay fit skipped: {_e}")

        # 6) Placebo check (random pre-dates vs observed effect)
        def _fit_at(date_monday):
            d = df.copy()
            d["post"] = (d.index >= date_monday).astype(int)
            d["time_post"] = d["time"] * d["post"]
            Xp = sm.add_constant(d[["time","post","time_post","film_lag1"]])
            return sm.OLS(d["outcome"], Xp).fit(cov_type="HAC", cov_kwds={"maxlags":4})

        pre_idx = df.index[df.index < metrics["t0_monday"]]
        pre_idx = pre_idx[8:]  # buffer
        n_placebo = int(min(50, max(0, len(pre_idx)-8)))
        placebo_effects = []
        if n_placebo >= 10:
            rng = np.random.default_rng(42)
            picks = np.sort(rng.choice(pre_idx, size=n_placebo, replace=False))
            for d0 in picks:
                res = _fit_at(d0)
                if "post" in res.params:
                    placebo_effects.append(float(res.params["post"]))
            if placebo_effects:
                fig = plt.figure(figsize=(10,3))
                plt.hist(placebo_effects, bins=20, alpha=0.8)
                plt.axvline(metrics["level_change"], color="red", linestyle="--", label="Observed level change")
                plt.title("Placebo distribution (pre-period dates)"); plt.legend(); plt.tight_layout()
                st.pyplot(fig)

        # 7) Interest by region (Trends only)
        if show_regions and "Google" in data_source:
            try:
                st.subheader("Interest by region (Top 15)")
                reg = cached_regions(film_input.split(",")[0].strip(), geo=geo, timeframe=timeframe)
                qcol = film_input.split(",")[0].strip()
                top = reg.sort_values(qcol, ascending=False).head(15)
                fig = plt.figure(figsize=(10,5))
                plt.barh(top["region"][::-1], top[qcol][::-1])
                plt.title("Where interest is highest"); plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Region breakdown unavailable: {e}")

        # ---- Optional: compare multiple dates ----
        if multi_dates.strip():
            dates = [d.strip() for d in multi_dates.split(",") if d.strip()]
            try:
                table = compare_interventions(dates, out_s, film_s)
                st.subheader("Date comparison")
                st.dataframe(table.style.format({
                    "level_change":"{:.0f}", "level_%_pre":"{:.1f}%", "level_p":"{:.3f}",
                    "slope_change":"{:.2f}", "slope_p":"{:.3f}"
                }))
                st.download_button("⬇️ Download comparison (CSV)", table.to_csv(index=False).encode(),
                                   file_name="its_date_comparison.csv")
            except Exception as e:
                st.warning(f"Could not run date comparison: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# --------------------------- Batch mode ---------------------------
if batch_file is not None:
    st.subheader("Batch results")
    try:
        bdf = pd.read_csv(batch_file)
        rows = []
        for _, r in bdf.iterrows():
            src = str(r.get("source","trends") or "trends").lower()
            geo_r = str(r.get("geo",""))
            try:
                if "trend" in src:
                    f = cached_trends([r["film"]], geo=geo_r, timeframe="today 5-y").rename("film_interest")
                    o = cached_trends([r["outcome"]], geo=geo_r, timeframe="today 5-y").rename("outcome")
                else:
                    f = cached_wiki(r["film"], "20190101", dt.date.today().strftime("%Y%m%d")).rename("film_interest")
                    o = cached_wiki(r["outcome"], "20190101", dt.date.today().strftime("%Y%m%d")).rename("outcome")
                model, dfX, _, m = quick_its(o, f, str(r["intervention"]))
                rows.append({
                    "film": r["film"], "outcome": r["outcome"], "intervention": r["intervention"],
                    "source": src, "geo": geo_r,
                    "level_change": m["level_change"], "level_%_pre": m["level_change_pct_of_pre"],
                    "level_p": float(model.pvalues.get("post", np.nan)),
                    "slope_change": m["slope_change_per_week"], "slope_p": float(model.pvalues.get("time_post", np.nan)),
                })
            except Exception as e:
                rows.append({
                    "film": r.get("film"), "outcome": r.get("outcome"),
                    "intervention": r.get("intervention"), "source": src, "geo": geo_r,
                    "error": str(e)
                })
        out = pd.DataFrame(rows)
        st.dataframe(out)
        st.download_button("⬇️ Download batch results (CSV)", out.to_csv(index=False).encode(),
                           file_name="batch_its_results.csv")
    except Exception as e:
        st.error(f"Batch error: {e}")
