# app.py — Media Impacts: Quick ITS (simplified UI)
# - Clear sidebar labels & guided inputs
# - Sources: Wikipedia (recommended) or Google Trends
# - Smart fallback + combined Trends call (≤5 terms)
# - Plain-English summary + captions under visuals
# - Downloads: CSV / ZIP / PPTX
# - Date comparison & Batch mode

import io, time, zipfile, datetime as dt
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="🎬 Media Impacts — Quick ITS", layout="wide")
st.title("🎬 Media Impacts — Quick ITS")
st.caption("Enter a project title and an outcome to measure, select a release date, then run the analysis. Get charts, plain-language results, and downloadable files.")

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
    for _ in range(tries):
        try:
            py.build_payload(queries, timeframe=timeframe, geo=geo)
            df = py.interest_over_time()
            if df is None or df.empty:
                raise RuntimeError("Google Trends returned no data; try a different GEO or timeframe.")
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            s = df[queries].mean(axis=1).resample("W-MON").mean()
            return s.rename("value")
        except Exception as e:
            msg = str(e)
            if isinstance(e, pex.TooManyRequestsError) or "429" in msg:
                st.info(f"Rate limited (429). Retrying in {wait}s…")
                time.sleep(wait)
                wait = min(wait * 2, 60)
                continue
            if "code 400" in msg and timeframe != "today 5-y":
                timeframe = "today 5-y"
                continue
            raise

def wiki_weekly_pageviews(title_or_query: str, start: str, end: str, lang: str = "en") -> pd.Series:
    """
    Wikipedia weekly pageviews (sum of daily), W-MON index, with polite UA.
    Smart: if the exact page doesn't exist (404), it searches and uses the top result.
    """
    import requests
    from urllib.parse import quote

    session = requests.Session()
    session.headers.update({"User-Agent": "media-impacts-app/0.1 (contact: you@example.com)"})

    def _summary_exists(title: str) -> bool:
        r = session.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}", timeout=15)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    title = title_or_query.strip()
    if not _summary_exists(title):
        # Resolve by search if exact page not found
        sr = session.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": title_or_query, "format": "json"},
            timeout=20,
        )
        sr.raise_for_status()
        hits = sr.json().get("query", {}).get("search", [])
        if not hits:
            raise RuntimeError(f"No Wikipedia page found for “{title_or_query}”. Try a different term.")
        title = hits[0]["title"]  # top match

    # Fetch daily views and aggregate weekly
    r = session.get(
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/user/{quote(title, safe='')}/daily/{start}/{end}",
        timeout=30,
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
    s = fetch_trends_weekly(queries, geo=geo, timeframe=timeframe)
    # Guard against None or empty results so callers don’t .rename() a None
    if s is None or not isinstance(s, pd.Series) or s.empty:
        raise RuntimeError("No Google Trends data returned. Try a simpler term, different GEO, or switch to Wikipedia.")
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wiki(title, start, end, lang="en"):
    return wiki_weekly_pageviews(title, start, end, lang=lang)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_regions(query, geo, timeframe):
    return trends_interest_by_region(query, geo=geo, timeframe=timeframe)

# --------------------------- Extra: combined Trends call ---------------------------
def trends_pair(film_qs, outcome_qs, geo, timeframe):
    """
    Try to fetch film + outcome in ONE Trends request (<=5 total terms).
    Auto-widen: try your settings -> 5y -> worldwide. Fallback to separate cached calls.
    """
    from pytrends.request import TrendReq

    all_q = [q.strip() for q in (film_qs + outcome_qs) if q.strip()]
    py = TrendReq(hl="en-US", tz=0, retries=0, backoff_factor=0,
                  requests_args={"headers":{"User-Agent":"media-impacts-app/0.1"}})

    def _one_call(geo_val, timeframe_val):
        qs = all_q[:5]
        py.build_payload(qs, timeframe=timeframe_val, geo=geo_val)
        df = py.interest_over_time()
        if df is None or df.empty:
            return None, None
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        # Build series by averaging the columns matching each subset
        fcols = [c for c in df.columns if c in film_qs]
        ocols = [c for c in df.columns if c in outcome_qs]
        if not fcols or not ocols:
            return None, None
        film_s = df[fcols].mean(axis=1).resample("W-MON").mean().rename("film_interest")
        out_s  = df[ocols].mean(axis=1).resample("W-MON").mean().rename("outcome")
        if film_s.empty or out_s.empty:
            return None, None
        return film_s, out_s

    if len(all_q) <= 5:
        # 1) As chosen
        fs, os_ = _one_call(geo, timeframe)
        if fs is not None: return fs, os_
        # 2) Widen timeframe
        if timeframe != "today 5-y":
            fs, os_ = _one_call(geo, "today 5-y")
            if fs is not None: return fs, os_
        # 3) Try worldwide
        if geo:
            fs, os_ = _one_call("", "today 5-y")
            if fs is not None: return fs, os_

    # Fallback to separate calls (still cached/guarded)
    fs = cached_trends(film_qs, geo, timeframe).rename("film_interest")
    os_ = cached_trends(outcome_qs, geo, timeframe).rename("outcome")
    return fs, os_

# =========================== UI: Sidebar ===========================
with st.sidebar:
    st.header("Start here")

    # 1) Where to measure attention
    data_source = st.selectbox(
        "Measurement source",
        ["Wikipedia pageviews (recommended)", "Google Trends (index 0–100)"],
        index=0,
        help="Wikipedia is stable for public use. Google Trends can rate-limit but shows relative interest."
    )
    auto_fallback = st.checkbox(
        "If Google Trends fails, automatically use Wikipedia",
        value=True
    )

    # 2) What are we studying?
    film_input = st.text_input(
        "Project title (film / short / series)",
        value="Eating Our Way to Extinction",
        placeholder="e.g., Don’t Look Up"
    )

    outcome_input = st.text_input(
        "Outcome topic (what do you want to measure?)",
        value="Plant-based diet",
        placeholder="e.g., vegan diet, meat consumption, climate change"
    )

    # 3) When was it released?
    intervention = st.date_input(
        "Release window — choose the start date",
        value=dt.date(2021, 9, 30),
        help="Exact day isn’t critical. The model aligns dates to Mondays automatically."
    )

    # 4) Advanced (optional)
    with st.expander("Advanced options (optional)", expanded=False):
        # --- Google Trends specific ---
        if "Google Trends" in data_source:
            st.caption("Google Trends settings")
            colA, colB = st.columns(2)
            with colA:
                geo = st.selectbox("Region", ["", "US", "GB", "TH"], index=1, help="Blank = worldwide")
            with colB:
                timeframe = st.selectbox("Timeframe", ["today 5-y", "today 12-m", "today 3-m"], index=0)
            show_regions = st.checkbox("Show interest by region map/table", value=False)
        else:
            # Defaults so the rest of the app works even if user never opens Advanced
            geo = ""
            timeframe = "today 5-y"
            show_regions = False

        # --- Wikipedia specific ---
        st.caption("Wikipedia pageviews settings")
        colC, colD, colE = st.columns(3)
        with colC:
            start_daily = st.text_input("Start (YYYYMMDD)", "20190101")
        with colD:
            end_daily = st.text_input("End (YYYYMMDD)", dt.date.today().strftime("%Y%m%d"))
        with colE:
            lang = st.text_input("Language code", "en")

        # --- Extras ---
        multi_dates = st.text_input(
            "Compare several dates (comma-separated, optional)",
            "2021-09-30, 2022-07-01",
            help="Useful if you want to test theatrical vs streaming vs YouTube dates."
        )

        st.markdown("---")
        st.caption("Batch mode CSV: film,outcome,intervention,source,geo")
        batch_file = st.file_uploader("Upload batch CSV (optional)", type=["csv"])

    run = st.button("▶️ Run analysis", use_container_width=True)

# =========================== RIGHT: Results ===========================
right = st.container()

if run:
    try:
        with right, st.spinner("Fetching data and running ITS…"):
            if "Google" in data_source:
                film_qs = [q.strip() for q in film_input.split(",")]
                outcome_qs = [q.strip() for q in outcome_input.split(",")]
                try:
                    film_s, out_s = trends_pair(film_qs, outcome_qs, geo, timeframe)
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

            model, df, figs, metrics = quick_its(out_s, film_s, intervention.isoformat())

            # ---- Headline summary ----
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

            # ---- Layman-friendly narrative ----
            with st.expander("What these charts mean (plain language)", expanded=True):
                design = sm.add_constant(df[["time","post","time_post","film_lag1"]])
                pred    = model.predict(design)
                design_cf = design.copy()
                design_cf["post"] = 0
                design_cf["time_post"] = 0
                pred_cf = model.predict(design_cf)

                after = df.index >= metrics["t0_monday"]
                excess = (df["outcome"] - pred_cf).where(after, 0.0)
                cum_excess = float(excess.cumsum().iloc[-1])
                weeks_positive = int((excess > 0).sum())

                peak_val = float(df["outcome"].max())
                peak_when = df["outcome"].idxmax().date()

                # Half-life (best-effort)
                half_life_text = "n/a"
                try:
                    from scipy.optimize import curve_fit
                    def _exp_decay(t, A, k, c): return A*np.exp(-k*t) + c
                    pos = after & (excess > 0)
                    if pos.sum() >= 6:
                        y = excess[pos].values
                        t = np.arange(len(y))
                        (A, k, c), _ = curve_fit(_exp_decay, t, y, p0=[max(y), 0.1, 0.0], maxfev=20000)
                        hl = (np.log(2)/k) if k > 0 else np.nan
                        if np.isfinite(hl):
                            half_life_text = f"{hl:.1f} weeks"
                except Exception:
                    pass

                # Optional: top regions (only for Trends)
                regions_text = ""
                if "Google" in data_source and 'show_regions' in locals() and show_regions:
                    try:
                        reg = cached_regions(film_input.split(",")[0].strip(), geo=geo, timeframe=timeframe)
                        qcol = film_input.split(",")[0].strip()
                        top3 = reg.sort_values(qcol, ascending=False).head(3)["region"].tolist()
                        if top3:
                            regions_text = f" Top regions: {', '.join(top3)}."
                    except Exception:
                        pass

                sig = lambda p: "statistically significant" if p < 0.05 else "not statistically significant"
                readable = (
                    f"**Headline** — Around **{intervention}**, attention to **{outcome_input}** changed.\n\n"
                    f"**First chart:** Blue = weekly interest; dashed line = release; orange = fitted model.\n\n"
                    f"**Immediate jump:** **{metrics['level_change']:.0f}** "
                    f"({metrics['level_change_pct_of_pre']:.1f}% of pre-release), {sig(p_post)}.\n\n"
                    f"**Trend change:** **{metrics['slope_change_per_week']:.3f}** per week, {sig(p_timepost)}.\n\n"
                    f"**Cumulative lift:** ~**{cum_excess:,.0f}** units over **{weeks_positive} weeks** vs the baseline path.\n\n"
                    f"**Peak:** **{peak_val:,.0f}** on **{peak_when}**. **Half-life:** {half_life_text}.\n\n"
                    f"**Data source:** {'Google Trends (0–100 index)' if 'Google' in data_source else 'Wikipedia pageviews (counts)'}."
                    f"{regions_text}\n\n*Correlations only; other events may also drive interest.*"
                )
                st.markdown(readable)
                st.download_button("⬇️ Download summary (.txt)", readable.encode("utf-8"), file_name="readable_summary.txt")

            # ---- Coeff table ----
            coef = (model.params.to_frame("coef")
                    .join(model.bse.to_frame("stderr"))
                    .join(model.pvalues.to_frame("pval")))
            st.subheader("Model coefficients")
            st.dataframe(coef.style.format({"coef": "{:.4f}", "stderr": "{:.4f}", "pval": "{:.4f}"}))

            # ---- Downloads (CSV/ZIP/PPT) ----
            merged = df.copy()
            merged.index.name = "week_start"
            st.download_button("⬇️ Coefficients (CSV)", coef.to_csv().encode(), file_name="its_coefficients.csv")
            st.download_button("⬇️ Weekly series (CSV)", merged.to_csv().encode(), file_name="weekly_series.csv")

            # ZIP bundle (figures + CSVs + summary)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for i, fig in enumerate(figs, start=1):
                    fbuf = io.BytesIO()
                    fig.savefig(fbuf, format="png", dpi=200, bbox_inches="tight")
                    z.writestr(f"figure_{i}.png", fbuf.getvalue())
                z.writestr("its_coefficients.csv", coef.to_csv().encode())
                z.writestr("weekly_series.csv", merged.to_csv().encode())
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
            st.download_button("⬇️ Report bundle (ZIP)", buf.getvalue(), file_name="its_report_bundle.zip")

            # PowerPoint
            st.download_button("⬇️ PowerPoint (.pptx)", make_ppt(film_input, outcome_input, intervention, figs, metrics),
                               file_name="its_report.pptx")

            # ---- Performance Visuals (with captions) ----
            st.subheader("Performance visuals")

            # Rebuild design matrices for predictions
            design = sm.add_constant(df[["time","post","time_post","film_lag1"]])
            pred    = model.predict(design)
            design_cf = design.copy()
            design_cf["post"] = 0
            design_cf["time_post"] = 0
            pred_cf = model.predict(design_cf)  # counterfactual (no intervention)

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
            st.caption(
                "Blue is what happened; orange is a modelled ‘no-release’ world. "
                "Shaded area after the dashed line is **extra attention** attributed to the release."
            )

            # 2) Weekly excess bars
            fig = plt.figure(figsize=(10,3))
            plt.bar(df.index[after_mask], excess[after_mask])
            plt.axvline(metrics["t0_monday"], linestyle="--")
            plt.title("Weekly Excess vs Counterfactual"); plt.tight_layout()
            st.pyplot(fig)
            st.caption("Each bar shows how far above (or below) baseline the outcome was that week after release.")

            # 3) Cumulative excess
            cum_excess = excess.cumsum()
            fig = plt.figure(figsize=(10,3))
            plt.plot(df.index, cum_excess)
            plt.axvline(metrics["t0_monday"], linestyle="--")
            plt.title("Cumulative Excess Interest"); plt.tight_layout()
            st.pyplot(fig)
            st.caption("Running total of extra attention vs the counterfactual. Steeper climb = faster gains.")

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
            st.caption("Lead-up (left) and after-effect (right) centered on the selected week (0).")

            # 5) Decay fit & half-life (best-effort)
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
                    st.caption("How fast the lift fades. Half-life ≈ weeks for the lift to drop by half.")
                except Exception as _e:
                    st.info(f"Decay fit skipped: {_e}")
                    st.caption("We couldn’t fit a decay curve this time (not enough stable positive weeks).")

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
                    st.caption(
                        "Grey bars = fake pre-dates. Red line = your jump. If it’s far to the right, "
                        "your effect is unlikely to be just random timing."
                    )

            # 7) Interest by region (Trends only)
            if "Google" in data_source and show_regions:
                try:
                    st.subheader("Interest by region (Top 15)")
                    reg = cached_regions(film_input.split(",")[0].strip(), geo=geo, timeframe=timeframe)
                    qcol = film_input.split(",")[0].strip()
                    top = reg.sort_values(qcol, ascending=False).head(15)
                    fig = plt.figure(figsize=(10,5))
                    plt.barh(top["region"][::-1], top[qcol][::-1])
                    plt.title("Where interest is highest"); plt.tight_layout()
                    st.pyplot(fig)
                    st.caption("Locations with the strongest search interest for the project (Google Trends).")
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
        with right:
            st.error(f"Error: {e}")
        st.stop()

# --------------------------- Batch mode ---------------------------
if 'batch_file' in locals() and batch_file is not None:
    with right:
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
          
