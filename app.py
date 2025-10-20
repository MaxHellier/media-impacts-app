# app.py — Media Impact Analyzer (clarified charts + optional AI explainer)
# Python 3.13 compatible. Adds:
# - Axis labels & "What this shows" captions for each chart
# - Axes key card (Y depends on source: Wikipedia vs Google Trends)
# - Optional AI explanation using OPENAI_API_KEY in Streamlit Secrets (or env)

import io, time, datetime as dt, zipfile, json, os
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
import requests
from urllib.parse import quote

# --------------------------- Page setup & CSS ---------------------------
st.set_page_config(page_title="🎬 Media Impact Analyzer", layout="wide")
st.markdown(
    """
    <style>
      .center-card {max-width: 920px; margin: 0 auto;}
      .stButton>button {width: 100%;}
      .muted {color: #9aa0a6;}
      .callout {background:#f6f8fa; padding:12px 14px; border-radius:8px; border:1px solid #e5e7eb;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Media Impact Analyzer — Google Trends & Wikipedia")
st.caption(
    "Fetches weekly public-interest signals from **Google Trends** or the **Wikimedia Pageviews API** for your "
    "project title and chosen outcome, then fits an **Interrupted Time Series** at your release date "
    "(level jump + post-release slope change)."
)

# --------------------------- Helpers ---------------------------
def _fmt_int(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)

def _fmt_float(x, d=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

def _y_label(source_name: str) -> str:
    if "Wikipedia" in source_name:
        return "Weekly Wikipedia pageviews"
    return "Weekly Google Trends interest (index 0–100)"

def _units(source_name: str) -> str:
    return "pageviews" if "Wikipedia" in source_name else "index points"

# =====================================================
#                Data fetchers (robust)
# =====================================================
def fetch_trends_weekly(queries: List[str], geo: str, timeframe: str = "today 5-y",
                        hl: str = "en-US", tz: int = 0, tries: int = 6) -> pd.Series:
    """Google Trends weekly series (W-MON), averaged across up to 5 queries; polite retries for 429s."""
    from pytrends.request import TrendReq
    from pytrends import exceptions as pex

    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise ValueError("Please provide at least one query.")
    queries = queries[:5]

    py = TrendReq(hl=hl, tz=tz, retries=0, backoff_factor=0, timeout=(10, 30),
                  requests_args={"headers": {"User-Agent": "media-impacts-app/0.4"}})

    wait = 5
    last_err = None
    for attempt in range(tries):
        try:
            py.build_payload(queries, timeframe=timeframe, geo=geo)
            df = py.interest_over_time()
            if df is None or df.empty:
                raise RuntimeError("Google Trends returned no data")
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            s = df[queries].mean(axis=1).resample("W-MON").mean()
            if s is None or s.empty:
                raise RuntimeError("Google Trends series empty after resample")
            return s.rename("value")
        except Exception as e:
            last_err = e
            msg = str(e)
            if isinstance(e, pex.TooManyRequestsError) or "429" in msg:
                st.info(f"Google Trends rate limited. Retrying in {wait}s… (try {attempt+1}/{tries})")
                time.sleep(wait)
                wait = min(wait * 2, 60)
                continue
            if "code 400" in msg and timeframe != "today 5-y":
                timeframe = "today 5-y"
                continue
            if any(k in msg for k in ["502", "503", "504", "timeout"]):
                time.sleep(3)
                continue
            break
    raise RuntimeError(f"Google Trends failed after retries: {last_err}")

def wiki_weekly_pageviews(title_or_query: str, start: str, end: str, lang: str = "en") -> pd.Series:
    """Wikipedia weekly pageviews (sum of daily), W-MON index, robust to 404 + search fallback."""
    session = requests.Session()
    session.headers.update({"User-Agent": "media-impacts-app/0.4 (contact: you@example.com)"})

    def _summary_exists(title: str) -> bool:
        r = session.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}", timeout=15)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    title = title_or_query.strip()
    try:
        if not _summary_exists(title):
            sr = session.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": title_or_query, "format": "json"},
                timeout=20,
            )
            sr.raise_for_status()
            hits = sr.json().get("query", {}).get("search", [])
            if not hits:
                raise RuntimeError(f"No Wikipedia page found for “{title_or_query}”.")
            title = hits[0]["title"]

        r = session.get(
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{lang}.wikipedia/all-access/user/{quote(title, safe='')}/daily/{start}/{end}",
            timeout=30,
        )
        r.raise_for_status()
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "")
        raise RuntimeError(f"Wikipedia API error {code}. Please rerun or try again later.") from e

    items = r.json().get("items", [])
    if not items:
        raise RuntimeError(f"No pageview data for “{title}”.")
    idx = [pd.to_datetime(it["timestamp"][:8]) for it in items]
    vals = [it["views"] for it in items]
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=title).resample("W-MON").sum()
    if s.empty:
        raise RuntimeError(f"Wikipedia returned no data after resample for “{title}”.")
    return s

# =====================================================
#                    Modeling (ITS)
# =====================================================
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

    # Figures (now with axis labels)
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["outcome"], label="Outcome interest")
    plt.axvline(t0_monday, linestyle="--", label=f"Release: {t0.date()}")
    plt.plot(df.index, model.predict(X), label="Fitted (segmented regression)")
    plt.title("Outcome vs release — quick ITS")
    plt.xlabel("Week (starts Monday)")
    # y-label filled later once we know source_name
    plt.legend(); plt.tight_layout()

    fig2 = plt.figure(figsize=(10, 2.8))
    plt.plot(df.index, df["film_interest"], label="Project interest")
    plt.axvline(t0_monday, linestyle="--")
    plt.title("Project interest")
    plt.xlabel("Week (starts Monday)")
    # y-label filled later too
    plt.tight_layout()

    pre_mean = df.loc[df.index < t0_monday, "outcome"].mean()
    ci = model.conf_int()
    level = float(model.params.get("post", np.nan))
    slope = float(model.params.get("time_post", np.nan))
    level_lo, level_hi = [float(x) for x in ci.loc["post"]] if "post" in ci.index else (np.nan, np.nan)
    slope_lo, slope_hi = [float(x) for x in ci.loc["time_post"]] if "time_post" in ci.index else (np.nan, np.nan)

    metrics = {
        "level_change": level,
        "level_change_pct_of_pre": float(100 * level / pre_mean) if pre_mean else np.nan,
        "level_ci_lo": level_lo, "level_ci_hi": level_hi,
        "slope_change_per_week": slope,
        "slope_ci_lo": slope_lo, "slope_ci_hi": slope_hi,
        "t0_monday": t0_monday, "pre_mean": float(pre_mean) if pre_mean else np.nan
    }
    return model, df, (fig1, fig2), metrics

def trends_interest_by_region(query: str, geo: str = "", timeframe: str = "today 5-y") -> pd.DataFrame:
    from pytrends.request import TrendReq
    py = TrendReq(hl="en-US", tz=0, retries=0, backoff_factor=0,
                  requests_args={"headers": {"User-Agent": "media-impacts-app/0.4"}})
    py.build_payload([query], timeframe=timeframe, geo=geo)
    df = py.interest_by_region(resolution="COUNTRY" if geo == "" else "REGION",
                               inc_low_vol=True, inc_geo_code=True).reset_index()
    name_col = "geoName" if "geoName" in df.columns else df.columns[0]
    out = df.rename(columns={name_col: "region"})
    return out.sort_values(query, ascending=False).reset_index(drop=True)

# =====================================================
#                     Cache wrappers
# =====================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_trends(queries, geo, timeframe):
    s = fetch_trends_weekly(queries, geo=geo, timeframe=timeframe)
    if s is None or not isinstance(s, pd.Series) or s.empty:
        raise RuntimeError("No Google Trends data returned. Try simpler terms, different GEO, or Wikipedia.")
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wiki(title, start, end, lang="en"):
    return wiki_weekly_pageviews(title, start, end, lang=lang)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_regions(query, geo, timeframe):
    return trends_interest_by_region(query, geo=geo, timeframe=timeframe)

# =====================================================
#            Combined Trends call (pair)
# =====================================================
def trends_pair(film_qs, outcome_qs, geo, timeframe):
    """Try film+outcome in one Trends request (≤5 total). Fallback to separate calls."""
    from pytrends.request import TrendReq
    all_q = [q.strip() for q in (film_qs + outcome_qs) if q.strip()]
    py = TrendReq(hl="en-US", tz=0, retries=0, backoff_factor=0,
                  requests_args={"headers": {"User-Agent": "media-impacts-app/0.4"}})

    def _one_call(geo_val, timeframe_val):
        qs = all_q[:5]
        py.build_payload(qs, timeframe=timeframe_val, geo=geo_val)
        df = py.interest_over_time()
        if df is None or df.empty:
            return None, None
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
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
        fs, os_ = _one_call(geo, timeframe)
        if fs is not None: return fs, os_
        if timeframe != "today 5-y":
            fs, os_ = _one_call(geo, "today 5-y")
            if fs is not None: return fs, os_
        if geo:
            fs, os_ = _one_call("", "today 5-y")
            if fs is not None: return fs, os_
    fs = cached_trends(film_qs, geo, timeframe).rename("film_interest")
    os_ = cached_trends(outcome_qs, geo, timeframe).rename("outcome")
    return fs, os_

# =====================================================
#                Optional AI explainer
# =====================================================
def ai_explain(context: dict) -> str:
    """
    Uses OpenAI Chat Completions if OPENAI_API_KEY is present in Streamlit Secrets or env.
    Returns plain text, or a helpful error if not configured.
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ("AI explanation unavailable: add OPENAI_API_KEY to Streamlit Secrets (or environment). "
                "Settings → Secrets → add OPENAI_API_KEY.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        sys_msg = (
            "You are a data journalist. Explain statistical results simply and accurately for a general audience. "
            "Avoid jargon. Always define what the axes mean. Be concise and concrete."
        )
        user_msg = (
            "Write a short, plain-English summary of a media release impact analysis. "
            "Use 5 bullets max: Overview, Immediate jump, Trend after release, Cumulative lift/half-life, Caveats. "
            "Define X and Y axes for the main charts. Here is the context JSON:\n\n"
            f"{json.dumps(context, default=str)}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI explanation error: {e}"

# =====================================================
#        Inputs (centered) and run controls
# =====================================================
st.session_state.setdefault("force_wiki", False)

st.markdown('<div class="center-card">', unsafe_allow_html=True)
with st.form("center_form"):
    film_input = st.text_input(
        "Project title (e.g., Eating Our Way to Extinction)",
        value="", placeholder="Type a film/series title…",
    )
    intervention = st.date_input(
        "Date of release",
        value=dt.date(2021, 9, 30),
        help="Approximate is OK. Weeks are aligned to Mondays."
    )

    with st.expander("Advanced options", expanded=False):
        st.caption("What should we measure?")
        outcome_input = st.text_input(
            "Outcome topic (default shown)",
            value="Plant-based diet",
            placeholder="e.g., vegan diet, meat consumption, food waste",
            help="This is the outcome/idea/behavior you want to track."
        )

        st.caption("Where should we measure attention?")
        data_source = st.selectbox(
            "Measurement source",
            ["Wikipedia pageviews (recommended)", "Google Trends (index 0–100)"],
            index=0,
            help="Wikipedia is stable; Google Trends can rate-limit but shows relative interest."
        )
        auto_fallback = st.checkbox("If Google Trends fails, automatically use Wikipedia", value=True)

        if "Google" in data_source:
            colA, colB = st.columns(2)
            with colA:
                geo = st.selectbox("Region", ["", "US", "GB", "TH"], index=1, help="Blank = worldwide")
            with colB:
                timeframe = st.selectbox("Timeframe", ["today 5-y", "today 12-m", "today 3-m"], index=0)
            show_regions = st.checkbox("Show interest by region (Trends only)", value=False)
        else:
            geo = ""; timeframe = "today 5-y"; show_regions = False

        st.caption("Wikipedia settings")
        colC, colD, colE = st.columns(3)
        with colC: start_daily = st.text_input("Start (YYYYMMDD)", "20190101")
        with colD: end_daily   = st.text_input("End (YYYYMMDD)", dt.date.today().strftime("%Y%m%d"))
        with colE: lang        = st.text_input("Language", "en")

        st.caption("Extras")
        multi_dates = st.text_input("Compare several dates (comma-separated, optional)", "2021-09-30, 2022-07-01")

    run = st.form_submit_button("▶️ Run analysis")
st.markdown("</div>", unsafe_allow_html=True)

colx, coly, colz = st.columns([1,1,1])
with colx:
    if st.button("🔄 Retry now"): st.rerun()
with coly:
    if st.button("🧭 Switch to Wikipedia and retry"):
        st.session_state["force_wiki"] = True
        st.rerun()
with colz:
    ai_on = st.checkbox("✨ Enable AI explanation (if API key set)", value=True)

# =====================================================
#                        RUN
# =====================================================
if run:
    try:
        with st.spinner("Fetching data and running ITS…"):
            use_wiki = ("Wikipedia" in data_source) or st.session_state.get("force_wiki", False)
            source_name = "Wikipedia pageviews" if use_wiki else "Google Trends index (0–100)"
            y_label = _y_label(source_name)
            units = _units(source_name)

            if not use_wiki:
                film_qs = [q.strip() for q in (film_input or "").split(",")]
                outcome_qs = [q.strip() for q in (outcome_input or "").split(",")]
                try:
                    film_s, out_s = trends_pair(film_qs, outcome_qs, geo, timeframe)
                except Exception as e:
                    if auto_fallback:
                        st.warning(f"Google Trends failed ({e}). Falling back to Wikipedia pageviews.")
                        film_s = cached_wiki(film_input, start_daily, end_daily, lang=lang).rename("film_interest")
                        out_s  = cached_wiki(outcome_input, start_daily, end_daily, lang=lang).rename("outcome")
                        source_name, y_label, units = "Wikipedia pageviews", _y_label("Wikipedia"), "pageviews"
                    else:
                        raise
            else:
                film_s = cached_wiki(film_input, start_daily, end_daily, lang=lang).rename("film_interest")
                out_s  = cached_wiki(outcome_input, start_daily, end_daily, lang=lang).rename("outcome")

            # Model
            model, df, figs, metrics = quick_its(out_s, film_s, intervention.isoformat())

            # Fill Y labels on first two figs now that we know the source
            for f in figs:
                ax = f.gca()
                ax.set_ylabel(y_label)

            # ---------------- Results (clear wording) ----------------
            st.subheader("Results")

            sig_level = 0.05
            p_post     = float(model.pvalues.get("post", 1.0))
            p_timepost = float(model.pvalues.get("time_post", 1.0))
            pre_avg    = metrics.get("pre_mean", np.nan)

            pct = np.nan
            if pre_avg not in (None, 0) and not (isinstance(pre_avg, float) and np.isnan(pre_avg)):
                try: pct = 100.0 * metrics["level_change"] / pre_avg
                except Exception: pct = np.nan

            level_line = (
                f"- **Immediate lift at release:** {_fmt_int(metrics['level_change'])} {units} "
                f"(95% CI {_fmt_int(metrics['level_ci_lo'])}…{_fmt_int(metrics['level_ci_hi'])}). "
                f"{'This change is statistically significant.' if p_post < sig_level else 'This change is not statistically significant.'}"
            )
            context_line = (
                f"- **Context:** about **{_fmt_float(pct, 1)}%** of a typical pre-release week."
                if not np.isnan(pct) else
                "- **Context:** couldn’t compute % of pre-release (too little pre data)."
            )
            slope_val = metrics["slope_change_per_week"]
            slope_dir = "increased" if slope_val > 0 else "decreased" if slope_val < 0 else "did not change"
            slope_line = (
                f"- **After release, the weekly trend {slope_dir} by** {_fmt_float(slope_val, 3)} {units} per week "
                f"(95% CI {_fmt_float(metrics['slope_ci_lo'], 3)}…{_fmt_float(metrics['slope_ci_hi'], 3)}). "
                f"{'Significant.' if p_timepost < sig_level else 'Not significant.'}"
            )

            st.markdown(
                f"""
**At the release week ({intervention}):**  
{level_line}  
{context_line}

**After the release:**  
{slope_line}
"""
            )
            st.caption("Notes: “Significant” means p<0.05. Positive numbers = more attention; negative = less.")

            # Axes key card
            st.markdown(
                f"""
<div class="callout">
<b>Axes key</b><br>
• X-axis = week (starts Monday).<br>
• Y-axis = <b>{y_label}</b>.<br>
• Source = <b>{source_name}</b>.
</div>
""", unsafe_allow_html=True)

            # Core charts with captions
            st.pyplot(figs[0])
            st.caption(f"Shows the outcome over time, the fitted ITS line, and the release week (dashed). "
                       f"Y = {y_label}. A higher value = more public attention.")

            st.pyplot(figs[1])
            st.caption(f"Project interest over time (proxy for media exposure). Y = {y_label}.")

            # Narrative explainer & computed extras
            design = sm.add_constant(df[["time","post","time_post","film_lag1"]])
            pred    = model.predict(design)
            cf = design.copy(); cf["post"]=0; cf["time_post"]=0
            pred_cf = model.predict(cf)
            after = df.index >= metrics["t0_monday"]
            excess = (df["outcome"] - pred_cf).where(after, 0.0)
            cum_excess = float(excess.cumsum().iloc[-1])
            weeks_positive = int((excess > 0).sum())
            peak_val = float(df["outcome"].max()); peak_when = df["outcome"].idxmax().date()

            # Half-life (best-effort)
            half_life_text = "n/a"
            try:
                from scipy.optimize import curve_fit
                def _exp_decay(t, A, k, c): return A*np.exp(-k*t) + c
                pos = after & (excess > 0)
                if pos.sum() >= 6:
                    y = excess[pos].values; t = np.arange(len(y))
                    (A, k, c), _ = curve_fit(_exp_decay, t, y, p0=[max(y),0.1,0.0], maxfev=20000)
                    hl = (np.log(2)/k) if k>0 else np.nan
                    if np.isfinite(hl): half_life_text = f"{hl:.1f} weeks"
            except Exception:
                pass

            # Optional AI explainer
            if ai_on:
                with st.expander("✨ AI explanation", expanded=False):
                    ctx = {
                        "project_title": film_input,
                        "outcome_topic": outcome_input,
                        "release_date": str(intervention),
                        "source": source_name,
                        "y_axis": y_label,
                        "level_change": metrics["level_change"],
                        "level_ci": [metrics["level_ci_lo"], metrics["level_ci_hi"]],
                        "level_change_pct_of_pre": None if np.isnan(pct) else float(round(pct,1)),
                        "slope_change_per_week": metrics["slope_change_per_week"],
                        "slope_ci": [metrics["slope_ci_lo"], metrics["slope_ci_hi"]],
                        "p_level": float(model.pvalues.get("post", np.nan)),
                        "p_slope": float(model.pvalues.get("time_post", np.nan)),
                        "cumulative_excess": cum_excess,
                        "weeks_positive_excess": weeks_positive,
                        "peak_value": peak_val,
                        "peak_when": str(peak_when),
                        "half_life": half_life_text,
                        "pre_weeks": int((df.index < metrics["t0_monday"]).sum()),
                        "post_weeks": int((df.index >= metrics["t0_monday"]).sum())
                    }
                    if st.button("Explain these results"):
                        st.markdown(ai_explain(ctx))

            # -------- Performance visuals with labels & captions --------
            # 1) Actual vs Counterfactual
            fig = plt.figure(figsize=(10,4))
            plt.plot(df.index, df["outcome"], label="Actual outcome")
            plt.plot(df.index, pred_cf, label="Counterfactual (no release)")
            plt.fill_between(df.index[after], pred_cf[after], df["outcome"][after], alpha=0.25)
            plt.axvline(metrics["t0_monday"], linestyle="--", label="Release")
            plt.title("Actual vs Counterfactual")
            plt.xlabel("Week (starts Monday)"); plt.ylabel(y_label)
            plt.legend(); plt.tight_layout()
            st.pyplot(fig)
            st.caption("Difference between the actual series and the modelled ‘no-release’ path after the release week "
                       "(shaded area). That gap is your weekly excess.")

            # 2) Weekly excess
            fig = plt.figure(figsize=(10,3))
            plt.bar(df.index[after], excess[after])
            plt.axvline(metrics["t0_monday"], linestyle="--")
            plt.title("Weekly Excess vs Counterfactual")
            plt.xlabel("Week (starts Monday)"); plt.ylabel(f"Excess {units}/week")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("How much higher (or lower) the outcome was each week vs the counterfactual after release.")

            # 3) Cumulative excess
            fig = plt.figure(figsize=(10,3))
            plt.plot(df.index, excess.cumsum())
            plt.axvline(metrics["t0_monday"], linestyle="--")
            plt.title("Cumulative Excess Interest")
            plt.xlabel("Week (starts Monday)"); plt.ylabel(f"Cumulative excess {units}")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Running total of extra attention vs the counterfactual. Upward = accumulating attention.")

            # 4) Event-time average (±26 weeks)
            weeks_from = ((df.index - metrics["t0_monday"]).days // 7).astype(int)
            window = weeks_from.between(-26, 26)
            es = df.loc[window, ["outcome"]].copy()
            es["k"] = weeks_from[window]
            es_avg = es.groupby("k")["outcome"].mean()
            fig = plt.figure(figsize=(10,3))
            plt.plot(es_avg.index, es_avg.values); plt.axvline(0, linestyle="--")
            plt.title("Event-time average (weeks relative to release)")
            plt.xlabel("Weeks from release"); plt.ylabel(y_label)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Average pattern around the release week (0): lead-up vs after-effects.")

            # 5) Region breakdown (Trends only)
            if ("Wikipedia" not in source_name) and 'geo' in locals() and show_regions:
                try:
                    st.subheader("Interest by region (Top 15)")
                    reg = cached_regions((film_input or "").split(",")[0].strip(), geo=geo, timeframe=timeframe)
                    qcol = (film_input or "").split(",")[0].strip()
                    top = reg.sort_values(qcol, ascending=False).head(15)
                    fig = plt.figure(figsize=(10,5))
                    plt.barh(top["region"][::-1], top[qcol][::-1])
                    plt.title("Where interest is highest")
                    plt.xlabel("Relative interest (index 0–100)"); plt.ylabel("Region")
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.caption("Locations with the strongest search interest for the project (Google Trends).")
                except Exception as e:
                    st.info(f"Region breakdown unavailable: {e}")

            # ---------------- Downloads ----------------
            coef = (model.params.to_frame("coef")
                    .join(model.bse.to_frame("stderr"))
                    .join(model.pvalues.to_frame("pval")))
            st.subheader("Model coefficients")
            st.dataframe(coef.style.format({"coef":"{:.4f}","stderr":"{:.4f}","pval":"{:.4f}"}))

            merged = df.copy(); merged.index.name = "week_start"
            st.download_button("⬇️ Coefficients (CSV)", coef.to_csv().encode(), file_name="its_coefficients.csv")
            st.download_button("⬇️ Weekly series (CSV)", merged.to_csv().encode(), file_name="weekly_series.csv")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for i, figz in enumerate(figs, start=1):
                    fbuf = io.BytesIO(); figz.savefig(fbuf, format="png", dpi=200, bbox_inches="tight")
                    z.writestr(f"figure_{i}.png", fbuf.getvalue())
                z.writestr("its_coefficients.csv", coef.to_csv().encode())
                z.writestr("weekly_series.csv", merged.to_csv().encode())
            st.download_button("⬇️ Report bundle (ZIP)", buf.getvalue(), file_name="its_report_bundle.zip")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
