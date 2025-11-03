# app.py — Media Impact Analyzer (Google Trends + Wikipedia, primary/fallback, AI summary)
# Python 3.13 compatible

import io, time, datetime as dt, zipfile, json, os
from typing import List, Dict, Any, Tuple, Optional
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
      .center-card {max-width: 980px; margin: 0 auto;}
      .stButton>button {width: 100%;}
      .callout {background:#f6f8fa; padding:12px 14px; border-radius:8px; border:1px solid #e5e7eb;}
      .badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-right:6px;}
      .ok {background:#e8fbf1; color:#0a6f4b; border:1px solid #bce5cf;}
      .warn {background:#fff7e6; color:#8a5b00; border:1px solid #f2dea6;}
      .err {background:#ffe8e8; color:#8a0000; border:1px solid #f3b3b3;}
      .muted {color:#8c9196;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Media Impact Analyzer — Public-attention signals & quick ITS")
st.caption(
    "Pulls weekly public-attention signals from **Google Trends** and/or **Wikipedia Pageviews** for a project title "
    "and an outcome topic, then fits an **Interrupted Time Series** at your release date (immediate jump + post-release trend change). "
    "Charts zoom to the months around release; a plain-English summary is generated automatically."
)

# --------------------------- Defaults ---------------------------
DEFAULT_PRE_WEEKS  = 16   # show ~4 months before release in charts
DEFAULT_POST_WEEKS = 52   # show ~12 months after release in charts
DEFAULT_TREND_TIMEFRAME = "today 5-y"

# --------------------------- Helpers ---------------------------
def _fmt_int(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)

def _fmt_float(x, d=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

def _y_label(source_name: str) -> str:
    return "Weekly Wikipedia pageviews" if "Wikipedia" in source_name else "Weekly Google Trends interest (0–100)"

def _units(source_name: str) -> str:
    return "pageviews" if "Wikipedia" in source_name else "index points"

def _zoom_mask(idx: pd.DatetimeIndex, t0_monday: pd.Timestamp, pre_w: int, post_w: int):
    start = t0_monday - pd.Timedelta(weeks=pre_w)
    end   = t0_monday + pd.Timedelta(weeks=post_w)
    return (idx >= start) & (idx <= end)

# =====================================================
#                      Data fetchers
# =====================================================
def fetch_trends_weekly(queries: List[str], geo: str, timeframe: str = DEFAULT_TREND_TIMEFRAME,
                        hl: str = "en-US", tz: int = 0, tries: int = 6) -> pd.Series:
    """Google Trends weekly series (W-MON), averaged across up to 5 queries; polite retries for 429s."""
    from pytrends.request import TrendReq
    from pytrends import exceptions as pex

    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise ValueError("Please provide at least one query.")
    queries = queries[:5]

    py = TrendReq(hl=hl, tz=tz, retries=0, backoff_factor=0, timeout=(10, 30),
                  requests_args={"headers": {"User-Agent": "media-impacts-app/0.7"}})

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
                st.info(f"Google Trends rate limited. Retrying in {wait}s… (attempt {attempt+1}/{tries})")
                time.sleep(wait)
                wait = min(wait * 2, 60)
                continue
            if "code 400" in msg and timeframe != DEFAULT_TREND_TIMEFRAME:
                timeframe = DEFAULT_TREND_TIMEFRAME
                continue
            if any(k in msg for k in ["502", "503", "504", "timeout"]):
                time.sleep(3)
                continue
            break
    raise RuntimeError(f"Google Trends failed after retries: {last_err}")

def wiki_weekly_pageviews(title_or_query: str, start: str, end: str, lang: str = "en") -> pd.Series:
    """Wikipedia weekly pageviews (sum of daily), W-MON index, robust to 404 + search fallback."""
    session = requests.Session()
    session.headers.update({"User-Agent": "media-impacts-app/0.7 (contact: you@example.com)"} )

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

    # Prebuild figs (we'll label after we know source)
    fig1 = plt.figure(figsize=(10, 4)); ax1 = fig1.gca()
    ax1.plot(df.index, df["outcome"], label="Outcome interest")
    ax1.axvline(t0_monday, linestyle="--", label=f"Release: {t0.date()}")
    ax1.plot(df.index, model.predict(X), label="Fitted (segmented regression)")
    ax1.set_title("Outcome vs release — quick ITS")
    ax1.set_xlabel("Week (starts Monday)")
    ax1.legend(); fig1.tight_layout()

    fig2 = plt.figure(figsize=(10, 2.8)); ax2 = fig2.gca()
    ax2.plot(df.index, df["film_interest"], label="Project interest")
    ax2.axvline(t0_monday, linestyle="--")
    ax2.set_title("Project interest"); ax2.set_xlabel("Week (starts Monday)")
    fig2.tight_layout()

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
    from pytrends.request import TrendReq
    py = TrendReq(hl="en-US", tz=0, retries=0, backoff_factor=0,
                  requests_args={"headers": {"User-Agent": "media-impacts-app/0.7"}})
    py.build_payload([query], timeframe=timeframe, geo=geo)
    df = py.interest_by_region(resolution="COUNTRY" if geo == "" else "REGION",
                               inc_low_vol=True, inc_geo_code=True).reset_index()
    name_col = "geoName" if "geoName" in df.columns else df.columns[0]
    out = df.rename(columns={name_col: "region"})
    return out

# =====================================================
#       Multi-source selection & primary/fallback
# =====================================================
def get_series_from_source(source: str,
                           film_title: str, outcome_topic: str,
                           geo: str, timeframe: str,
                           start_daily: str, end_daily: str, lang: str) -> Tuple[pd.Series, pd.Series, str, str]:
    """
    Returns (film_series, outcome_series, source_name, y_label) or raises an Exception.
    """
    if source == "Google Trends":
        film_qs    = [q.strip() for q in (film_title or "").split(",") if q.strip()]
        outcome_qs = [q.strip() for q in (outcome_topic or "").split(",") if q.strip()]
        if not film_qs:    raise RuntimeError("Google Trends needs at least one film query.")
        if not outcome_qs: raise RuntimeError("Google Trends needs at least one outcome query.")
        film_s = cached_trends(film_qs, geo=geo, timeframe=timeframe).rename("film_interest")
        out_s  = cached_trends(outcome_qs, geo=geo, timeframe=timeframe).rename("outcome")
        return film_s, out_s, "Google Trends index (0–100)", _y_label("Trends")
    elif source == "Wikipedia":
        film_s = cached_wiki(film_title,  start_daily, end_daily, lang=lang).rename("film_interest")
        out_s  = cached_wiki(outcome_topic, start_daily, end_daily, lang=lang).rename("outcome")
        return film_s, out_s, "Wikipedia pageviews", _y_label("Wikipedia")
    else:
        raise ValueError("Unknown source")

# =====================================================
#                Optional AI explainer
# =====================================================
def ai_explain(context: dict) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(AI explainer unavailable — add OPENAI_API_KEY in Settings → Secrets.)"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        sys_msg = (
            "You are a data journalist. Explain statistical results simply and accurately for a general audience. "
            "Avoid jargon. Always define axes. Keep under ~120 words with bullets."
        )
        user_msg = (
            "Explain a media release impact analysis. Focus on: Overview, Immediate % jump vs pre, "
            "Trend after release (direction only), Cumulative lift idea, Axes meanings, Caveats. "
            "No confidence intervals. Context JSON:\n\n"
            f"{json.dumps(context, default=str)}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI explanation error: {e})"

# =====================================================
#                    UI (center form)
# =====================================================
st.session_state.setdefault("last_params", None)
st.session_state.setdefault("last_results", None)

st.markdown('<div class="center-card">', unsafe_allow_html=True)
with st.form("center_form"):
    film_input = st.text_input(
        "Project title (e.g., Eating Our Way to Extinction)",
        value="", placeholder="Type a film/series title…",
    )
    intervention = st.date_input(
        "Date of release",
        value=dt.date(2021, 9, 30),
        help="Approximate is OK. Weeks align to Mondays."
    )

    with st.expander("Advanced options", expanded=False):
        st.caption("What outcome do you want to measure?")
        outcome_input = st.text_input(
            "Outcome topic",
            value="Plant-based diet",
            placeholder="e.g., vegan diet, meat consumption, food waste",
            help="This is the outcome/idea/behavior to track."
        )

        st.caption("Which data sources?")
        sources = st.multiselect(
            "Measurement sources",
            ["Google Trends", "Wikipedia"],
            default=["Google Trends", "Wikipedia"],
            help="You can select one or both. We’ll use the Primary source for modeling and show the other (if available) for context."
        )
        primary = st.selectbox(
            "Primary source for modeling",
            ["Google Trends", "Wikipedia"],
            index=0 if "Google Trends" in sources else 1,
            help="If the primary fails, we’ll fall back to the other selected source."
        )
        auto_fallback = st.checkbox("Automatically fall back to the other source if the primary fails", value=True)

        st.caption("Google Trends settings")
        colA, colB, colC = st.columns(3)
        with colA:
            geo = st.selectbox("Region (Trends)", ["", "US", "GB", "TH"], index=1, help="Blank = worldwide")
        with colB:
            timeframe = st.selectbox("Timeframe (Trends)", [DEFAULT_TREND_TIMEFRAME, "today 12-m", "today 3-m"], index=0)
        with colC:
            show_regions = st.checkbox("Show interest by region (Trends only)", value=False)

        st.caption("Wikipedia settings")
        colD, colE, colF = st.columns(3)
        with colD: start_daily = st.text_input("Start (YYYYMMDD)", "20190101")
        with colE: end_daily   = st.text_input("End (YYYYMMDD)", dt.date.today().strftime("%Y%m%d"))
        with colF: lang        = st.text_input("Language", "en")

        st.caption("Chart zoom (display only; model uses full range)")
        colZ1, colZ2 = st.columns(2)
        with colZ1:
            pre_w  = st.number_input("Weeks shown before release", min_value=4, max_value=52, value=DEFAULT_PRE_WEEKS, step=1)
        with colZ2:
            post_w = st.number_input("Weeks shown after release",  min_value=12, max_value=156, value=DEFAULT_POST_WEEKS, step=1)

    run = st.form_submit_button("▶️ Run analysis")
st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
#                     Execute analysis
# =====================================================
def _analyze(params: Dict[str, Any]):
    # 1) Try primary
    tried = []
    errors: Dict[str, str] = {}
    film_s = out_s = None
    used_source = None
    y_label = None
    source_name = None

    ordered_sources = [params["primary"]] + [s for s in params["sources"] if s != params["primary"]]

    for src in ordered_sources:
        if src not in params["sources"]:
            continue
        tried.append(src)
        try:
            fs, os_, src_name, ylab = get_series_from_source(
                src,
                params["film_input"], params["outcome_input"],
                params["geo"], params["timeframe"],
                params["start_daily"], params["end_daily"], params["lang"]
            )
            film_s, out_s = fs, os_
            used_source, y_label, source_name = src, ylab, src_name
            break
        except Exception as e:
            errors[src] = str(e)

    if film_s is None or out_s is None:
        raise RuntimeError(f"No data available from selected sources.\nDetails: {errors}")

    # 2) Model on the used source
    model, df, figs, metrics = quick_its(out_s, film_s, params["intervention"].isoformat())
    for f in figs:
        f.gca().set_ylabel(y_label)

    # 3) Build extra displays
    # Counterfactual
    design = sm.add_constant(df[["time","post","time_post","film_lag1"]])
    pred    = model.predict(design)
    cf = design.copy(); cf["post"]=0; cf["time_post"]=0
    pred_cf = model.predict(cf)
    after_mask = df.index >= metrics["t0_monday"]
    excess = (df["outcome"] - pred_cf).where(after_mask, 0.0)

    # Zoom window
    mask_zoom = _zoom_mask(df.index, metrics["t0_monday"], params["pre_w"], params["post_w"])
    dfz = df.loc[mask_zoom].copy()
    pred_cf_z = pred_cf.loc[mask_zoom]
    pred_z    = pred.loc[mask_zoom]
    after_z   = (dfz.index >= metrics["t0_monday"])
    excess_z  = (dfz["outcome"] - pred_cf_z).where(after_z, 0.0)

    # 4) Optional secondary source chart (context only)
    secondary_plot = None
    secondary_name = None
    secondary_y = None
    secondary_err = None
    for src in params["sources"]:
        if src == used_source:
            continue
        try:
            fs2, os2, src2_name, y2 = get_series_from_source(
                src,
                params["film_input"], params["outcome_input"],
                params["geo"], params["timeframe"],
                params["start_daily"], params["end_daily"], params["lang"]
            )
            # Normalize for display (0–100) so it's comparable visually (not for modeling)
            def _normalize(s: pd.Series) -> pd.Series:
                s = s.copy()
                rng = (s.max() - s.min())
                return 100 * (s - s.min()) / rng if rng else s*0
            sec_df = pd.concat({"Outcome (norm)": _normalize(os2), "Project (norm)": _normalize(fs2)}, axis=1).dropna()
            figS = plt.figure(figsize=(10,3)); ax = figS.gca()
            sec_df.plot(ax=ax)
            ax.axvline(metrics["t0_monday"], linestyle="--", label="Release")
            ax.set_title(f"{src2_name} (normalized for comparison)")
            ax.set_xlabel("Week (starts Monday)"); ax.set_ylabel("0–100 (normalized for chart)")
            ax.legend(); figS.tight_layout()
            secondary_plot, secondary_name, secondary_y = figS, src2_name, y2
            break
        except Exception as e:
            secondary_err = str(e)

    # 5) AI context
    sig_level = 0.05
    p_post     = float(model.pvalues.get("post", 1.0))
    p_timepost = float(model.pvalues.get("time_post", 1.0))
    pre_avg    = metrics.get("pre_mean", np.nan)
    pct = np.nan
    if pre_avg not in (None, 0) and not (isinstance(pre_avg, float) and np.isnan(pre_avg)):
        try:
            pct = 100.0 * metrics["level_change"] / pre_avg
        except Exception:
            pct = np.nan
    trend_word = "increased" if metrics["slope_change_per_week"] > 0 else "decreased" if metrics["slope_change_per_week"] < 0 else "did not change"

    # crude half-life estimate (if positive excess)
    half_life_text = "n/a"
    try:
        from scipy.optimize import curve_fit
        pos = (df.index >= metrics["t0_monday"]) & ((df["outcome"] - pred_cf) > 0)
        if pos.sum() >= 6:
            y = (df["outcome"] - pred_cf)[pos].values; t = np.arange(len(y))
            def _exp_decay(t, A, k, c): return A*np.exp(-k*t) + c
            (A, k, c), _ = curve_fit(_exp_decay, t, y, p0=[max(y),0.1,0.0], maxfev=20000)
            hl = (np.log(2)/k) if k>0 else np.nan
            if np.isfinite(hl): half_life_text = f"{hl:.1f} weeks"
    except Exception:
        pass

    ctx = {
        "project_title": params["film_input"],
        "outcome_topic": params["outcome_input"],
        "release_date": str(params["intervention"]),
        "source_used_for_model": source_name,
        "y_axis": _y_label(source_name),
        "immediate_percent_jump": None if np.isnan(pct) else float(round(pct,1)),
        "trend_direction_after": trend_word,
        "significant_level_change": (p_post < sig_level),
        "significant_trend_change": (p_timepost < sig_level),
        "half_life": half_life_text
    }

    return {
        "used_source": used_source,
        "source_name": source_name,
        "y_label": _y_label(source_name),
        "units": _units(source_name),
        "errors": errors,
        "model": model, "df": df, "figs": figs, "metrics": metrics,
        "zoom": (dfz, pred_z, pred_cf_z, after_z, excess_z),
        "secondary": (secondary_plot, secondary_name, secondary_y, secondary_err),
        "ai_context": ctx
    }

if run:
    params = dict(
        film_input=film_input.strip(),
        outcome_input=outcome_input.strip(),
        intervention=intervention,
        sources=sources,
        primary=primary,
        auto_fallback=auto_fallback,
        geo=geo, timeframe=timeframe,
        start_daily=start_daily, end_daily=end_daily, lang=lang,
        pre_w=pre_w, post_w=post_w,
        show_regions=show_regions
    )
    with st.spinner("Fetching data and running ITS…"):
        try:
            res = _analyze(params)
            st.session_state["last_params"] = params
            st.session_state["last_results"] = res
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# Persist results across reruns (e.g., when you click anything)
res = st.session_state.get("last_results")
params = st.session_state.get("last_params")

if res:
    # Source status box
    st.subheader("Data sources")
    statuses = []
    for s in params["sources"]:
        if s == res["used_source"]:
            statuses.append(f'<span class="badge ok">✓ {s} (used)</span>')
        elif res["errors"].get(s):
            statuses.append(f'<span class="badge err">✗ {s} — {res["errors"][s]}</span>')
        else:
            statuses.append(f'<span class="badge warn">• {s} (available for context)</span>')
    st.markdown(" ".join(statuses), unsafe_allow_html=True)

    df = res["df"]
    (dfz, pred_z, pred_cf_z, after_z, excess_z) = res["zoom"]
    y_label = res["y_label"]
    units   = res["units"]
    t0 = res["metrics"]["t0_monday"]

    # ---------------- Results (clean, % only, no CI numbers) ----------------
    st.subheader("Results")
    model = res["model"]
    sig_level = 0.05
    p_post     = float(model.pvalues.get("post", 1.0))
    p_timepost = float(model.pvalues.get("time_post", 1.0))
    pre_avg    = res["metrics"].get("pre_mean", np.nan)

    pct = np.nan
    if pre_avg not in (None, 0) and not (isinstance(pre_avg, float) and np.isnan(pre_avg)):
        try:
            pct = 100.0 * res["metrics"]["level_change"] / pre_avg
        except Exception:
            pct = np.nan
    pct_str = f"{_fmt_float(pct, 1)}%" if not np.isnan(pct) else "n/a"
    trend_word = "increased" if res["metrics"]["slope_change_per_week"] > 0 else "decreased" if res["metrics"]["slope_change_per_week"] < 0 else "did not change"

    st.markdown(
        f"""
- **Immediate change at release:** **{pct_str}** relative to a typical pre-release week. {'Significant.' if p_post < sig_level else 'Not statistically significant.'}
- **After-release trend:** **{trend_word}** over subsequent weeks. {'Significant.' if p_timepost < sig_level else 'Not statistically significant.'}
- **Measurement:** **{res["source_name"]}** (Y-axis = {y_label}).
"""
    )

    # Axes key card
    st.markdown(
        f"""
<div class="callout">
<b>Axes key</b><br>
• X-axis = week (starts Monday).<br>
• Y-axis = <b>{y_label}</b>.<br>
• Source used for modeling = <b>{res["source_name"]}</b>.
</div>
""", unsafe_allow_html=True)

    # ---------------- Core charts (zoomed window) + captions ----------------
    # 1) Outcome vs release (zoomed, fitted)
    figA = plt.figure(figsize=(10,4))
    axA = figA.gca()
    axA.plot(dfz.index, dfz["outcome"], label="Outcome interest")
    axA.plot(dfz.index, pred_z, label="Fitted (segmented)")
    axA.axvline(t0, linestyle="--", label="Release")
    axA.set_title("Outcome vs release — zoomed"); axA.set_xlabel("Week (starts Monday)"); axA.set_ylabel(y_label)
    axA.legend(); figA.tight_layout()
    st.pyplot(figA)
    st.caption("Focuses on the months around release to show the jump and post-release pattern. "
               "Y = attention level; higher means more public interest.")

    # 2) Project interest (zoomed)
    figB = plt.figure(figsize=(10,3))
    axB = figB.gca()
    axB.plot(dfz.index, dfz["film_interest"])
    axB.axvline(t0, linestyle="--")
    axB.set_title("Project interest — zoomed"); axB.set_xlabel("Week (starts Monday)"); axB.set_ylabel(y_label)
    figB.tight_layout()
    st.pyplot(figB)
    st.caption("How much attention the project itself received around release.")

    # 3) Actual vs Counterfactual (zoomed)
    figC = plt.figure(figsize=(10,4))
    axC = figC.gca()
    axC.plot(dfz.index, dfz["outcome"], label="Actual outcome")
    axC.plot(dfz.index, pred_cf_z, label="Counterfactual (no release)")
    axC.fill_between(dfz.index[after_z], pred_cf_z[after_z], dfz["outcome"][after_z], alpha=0.25)
    axC.axvline(t0, linestyle="--", label="Release")
    axC.set_title("Actual vs Counterfactual — zoomed"); axC.set_xlabel("Week (starts Monday)"); axC.set_ylabel(y_label)
    axC.legend(); figC.tight_layout()
    st.pyplot(figC)
    st.caption("Shaded area after release shows weekly lift vs a ‘no-release’ baseline.")

    # 4) Weekly excess (zoomed)
    figD = plt.figure(figsize=(10,3))
    axD = figD.gca()
    axD.bar(dfz.index[after_z], excess_z[after_z])
    axD.axvline(t0, linestyle="--")
    axD.set_title("Weekly Excess vs Counterfactual — zoomed"); axD.set_xlabel("Week (starts Monday)"); axD.set_ylabel(f"Excess {res['units']}/week")
    figD.tight_layout()
    st.pyplot(figD)
    st.caption("How much higher (or lower) the outcome was each week vs the counterfactual after release.")

    # 5) Cumulative excess (full period)
    excess_full = (df["outcome"] - sm.OLS(df["outcome"], sm.add_constant(df[["time","film_lag1"]])).fit().predict(sm.add_constant(df[["time","film_lag1"]]))).where(df.index>=t0, 0.0)
    figE = plt.figure(figsize=(10,3))
    axE = figE.gca()
    axE.plot(df.index, excess_full.cumsum())
    axE.axvline(t0, linestyle="--")
    axE.set_title("Cumulative Excess Interest — full period"); axE.set_xlabel("Week (starts Monday)"); axE.set_ylabel(f"Cumulative {res['units']}")
    figE.tight_layout()
    st.pyplot(figE)
    st.caption("Running total of extra attention vs a simple baseline over the entire time window.")

    # 6) Secondary source (if available)
    sec_plot, sec_name, sec_y, sec_err = res["secondary"]
    if sec_plot is not None:
        st.subheader("Secondary source (context)")
        st.pyplot(sec_plot)
        st.caption(f"Normalized for visual comparison. Source: {sec_name}.")
    elif sec_err:
        st.info(f"Secondary source unavailable: {sec_err}")

    # ---------------- Automatic AI explanation ----------------
    st.subheader("Plain-English summary")
    st.markdown(ai_explain(res["ai_context"]))

    # ---------------- Model table & downloads ----------------
    coef = (model.params.to_frame("coef")
            .join(model.bse.to_frame("stderr"))
            .join(model.pvalues.to_frame("pval")))
    st.subheader("Model coefficients (technical)")
    st.dataframe(coef.style.format({"coef":"{:.4f}","stderr":"{:.4f}","pval":"{:.4f}"}))

    merged = df.copy(); merged.index.name = "week_start"
    st.download_button("⬇️ Coefficients (CSV)", coef.to_csv().encode(), file_name="its_coefficients.csv")
    st.download_button("⬇️ Weekly series (CSV)", merged.to_csv().encode(), file_name="weekly_series.csv")

    # ZIP bundle
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, figz in enumerate([figA, figB, figC, figD, figE], start=1):
            fbuf = io.BytesIO(); figz.savefig(fbuf, format="png", dpi=200, bbox_inches="tight")
            z.writestr(f"figure_{i}.png", fbuf.getvalue())
        z.writestr("its_coefficients.csv", coef.to_csv().encode())
        z.writestr("weekly_series.csv", merged.to_csv().encode())
    st.download_button("⬇️ Report bundle (ZIP)", buf.getvalue(), file_name="its_report_bundle.zip")

    # Optional: Trends regional breakdown
    if (res["used_source"] == "Google Trends") and params["show_regions"]:
        try:
            st.subheader("Interest by region (Top 15)")
            qcol = (params["film_input"] or "").split(",")[0].strip()
            reg = cached_regions(qcol, geo=params["geo"], timeframe=params["timeframe"])
            top = reg.sort_values(qcol, ascending=False).head(15)
            figR = plt.figure(figsize=(10,5))
            axR = figR.gca()
            axR.barh(top["region"][::-1], top[qcol][::-1])
            axR.set_title("Where interest is highest (Google Trends)")
            axR.set_xlabel("Relative interest (0–100 index)"); axR.set_ylabel("Region")
            figR.tight_layout()
            st.pyplot(figR)
        except Exception as e:
            st.info(f"Region breakdown unavailable: {e}")
