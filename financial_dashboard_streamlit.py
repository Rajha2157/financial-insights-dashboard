# financial_dashboard_streamlit.py
# Interactive Financial Insights AI Dashboard (Streamlit)

import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

# Optional XGBoost support
_HAS_XGB = False
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

st.set_page_config(page_title="Financial Insights AI Dashboard", layout="wide")
st.title("📊 Financial Insights AI Dashboard (Streamlit)")
st.caption("Upload a financial CSV, filter it, select an AI model, and generate model-adjusted risk insights.")

# ---------- Helpers ----------
CANDIDATES = {
    "date": ["date", "Date", "timestamp", "Timestamp", "time", "Time"],
    "symbol": ["symbol", "Symbol", "ticker", "Ticker"],
    "price": ["close", "Close", "price", "Price", "stock_price", "Stock Price", "adj_close", "Adj Close", "Adj_Close"],
    "volume": ["volume", "Volume", "trading_volume", "Trading Volume", "shares_traded"],
    "interest": ["interest_rate", "Interest Rate", "interest_rates", "Interest Rates", "fed_funds_rate_pct", "Fed Funds Rate", "rate"],
    "gdp": ["gdp_growth", "GDP Growth", "gdp_qoq_pct", "GDP QoQ", "gdp_qoq", "gdp"],
    "inflation": ["inflation", "Inflation", "inflation_yoy_pct", "Inflation YoY", "cpi_yoy", "CPI YoY"],
}
FEATURE_KEYS = ["price", "volume", "interest", "gdp", "inflation"]

def auto_pick_column(df: pd.DataFrame, candidates: list[str]):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mn) / (mx - mn)

def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")

def compute_base_risk(df: pd.DataFrame, colmap: dict) -> pd.Series:
    p = minmax01(df[colmap["price"]])
    v = minmax01(df[colmap["volume"]])
    r = minmax01(df[colmap["interest"]])
    g = minmax01(df[colmap["gdp"]])
    inf = minmax01(df[colmap["inflation"]])

    # Economic intuition: higher rates & inflation increase risk; stronger GDP reduces risk
    w = {"price": 0.25, "volume": 0.20, "interest": 0.25, "gdp": -0.15, "inflation": 0.25}
    raw = w["price"]*p + w["volume"]*v + w["interest"]*r + w["gdp"]*g + w["inflation"]*inf
    return (minmax01(raw) * 100.0).round(2)

def get_model(model_name: str, random_state: int = 42):
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        )
    if model_name == "XGBoost":
        if not _HAS_XGB:
            return None
        return XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, objective="reg:squarederror", n_jobs=-1
        )
    if model_name == "Neural Network":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=1e-3, max_iter=1200, random_state=random_state
        )
    return LinearRegression()

def derive_feature_weights(model_name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
    feat_names = list(X.columns)

    if model_name == "Linear Regression":
        coef = getattr(model, "coef_", None)
        imp = np.abs(np.array(coef, dtype=float)) if coef is not None else np.ones(len(feat_names))
    elif model_name in ("Random Forest", "XGBoost"):
        imp = getattr(model, "feature_importances_", None)
        imp = np.array(imp, dtype=float) if imp is not None else np.ones(len(feat_names))
    else:
        # Neural net: permutation importance (interpretable)
        try:
            r = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
            imp = np.clip(r.importances_mean, 0, None)
        except Exception:
            imp = np.ones(len(feat_names))

    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    if imp.sum() <= 0:
        imp = np.ones_like(imp)
    imp = imp / imp.sum()
    return dict(zip(feat_names, imp))

def adjusted_risk_from_weights(df: pd.DataFrame, colmap: dict, weights: dict) -> pd.Series:
    p = minmax01(df[colmap["price"]])
    v = minmax01(df[colmap["volume"]])
    r = minmax01(df[colmap["interest"]])
    g = minmax01(df[colmap["gdp"]])
    inf = minmax01(df[colmap["inflation"]])

    g_risk = 1.0 - g  # low GDP => higher risk

    feat_series = {"price": p, "volume": v, "interest": r, "gdp_risk": g_risk, "inflation": inf}
    score01 = 0.0
    for k, s in feat_series.items():
        score01 += weights.get(k, 0.0) * s
    return (minmax01(pd.Series(score01, index=df.index)) * 100.0).round(2)

def gdp_group(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1, q2 = np.nanquantile(s, [0.33, 0.66])
    def grp(x):
        if not np.isfinite(x): return "Unknown"
        if x <= q1: return "Low GDP growth"
        if x <= q2: return "Medium GDP growth"
        return "High GDP growth"
    return s.apply(grp)

def chart_summary_risk(scores: pd.Series, weights: dict) -> str:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if len(s) == 0: return "No risk scores available after filtering."
    p50 = float(np.nanpercentile(s, 50))
    p90 = float(np.nanpercentile(s, 90))
    p10 = float(np.nanpercentile(s, 10))
    spread = p90 - p10
    top_feats = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:2]
    feat_text = ", ".join([f"{k} ({v:.0%})" for k, v in top_feats])
    return f"Median risk is **{p50:.1f}**. Spread (P10–P90) ≈ **{spread:.1f}**. Top drivers: **{feat_text}**."

def chart_summary_scatter(df: pd.DataFrame, colmap: dict, weights: dict) -> str:
    x = safe_numeric(df, colmap["price"])
    y = safe_numeric(df, colmap["volume"])
    valid = x.notna() & y.notna()
    if valid.sum() < 5: return "Not enough points to describe price–volume relationship."
    corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
    direction = "weak" if abs(corr) < 0.2 else ("moderate" if abs(corr) < 0.5 else "strong")
    sign = "positive" if corr >= 0 else "negative"
    top_feat = max(weights.items(), key=lambda kv: kv[1])[0] if weights else "price"
    return f"Price–volume correlation is **{direction}** and **{sign}** (≈ **{corr:.2f}**). Model emphasizes **{top_feat}** most."

def chart_summary_trend(df: pd.DataFrame, colmap: dict, weights: dict, date_col):
    if date_col is None:
        return "No date column mapped, so time trend summary is limited."
    dd = pd.to_datetime(df[date_col], errors="coerce")
    pr = safe_numeric(df, colmap["price"])
    tmp = pd.DataFrame({"date": dd, "price": pr}).dropna().sort_values("date")
    if len(tmp) < 10:
        return "Not enough time-stamped rows to summarize the price trend."
    first, last = float(tmp["price"].iloc[0]), float(tmp["price"].iloc[-1])
    chg = (last - first) / first * 100 if first else np.nan
    top2 = ", ".join([k for k,_ in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:2]])
    return f"Price moved **{chg:.1f}%** over the filtered period. Model is most sensitive to **{top2}**."

def chart_summary_interest_by_gdp(df: pd.DataFrame, colmap: dict, weights: dict) -> str:
    tmp = pd.DataFrame({
        "group": gdp_group(df[colmap["gdp"]]),
        "interest": safe_numeric(df, colmap["interest"])
    }).dropna()
    if tmp["group"].nunique() < 2:
        return "Not enough GDP variation to compare interest by GDP group."
    means = tmp.groupby("group")["interest"].mean().sort_values()
    gap = float(means.iloc[-1] - means.iloc[0])
    w_rate = weights.get("interest", 0.0)
    return f"Avg interest differs by about **{gap:.2f}** across GDP groups. Model weight on interest: **{w_rate:.0%}**."

# ---------- Sidebar ----------
st.sidebar.header("1) Upload CSV")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to begin.")
    st.stop()

st.sidebar.header("2) Column Mapping")
colmap = {}
date_col = auto_pick_column(df, CANDIDATES["date"])
symbol_col = auto_pick_column(df, CANDIDATES["symbol"])
for k in FEATURE_KEYS:
    colmap[k] = auto_pick_column(df, CANDIDATES[k])

with st.sidebar.expander("Adjust mapping", expanded=False):
    cols = list(df.columns)
    date_choice = st.selectbox("Date column (optional)", options=["(none)"] + cols,
                               index=(0 if date_col is None else cols.index(date_col) + 1))
    date_col = None if date_choice == "(none)" else date_choice

    sym_choice = st.selectbox("Symbol/Ticker column (optional)", options=["(none)"] + cols,
                              index=(0 if symbol_col is None else cols.index(symbol_col) + 1))
    symbol_col = None if sym_choice == "(none)" else sym_choice

    for k in FEATURE_KEYS:
        default = colmap[k]
        idx = 0 if default is None else cols.index(default) + 1
        choice = st.selectbox(f"{k} column", options=["(choose)"] + cols, index=idx)
        colmap[k] = None if choice == "(choose)" else choice

missing = [k for k,v in colmap.items() if v is None]
if missing:
    st.error(f"Please map these required columns: {', '.join(missing)}")
    st.stop()

# Numeric conversion
for k,v in colmap.items():
    df[v] = pd.to_numeric(df[v], errors="coerce")
if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ---------- Filters ----------
st.sidebar.header("3) Filters")
def _minmax(col): 
    return float(np.nanmin(df[col])), float(np.nanmax(df[col]))

pmin, pmax = _minmax(colmap["price"])
vmin, vmax = _minmax(colmap["volume"])
rmin, rmax = _minmax(colmap["interest"])
gmin, gmax = _minmax(colmap["gdp"])

price_range = st.sidebar.slider("Stock price range", pmin, pmax, (pmin, pmax))
vol_range   = st.sidebar.slider("Trading volume range", vmin, vmax, (vmin, vmax))
ir_range    = st.sidebar.slider("Interest rate range", rmin, rmax, (rmin, rmax))
gdp_range   = st.sidebar.slider("GDP growth range", gmin, gmax, (gmin, gmax))

fdf = df.copy()
fdf = fdf[(fdf[colmap["price"]] >= price_range[0]) & (fdf[colmap["price"]] <= price_range[1])]
fdf = fdf[(fdf[colmap["volume"]] >= vol_range[0]) & (fdf[colmap["volume"]] <= vol_range[1])]
fdf = fdf[(fdf[colmap["interest"]] >= ir_range[0]) & (fdf[colmap["interest"]] <= ir_range[1])]
fdf = fdf[(fdf[colmap["gdp"]] >= gdp_range[0]) & (fdf[colmap["gdp"]] <= gdp_range[1])]

st.sidebar.write(f"Rows after filtering: **{len(fdf)}**")

# ---------- Model ----------
st.sidebar.header("4) AI Model")
model_choice = st.sidebar.selectbox("Select model", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"], index=1)
if model_choice == "XGBoost" and not _HAS_XGB:
    st.sidebar.warning("XGBoost not installed here. Install `xgboost` locally or choose another model.")
train_btn = st.sidebar.button("Train model & compute risk scores")

# ---------- Main ----------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Filtered Data Preview")
    st.dataframe(fdf.head(30), use_container_width=True)

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="filtered_financial_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Download dashboard code from within app (works when running from saved .py)
    try:
        with open(__file__, "rb") as fh:
            st.download_button(
                "⬇️ Download this dashboard code (.py)",
                data=fh.read(),
                file_name="financial_dashboard_streamlit.py",
                mime="text/x-python",
                use_container_width=True
            )
    except Exception:
        st.info("Code download becomes available after you run the saved .py file.")

# Base risk
fdf = fdf.copy()
fdf["base_risk_score"] = compute_base_risk(fdf, colmap)

X = pd.DataFrame({
    "price": fdf[colmap["price"]],
    "volume": fdf[colmap["volume"]],
    "interest": fdf[colmap["interest"]],
    "gdp_risk": 1.0 - minmax01(fdf[colmap["gdp"]]),
    "inflation": fdf[colmap["inflation"]],
})
y = fdf["base_risk_score"]

weights = {c: 1/len(X.columns) for c in X.columns}

if train_btn:
    if model_choice == "XGBoost" and not _HAS_XGB:
        st.error("XGBoost selected but not installed. Install `xgboost` locally or choose another model.")
    else:
        data = pd.concat([X, y], axis=1).dropna()
        Xc, yc = data[X.columns], data[y.name]
        if len(Xc) < 40:
            st.warning("Not enough clean rows to train. Widen filters or fix mapping.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.25, random_state=42)
            base_model = get_model(model_choice)
            if model_choice in ("Linear Regression", "Neural Network"):
                model = Pipeline([("scaler", StandardScaler()), ("model", base_model)])
            else:
                model = base_model
            model.fit(X_train, y_train)

            # unwrap for importances when needed
            imp_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
            X_imp = X_test if len(X_test) >= 30 else X_train
            y_imp = y_test if len(y_test) >= 30 else y_train
            weights = derive_feature_weights(model_choice, imp_model if model_choice != "Neural Network" else model, X_imp, y_imp)

            try:
                st.sidebar.success(f"Model trained. Test R² ≈ {model.score(X_test, y_test):.3f}")
            except Exception:
                st.sidebar.success("Model trained.")

fdf["model_adjusted_risk_score"] = adjusted_risk_from_weights(fdf, colmap, weights)

with right:
    st.subheader("Model Weights (for adjusted risk)")
    wdf = pd.DataFrame({"feature": list(weights.keys()), "weight": list(weights.values())}).sort_values("weight", ascending=False)
    st.dataframe(wdf, use_container_width=True, hide_index=True)
    st.metric("Median Base Risk", f"{float(np.nanmedian(fdf['base_risk_score'])):.1f}")
    st.metric("Median Adjusted Risk", f"{float(np.nanmedian(fdf['model_adjusted_risk_score'])):.1f}")

st.divider()
st.subheader("📈 Charts + AI summaries")

c1, c2 = st.columns(2, gap="large")
c3, c4 = st.columns(2, gap="large")

with c1:
    st.markdown("#### Risk Score Distribution (Adjusted)")
    s = pd.to_numeric(fdf["model_adjusted_risk_score"], errors="coerce").dropna()
    hist = np.histogram(s, bins=20)
    chart_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
    st.bar_chart(chart_df.set_index("bin_left"))
    st.info(chart_summary_risk(fdf["model_adjusted_risk_score"], weights))

with c2:
    st.markdown("#### Stock Price vs Trading Volume")
    scat = fdf[[colmap["price"], colmap["volume"]]].dropna().copy()
    scat.columns = ["price", "volume"]
    st.scatter_chart(scat, x="price", y="volume")
    st.info(chart_summary_scatter(fdf, colmap, weights))

with c3:
    st.markdown("#### Stock Price Trend Over Time")
    if date_col is None:
        st.warning("Map a date column in the sidebar to see trend over time.")
    else:
        tdf = fdf[[date_col, colmap["price"]]].dropna().sort_values(date_col).copy()
        tdf.columns = ["date", "price"]
        st.line_chart(tdf.set_index("date"))
    st.info(chart_summary_trend(fdf, colmap, weights, date_col))

with c4:
    st.markdown("#### Average Interest Rates by GDP Growth Group")
    tmp = pd.DataFrame({
        "gdp_group": gdp_group(fdf[colmap["gdp"]]),
        "interest": safe_numeric(fdf, colmap["interest"])
    }).dropna()
    if tmp.empty:
        st.warning("Not enough rows for grouped averages.")
    else:
        g = tmp.groupby("gdp_group", as_index=False)["interest"].mean()
        st.bar_chart(g.set_index("gdp_group"))
    st.info(chart_summary_interest_by_gdp(fdf, colmap, weights))

st.divider()
st.subheader("How to run (VS Code)")
st.code(
    "1) Save this file as financial_dashboard_streamlit.py\n"
    "2) Open VS Code > Terminal\n"
    "3) (Optional) Create venv and install:\n"
    "   pip install streamlit pandas numpy scikit-learn xgboost\n"
    "4) Run:\n"
    "   streamlit run financial_dashboard_streamlit.py\n"
    "5) Upload your CSV in the sidebar\n",
    language="text"
)
