import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="AI Predictive Maintenance", page_icon="🏭", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "predictive_maintenance_day2_artifacts"

FILES = {
    "telemetry": BASE_DIR / "PdM_telemetry.csv",
    "errors": BASE_DIR / "PdM_errors.csv",
    "failures": BASE_DIR / "PdM_failures.csv",
    "machines": BASE_DIR / "PdM_machines.csv",
    "maint": BASE_DIR / "PdM_maint.csv",
}

st.markdown("""
<style>
/* ---------- Global ---------- */
:root {
    --ink:#172033;
    --muted:#667085;
    --line:#E6EAF0;
    --card:#FFFFFF;
    --blue:#4F46E5;
    --cyan:#06B6D4;
    --green:#10B981;
    --amber:#F59E0B;
    --red:#EF4444;
    --purple:#8B5CF6;
}
.stApp {
    background: linear-gradient(135deg,#F7F9FC 0%,#F4F7FF 55%,#F8F5FF 100%);
    color:var(--ink);
}
.block-container { padding-top:1.1rem; max-width:1450px; padding-bottom:3rem; }
[data-testid="stHeader"] { background:rgba(255,255,255,.75); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#111827 0%,#1E1B4B 55%,#312E81 100%);
}
section[data-testid="stSidebar"] * { color:#F8FAFC !important; }
section[data-testid="stSidebar"] .stRadio label {
    border-radius:10px; padding:7px 9px; margin:2px 0; transition:.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.10); }
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18);
}

/* ---------- Hero ---------- */
.hero {
    position:relative; overflow:hidden; padding:1.35rem 1.55rem;
    border:1px solid #DDE4F5; border-radius:20px;
    background:linear-gradient(120deg,#FFFFFF 0%,#EEF4FF 58%,#F3EEFF 100%);
    box-shadow:0 10px 30px rgba(44,62,100,.08); margin-bottom:1rem;
}
.hero:after {
    content:""; position:absolute; width:180px; height:180px; border-radius:50%;
    right:-55px; top:-75px; background:rgba(99,102,241,.10);
}
.hero h1 { margin:0; color:#172033; font-size:2.05rem; font-weight:800; letter-spacing:-.02em; }
.hero p { margin:.4rem 0 0; color:#667085; font-size:.98rem; }

/* ---------- Section headings ---------- */
h2,h3 { color:#172033 !important; letter-spacing:-.015em; }
h3 { margin-top:1.25rem !important; }

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
    background:#FFFFFF; border:1px solid #E3E8F2; border-radius:15px;
    padding:12px 15px; box-shadow:0 5px 16px rgba(31,41,55,.055);
    min-height:105px;
}
div[data-testid="stMetric"] label { color:#667085 !important; font-weight:650; }
div[data-testid="stMetricValue"] { color:#172033 !important; font-weight:800; }
div[data-testid="stMetricDelta"] { font-weight:700; }

/* ---------- Priority cards ---------- */
.priority-high { background:linear-gradient(90deg,#FFF0F0,#FFF8F8); border:1px solid #F3B4B4; color:#B42318; padding:.95rem 1.05rem; border-radius:14px; font-weight:800; box-shadow:0 5px 15px rgba(239,68,68,.08); }
.priority-medium { background:linear-gradient(90deg,#FFF7E6,#FFFCF2); border:1px solid #F3D18A; color:#9A6700; padding:.95rem 1.05rem; border-radius:14px; font-weight:800; box-shadow:0 5px 15px rgba(245,158,11,.08); }
.priority-low { background:linear-gradient(90deg,#ECFDF5,#F5FFF9); border:1px solid #A7E2C4; color:#087443; padding:.95rem 1.05rem; border-radius:14px; font-weight:800; box-shadow:0 5px 15px rgba(16,185,129,.08); }

/* ---------- Info / success / warning ---------- */
div[data-testid="stAlert"] { border-radius:14px; border-width:1px; box-shadow:0 4px 14px rgba(31,41,55,.045); }

/* ---------- Tables ---------- */
div[data-testid="stDataFrame"] {
    border:1px solid #E2E7F0; border-radius:14px; overflow:hidden;
    box-shadow:0 5px 18px rgba(31,41,55,.045); background:#fff;
}

/* ---------- Select boxes ---------- */
div[data-baseweb="select"] > div { border-radius:11px; border-color:#D9E0EC; }

/* ---------- Expanders ---------- */
details { background:rgba(255,255,255,.8); border:1px solid #E3E8F2 !important; border-radius:13px !important; }

/* ---------- Charts ---------- */
div[data-testid="stVegaLiteChart"], div[data-testid="stArrowVegaLiteChart"] {
    background:#FFFFFF; border:1px solid #E3E8F2; border-radius:15px;
    padding:8px; box-shadow:0 5px 18px rgba(31,41,55,.045);
}

/* ---------- Small helper pills ---------- */
.section-pill {
    display:inline-block; padding:.28rem .65rem; border-radius:999px;
    background:#EEF2FF; color:#4338CA; font-size:.78rem; font-weight:750;
    border:1px solid #D9DDFC; margin-bottom:.35rem;
}

/* ---------- Mobile-ish width ---------- */
@media (max-width: 900px) {
    .hero h1 { font-size:1.55rem; }
}
</style>
""", unsafe_allow_html=True)

required = list(FILES.values()) + [
    ARTIFACT_DIR / "enterprise_preprocessor.joblib",
    ARTIFACT_DIR / "selected_model_6h.joblib",
    ARTIFACT_DIR / "selected_model_24h.joblib",
    ARTIFACT_DIR / "selected_models.csv",
    ARTIFACT_DIR / "metrics_all_models.csv",
    ARTIFACT_DIR / "cost_thresholds_all_models.csv",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    st.error("Required files are missing.")
    st.code("\n".join(missing))
    st.stop()

@st.cache_data(show_spinner="Loading data...")
def load_data():
    telemetry = pd.read_csv(FILES["telemetry"], parse_dates=["datetime"])
    errors = pd.read_csv(FILES["errors"], parse_dates=["datetime"])
    failures = pd.read_csv(FILES["failures"], parse_dates=["datetime"])
    machines = pd.read_csv(FILES["machines"])
    maint = pd.read_csv(FILES["maint"], parse_dates=["datetime"])
    return telemetry, errors, failures, machines, maint

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    pre = joblib.load(ARTIFACT_DIR / "enterprise_preprocessor.joblib")
    m6 = joblib.load(ARTIFACT_DIR / "selected_model_6h.joblib")
    m24 = joblib.load(ARTIFACT_DIR / "selected_model_24h.joblib")
    selected = pd.read_csv(ARTIFACT_DIR / "selected_models.csv")
    metrics = pd.read_csv(ARTIFACT_DIR / "metrics_all_models.csv")
    thresholds = pd.read_csv(ARTIFACT_DIR / "cost_thresholds_all_models.csv")
    return pre, m6, m24, selected, metrics, thresholds

@st.cache_data(show_spinner="Building enterprise features... first run may take a little while.")
def build_features(telemetry, errors, machines, maint):
    base = telemetry.sort_values(["datetime","machineID"]).reset_index(drop=True)
    df = base.sort_values(["machineID","datetime"]).reset_index(drop=True).copy()
    sensors = ["volt","rotate","pressure","vibration"]
    for c in sensors: df[c] = df[c].astype("float32")
    g = df.groupby("machineID", sort=False)

    for c in sensors:
        for w,label in [(6,"6h"),(24,"24h")]:
            r = g[c].rolling(w,min_periods=w)
            df[f"{c}_mean_{label}"] = r.mean().reset_index(level=0,drop=True).astype("float32")
            df[f"{c}_std_{label}"] = r.std().reset_index(level=0,drop=True).astype("float32")
            df[f"{c}_min_{label}"] = r.min().reset_index(level=0,drop=True).astype("float32")
            df[f"{c}_max_{label}"] = r.max().reset_index(level=0,drop=True).astype("float32")
        df[f"{c}_change_6h"] = (df[c]-g[c].shift(6)).astype("float32")
        df[f"{c}_change_24h"] = (df[c]-g[c].shift(24)).astype("float32")
        df[f"{c}_trend_24h"] = (df[f"{c}_change_24h"]/24.0).astype("float32")

    for c in sensors:
        hist = g[c].rolling(168,min_periods=24)
        hm = hist.mean().reset_index(level=0,drop=True).groupby(df.machineID).shift(1)
        hs = hist.std().reset_index(level=0,drop=True).groupby(df.machineID).shift(1)
        df[f"{c}_baseline_7d"] = hm.astype("float32")
        df[f"{c}_zscore_7d"] = ((df[c]-hm)/(hs+1e-6)).astype("float32")

    zcols=[f"{c}_zscore_7d" for c in sensors]
    df["sensor_anomaly_count"]=sum(df[z].abs().gt(3).astype("int8") for z in zcols).astype("int8")
    df["sensor_anomaly_flag"]=(df.sensor_anomaly_count>0).astype("int8")
    df["pressure_vibration_ratio"]=(df.pressure/(df.vibration.abs()+1e-3)).astype("float32")
    df["rotation_vibration_interaction"]=(df.rotate*df.vibration).astype("float32")
    df["voltage_rotation_interaction"]=(df.volt*df.rotate).astype("float32")

    df=df.merge(machines[["machineID","model","age"]],on="machineID",how="left")
    df["age_bucket"]=pd.cut(df.age,[-1,5,10,15,100],labels=["0-5","6-10","11-15","16+"]).astype(str)

    eh=errors.assign(event=1).groupby(["machineID","datetime"],as_index=False).event.sum().rename(columns={"event":"error_count"})
    df=df.merge(eh,on=["machineID","datetime"],how="left"); df["error_count"]=df.error_count.fillna(0).astype("float32")
    et=errors.assign(event=1).pivot_table(index=["machineID","datetime"],columns="errorID",values="event",aggfunc="sum",fill_value=0).reset_index()
    etcols=[c for c in et.columns if c not in ["machineID","datetime"]]
    et["distinct_error_types"]=et[etcols].gt(0).sum(axis=1).astype("int8")
    df=df.merge(et[["machineID","datetime","distinct_error_types"]],on=["machineID","datetime"],how="left")
    df["distinct_error_types"]=df.distinct_error_types.fillna(0).astype("int8")
    df.sort_values(["machineID","datetime"],inplace=True); g=df.groupby("machineID",sort=False)
    for w,label in [(6,"6h"),(24,"24h"),(168,"7d")]:
        df[f"errors_{label}"]=g.error_count.rolling(w,min_periods=1).sum().reset_index(level=0,drop=True).astype("float32")
    df["distinct_error_types_24h"]=g.distinct_error_types.rolling(24,min_periods=1).max().reset_index(level=0,drop=True).astype("int8")
    df["error_burst_flag"]=(df.errors_24h>=2).astype("int8")

    last_err=pd.merge_asof(
        df[["machineID","datetime"]].sort_values(["datetime","machineID"]),
        errors[["machineID","datetime"]].rename(columns={"datetime":"error_time"}).sort_values(["error_time","machineID"]),
        left_on="datetime",right_on="error_time",by="machineID",direction="backward",allow_exact_matches=True)
    df=df.merge(last_err[["machineID","datetime","error_time"]],on=["machineID","datetime"],how="left")
    df["hours_since_error"]=((df.datetime-df.error_time).dt.total_seconds()/3600).astype("float32")
    df.drop(columns="error_time",inplace=True)

    mh=maint.assign(event=1).groupby(["machineID","datetime"],as_index=False).event.sum().rename(columns={"event":"maintenance_count"})
    df=df.merge(mh,on=["machineID","datetime"],how="left"); df["maintenance_count"]=df.maintenance_count.fillna(0).astype("float32")
    df.sort_values(["machineID","datetime"],inplace=True); g=df.groupby("machineID",sort=False)
    for w,label in [(168,"7d"),(720,"30d"),(2160,"90d")]:
        df[f"maintenance_count_{label}"]=g.maintenance_count.rolling(w,min_periods=1).sum().reset_index(level=0,drop=True).astype("float32")

    last_maint=pd.merge_asof(
        df[["machineID","datetime"]].sort_values(["datetime","machineID"]),
        maint[["machineID","datetime"]].rename(columns={"datetime":"maintenance_time"}).sort_values(["maintenance_time","machineID"]),
        left_on="datetime",right_on="maintenance_time",by="machineID",direction="backward",allow_exact_matches=True)
    df=df.merge(last_maint[["machineID","datetime","maintenance_time"]],on=["machineID","datetime"],how="left")
    df["hours_since_maintenance"]=((df.datetime-df.maintenance_time).dt.total_seconds()/3600).astype("float32")
    df["days_since_maintenance"]=df.hours_since_maintenance/24.0
    df["maintenance_overdue_30d"]=(df.hours_since_maintenance>720).astype("int8")
    df.drop(columns="maintenance_time",inplace=True)

    mc=maint.assign(event=1).pivot_table(index=["machineID","datetime"],columns="comp",values="event",aggfunc="sum",fill_value=0).reset_index()
    for c in [c for c in mc.columns if c not in ["machineID","datetime"]]:
        mc.rename(columns={c:f"maint_comp_{c}_event"},inplace=True)
    df=df.merge(mc,on=["machineID","datetime"],how="left")
    df.sort_values(["machineID","datetime"],inplace=True); g=df.groupby("machineID",sort=False)
    for c in [c for c in mc.columns if c.startswith("maint_comp_")]:
        df[c]=df[c].fillna(0).astype("float32")
        df[c.replace("_event","_30d")]=g[c].rolling(720,min_periods=1).sum().reset_index(level=0,drop=True).astype("float32")

    for c in sensors:
        valid=g[c].rolling(24,min_periods=1).count().reset_index(level=0,drop=True)
        df[f"{c}_coverage_24h"]=(valid/24.0).astype("float32")
        df[f"{c}_low_coverage_flag"]=(valid<18).astype("int8")
    df["sensor_coverage_24h"]=df[[f"{c}_coverage_24h" for c in sensors]].mean(axis=1).astype("float32")
    df["sensor_low_coverage_flag"]=(df.sensor_coverage_24h<0.75).astype("int8")
    df["data_freshness_hours"]=0.0

    return df.dropna(subset=["vibration_mean_24h","vibration_std_24h"]).reset_index(drop=True)

telemetry, errors, failures, machines, maint = load_data()
pre, model6, model24, selected, metrics, thresholds = load_models()
features = build_features(telemetry, errors, machines, maint)

feature_names = list(pre.feature_names_in_)
missing_features = [c for c in feature_names if c not in features.columns]
if missing_features:
    st.error("Feature mismatch between dashboard and trained model.")
    st.code("\n".join(missing_features))
    st.stop()

def selected_row(h):
    return selected[selected["horizon"] == h].iloc[0]

threshold6 = float(selected_row(6)["threshold"])
threshold24 = float(selected_row(24)["threshold"])
model_name6 = str(selected_row(6)["model"])
model_name24 = str(selected_row(24)["model"])

def priority(row,r6,r24):
    if r6 >= threshold6 or r24 >= threshold24: return "HIGH"
    supporting = (
        row.sensor_anomaly_count > 0 or row.errors_24h >= 1 or
        row.error_burst_flag == 1 or
        (pd.notna(row.hours_since_maintenance) and row.hours_since_maintenance > 720) or
        row.sensor_coverage_24h < 0.75
    )
    return "MEDIUM" if supporting else "LOW"

def reasons(row,r6,r24):
    out=[]
    if r6 >= threshold6: out.append(f"6h failure risk ({r6:.1%}) is above its cost-optimized threshold ({threshold6:.2f}).")
    if r24 >= threshold24: out.append(f"24h failure risk ({r24:.1%}) is above its cost-optimized threshold ({threshold24:.2f}).")
    if row.sensor_anomaly_count > 0: out.append(f"{int(row.sensor_anomaly_count)} sensor anomaly flag(s) versus the machine's 7-day baseline.")
    if row.errors_24h >= 1: out.append(f"{int(row.errors_24h)} error event(s) in the last 24 hours.")
    if row.error_burst_flag == 1: out.append("Recent error activity shows a burst pattern.")
    if pd.notna(row.hours_since_maintenance) and row.hours_since_maintenance > 720: out.append("More than 30 days since recorded maintenance.")
    if row.sensor_coverage_24h < 0.75: out.append("Recent sensor coverage is below 75%.")
    return out or ["No strong warning signal detected."]


def calculate_risk_history(history_df, preprocessor, model6h, model24h, feature_cols):
    """Calculate historical 6h/24h model risk for the selected machine."""
    usable = history_df.dropna(subset=feature_cols).copy()
    if usable.empty:
        return pd.DataFrame()
    Xh = preprocessor.transform(usable[feature_cols]).astype("float32")
    usable["risk_6h"] = model6h.predict_proba(Xh)[:, 1]
    usable["risk_24h"] = model24h.predict_proba(Xh)[:, 1]
    return usable[["datetime", "risk_6h", "risk_24h"]]

st.sidebar.title("🏭 Predictive Maintenance")
page = st.sidebar.radio("Navigate",["Equipment Monitor","Fleet Overview","Model Comparison"])

if page == "Equipment Monitor":
    ids = sorted(features.machineID.unique())
    machine = st.sidebar.selectbox("Select equipment", ids, format_func=lambda x:f"Machine {int(x)}")
    hist = features[features.machineID == machine].sort_values("datetime")
    row = hist.iloc[-1]

    X = pre.transform(pd.DataFrame([row[feature_names].to_dict()])).astype("float32")
    r6 = float(model6.predict_proba(X)[0, 1])
    r24 = float(model24.predict_proba(X)[0, 1])
    p = priority(row, r6, r24)
    status = "HIGH RISK" if r24 >= threshold24 else ("WATCH" if p == "MEDIUM" else "NORMAL")

    st.markdown(
        '<div class="hero"><h1>🏭 AI-Powered Predictive Maintenance</h1>'
        '<p>Campus equipment failure early-warning and maintenance decision system</p></div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"Machine {int(machine)} • Asset model {row.model} • Age {int(row.age)} years • "
        f"Latest observation: {row.datetime}"
    )

    # Executive status card
    css = {"HIGH":"priority-high","MEDIUM":"priority-medium","LOW":"priority-low"}[p]
    st.markdown(
        f'<div class="{css}">● {p} MAINTENANCE PRIORITY &nbsp; | &nbsp; STATUS: {status}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<span class="section-pill">LIVE MODEL OUTPUT</span>', unsafe_allow_html=True)
    st.markdown("### Failure Early-Warning")
    a,b,c,d = st.columns(4)
    a.metric("6h Failure Risk", f"{r6:.1%}", delta=f"Threshold {threshold6:.2f}")
    b.metric("24h Failure Risk", f"{r24:.1%}", delta=f"Threshold {threshold24:.2f}")
    c.metric("Errors — 24h", f"{row.errors_24h:.0f}")
    d.metric("Sensor Coverage", f"{row.sensor_coverage_24h:.0%}")

    # Action
    if p == "HIGH":
        action = "Inspect this equipment promptly and review the latest sensor and error history before the next operating cycle."
    elif p == "MEDIUM":
        action = "Schedule an inspection and continue monitoring the highlighted sensor/error signals."
    else:
        action = "No immediate maintenance action indicated; continue routine monitoring."

    st.info(f"**🛠 Recommended action:** {action}")

    # Explainability
    st.markdown('<span class="section-pill">EXPLAINABILITY</span>', unsafe_allow_html=True)
    st.markdown("### 🔍 Why was this equipment flagged?")

    if r6 >= threshold6 or r24 >= threshold24:
        model_lines = []
        if r6 >= threshold6:
            model_lines.append(
                f"6-hour model risk is **{r6:.1%}**, above its cost-optimized threshold of **{threshold6:.2f}**."
            )
        if r24 >= threshold24:
            model_lines.append(
                f"24-hour model risk is **{r24:.1%}**, above its cost-optimized threshold of **{threshold24:.2f}**."
            )
        st.error("**🤖 MODEL SIGNAL — ALERT TRIGGERED**\n\n" + "\n\n".join(model_lines))
    else:
        st.success(
            f"**🤖 MODEL SIGNAL — NO FAILURE ALERT**\n\n"
            f"6h risk: **{r6:.1%}** vs **{threshold6:.2f}**  |  "
            f"24h risk: **{r24:.1%}** vs **{threshold24:.2f}**"
        )

    evidence = []
    for sensor in ["vibration", "pressure", "rotate", "volt"]:
        zcol = f"{sensor}_zscore_7d"
        if zcol in row.index and pd.notna(row[zcol]) and abs(float(row[zcol])) >= 2:
            direction = "above" if float(row[zcol]) > 0 else "below"
            evidence.append(
                f"**{sensor.title()}** is {abs(float(row[zcol])):.1f}σ {direction} its 7-day machine baseline."
            )

    change = float(row["vibration_change_6h"]) if pd.notna(row["vibration_change_6h"]) else 0.0
    if abs(change) >= 5:
        direction = "increased" if change > 0 else "decreased"
        evidence.append(
            f"**Vibration {direction} by {abs(change):.2f} units** over the last 6 hours."
        )

    if row.errors_24h >= 1:
        evidence.append(f"**{int(row.errors_24h)} error event(s)** were recorded in the last 24 hours.")
    if row.error_burst_flag == 1:
        evidence.append("**Recent error activity shows a burst pattern**, indicating repeated events in a short period.")
    if pd.notna(row.hours_since_maintenance) and row.hours_since_maintenance > 720:
        evidence.append(f"**{row.hours_since_maintenance/24:.0f} days** have passed since the last recorded maintenance.")
    if row.sensor_coverage_24h < 0.75:
        evidence.append(f"**Sensor coverage is only {row.sensor_coverage_24h:.0%}** over the last 24 hours.")

    if evidence:
        st.warning("**🔧 EQUIPMENT EVIDENCE**\n\n" + "\n\n".join("• " + e for e in evidence))
    else:
        st.success("**🔧 EQUIPMENT EVIDENCE**\n\nNo strong abnormal sensor, error, maintenance, or data-quality signal was detected.")

    # Risk trend
    st.markdown("### 📈 Failure Risk — Last 48 Hours")
    risk_hist = calculate_risk_history(hist.tail(48), pre, model6, model24, feature_names)
    if not risk_hist.empty:
        risk_chart = risk_hist.set_index("datetime")[["risk_6h", "risk_24h"]]
        st.line_chart(risk_chart)
    else:
        st.info("Not enough feature history to calculate the risk trend.")

    # Sensor trends
    st.markdown("### 📊 Sensor Trends — Last 48 Hours")
    sensor_chart = hist.tail(48).set_index("datetime")[["volt","rotate","pressure","vibration"]]
    st.line_chart(sensor_chart)

    # Current condition
    st.markdown('<span class="section-pill">REAL-TIME CONDITION</span>', unsafe_allow_html=True)
    st.markdown("### 🔧 Current Equipment Condition")
    a,b,c,d = st.columns(4)
    a.metric("Voltage", f"{row.volt:.2f}")
    b.metric("Rotation", f"{row.rotate:.2f}")
    c.metric("Pressure", f"{row.pressure:.2f}")
    d.metric("Vibration", f"{row.vibration:.2f}")

    # Operational context
    st.markdown("### 📋 Operational Context")
    a,b,c,d = st.columns(4)
    a.metric("Sensor coverage", f"{row.sensor_coverage_24h:.0%}")
    b.metric("Errors — 24h", f"{row.errors_24h:.0f}")
    c.metric(
        "Since maintenance",
        "—" if pd.isna(row.hours_since_maintenance) else f"{row.hours_since_maintenance/24:.1f} days"
    )
    d.metric("7d sensor anomalies", f"{int(row.sensor_anomaly_count)}")

    # History
    st.markdown("### 📋 Machine History")
    failure_hist = failures[failures.machineID == machine].sort_values("datetime")
    maint_hist = maint[maint.machineID == machine].sort_values("datetime")
    error_hist = errors[errors.machineID == machine].sort_values("datetime")

    h1,h2,h3 = st.columns(3)
    with h1:
        st.write("**Recorded failures**")
        if failure_hist.empty:
            st.write("No recorded failures.")
        else:
            last_fail = failure_hist.iloc[-1]
            st.write(f"Component: `{last_fail['failure']}`")
            st.write(f"Date: `{last_fail['datetime']}`")
            st.write(f"Total: **{len(failure_hist)}**")
    with h2:
        st.write("**Maintenance history**")
        if maint_hist.empty:
            st.write("No recorded maintenance.")
        else:
            last_m = maint_hist.iloc[-1]
            st.write(f"Component: `{last_m['comp']}`")
            st.write(f"Date: `{last_m['datetime']}`")
            st.write(f"Total events: **{len(maint_hist)}**")
    with h3:
        st.write("**Recent errors**")
        recent_errors = error_hist.tail(5)
        if recent_errors.empty:
            st.write("No recorded errors.")
        else:
            st.dataframe(
                recent_errors[["datetime","errorID"]].sort_values("datetime", ascending=False),
                use_container_width=True,
                hide_index=True
            )

    with st.expander("🔎 Advanced model details"):
        st.write(f"**6h selected model:** {model_name6}")
        st.write(f"**24h selected model:** {model_name24}")
        st.write("**Missed failure cost:** ₹10,000")
        st.write("**False alarm cost:** ₹500")
        st.write("**Threshold objective:** minimum validation expected operational cost.")
        st.write("The displayed equipment evidence is supporting evidence, not a claim that any single feature caused the model prediction.")

elif page == "Fleet Overview":
    st.markdown(
        '<div class="hero"><h1>📊 Fleet Overview</h1>'
        '<p>Current equipment risk and maintenance priority across the monitored fleet.</p></div>',
        unsafe_allow_html=True
    )

    latest = features.sort_values("datetime").groupby("machineID", as_index=False).tail(1).copy()
    Xlatest = pre.transform(latest[feature_names]).astype("float32")
    latest["risk_6h"] = model6.predict_proba(Xlatest)[:,1]
    latest["risk_24h"] = model24.predict_proba(Xlatest)[:,1]
    latest["priority"] = [priority(r,a,b) for (_,r),a,b in zip(latest.iterrows(), latest.risk_6h, latest.risk_24h)]

    # Fleet summary
    a,b,c,d = st.columns(4)
    a.metric("🏭 Total Machines", len(latest))
    b.metric("🔴 HIGH", int((latest.priority=="HIGH").sum()))
    c.metric("🟠 MEDIUM", int((latest.priority=="MEDIUM").sum()))
    d.metric("🟢 LOW", int((latest.priority=="LOW").sum()))

    # Filters
    st.markdown('<span class="section-pill">FLEET CONTROL CENTER</span>', unsafe_allow_html=True)
    st.markdown("### 🔎 Fleet Filters")
    f1,f2 = st.columns(2)
    with f1:
        priority_filter = st.selectbox("Priority", ["All","HIGH","MEDIUM","LOW"])
    with f2:
        model_filter = st.selectbox("Asset model", ["All"] + sorted(latest["model"].astype(str).unique().tolist()))

    filtered = latest.copy()
    if priority_filter != "All":
        filtered = filtered[filtered.priority == priority_filter]
    if model_filter != "All":
        filtered = filtered[filtered["model"].astype(str) == model_filter]

    # Fleet ranking: every machine is scored automatically; this is the operator's shortlist.
    st.markdown('<div class="section-pill">AUTOMATED FLEET SCREENING</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Top 10 Highest-Risk Equipment")
    st.caption("All machines are scored automatically. This table ranks the highest predicted 24h failure risk after the selected fleet filters.")
    top_risk = filtered.sort_values("risk_24h", ascending=False).head(10).copy()
    top_risk["machineID"] = top_risk.machineID.map(lambda x:f"M-{int(x):03d}")
    top_risk_view = top_risk[
        ["machineID","model","age","risk_6h","risk_24h","priority","errors_24h","sensor_anomaly_count"]
    ]
    st.dataframe(
        top_risk_view.style.format({
            "risk_6h":"{:.3%}","risk_24h":"{:.3%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Action queue: only HIGH/MEDIUM machines require operational attention.
    st.markdown("### 🚨 Equipment Requiring Attention")
    attention = filtered[filtered["priority"].isin(["HIGH", "MEDIUM"])].sort_values(
        "risk_24h", ascending=False
    ).head(10).copy()

    if attention.empty:
        st.success("No HIGH or MEDIUM priority equipment in the current filter. Continue routine monitoring.")
    else:
        st.caption("Only HIGH and MEDIUM priority equipment appears here. These are the assets the maintenance team should investigate first.")
        attention["machineID"] = attention.machineID.map(lambda x:f"M-{int(x):03d}")
        attention_view = attention[
            ["machineID","model","age","risk_6h","risk_24h","priority","errors_24h","sensor_anomaly_count"]
        ]
        st.dataframe(
            attention_view.style.format({
                "risk_6h":"{:.3%}","risk_24h":"{:.3%}"
            }),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View all filtered machines"):
        all_view = filtered[
            ["machineID","model","age","risk_6h","risk_24h","priority","volt","rotate","pressure","vibration"]
        ].copy()
        all_view["machineID"] = all_view.machineID.map(lambda x:f"M-{int(x):03d}")
        st.dataframe(
            all_view.style.format({"risk_6h":"{:.3%}","risk_24h":"{:.3%}"}),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### 📈 Highest 24h Risk")
    chart = filtered.set_index("machineID")[["risk_24h"]].sort_values("risk_24h", ascending=False).head(20)
    st.bar_chart(chart)

else:
    st.markdown(
        '<div class="hero"><h1>🧠 Model Lab & Governance</h1>'
        '<p>Compare candidate models, choose the operationally appropriate model for each warning horizon, and verify generalization.</p></div>',
        unsafe_allow_html=True
    )

    st.markdown('<span class="section-pill">FROM CANDIDATES → VALIDATION → DEPLOYMENT</span>', unsafe_allow_html=True)
    st.markdown(
        "### The model-development decision"
    )
    st.caption(
        "We benchmark the same candidate models under the same methodology. "
        "The deployed model for each horizon is selected using minimum validation expected operational cost — not test-set accuracy alone."
    )

    # Selected/deployed models: make the actual production decision the visual focal point.
    c1, c2 = st.columns(2)
    for col, horizon, accent in [(c1, 6, "⚡"), (c2, 24, "🕒")]:
        sr = selected_row(horizon)
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#FFFFFF,#EEF4FF);border:1px solid #DCE5F5;'
                f'border-radius:18px;padding:1.15rem 1.25rem;box-shadow:0 8px 22px rgba(31,41,55,.06);">'
                f'<div style="font-size:.78rem;color:#4F46E5;font-weight:800;letter-spacing:.06em;">{accent} {horizon}-HOUR WARNING HORIZON</div>'
                f'<div style="font-size:1.55rem;font-weight:850;color:#172033;margin:.35rem 0;">{sr["model"]}</div>'
                f'<div style="color:#667085;font-size:.9rem;">Selected for deployment</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            a,b,c = st.columns(3)
            a.metric("Alert threshold", f'{float(sr["threshold"]):.2f}')
            b.metric("Validation cost", f'₹{float(sr["validation_expected_cost"]):,.0f}')
            c.metric("Validation F1", f'{float(sr["validation_f1"]):.1%}')

    st.markdown("### 💰 Why these models were selected")
    st.caption("Lower validation expected cost means a better trade-off under the stated project costs: missed failure ₹10,000; false alarm ₹500.")

    # Prefer the cost table because this is the actual selection criterion.
    cost = thresholds.copy()
    # Normalize likely column names without assuming an exact artifact schema.
    cost_model_col = next((c for c in ["model","model_name"] if c in cost.columns), None)
    cost_h_col = next((c for c in ["horizon"] if c in cost.columns), None)
    cost_val_col = next((c for c in ["expected_cost","validation_expected_cost","min_expected_cost"] if c in cost.columns), None)
    if cost_model_col and cost_h_col and cost_val_col:
        cost_view = cost[[cost_h_col,cost_model_col,cost_val_col]].copy()
        cost_view.columns = ["horizon","model","validation_cost"]
        cost_view = cost_view.sort_values(["horizon","validation_cost"])
        cost_view["Horizon"] = cost_view["horizon"].astype(str) + "h"
        pivot = cost_view.pivot(index="model",columns="Horizon",values="validation_cost").fillna(0)
        st.bar_chart(pivot)

        st.markdown("### 📋 Candidate models")
        st.caption("Test-set metrics are supporting evidence. They are not used to tune the alert threshold or select the final model.")
    else:
        st.markdown("### 📋 Candidate models")

    # Compact candidate benchmark, split by horizon so the comparison is easier to read.
    test = metrics[metrics.split == "test"].copy()
    display_cols = ["model","precision","recall","f1","false_alarm_rate","TP","FP","FN"]
    for horizon, label in [(6,"⚡ 6-hour candidates"),(24,"🕒 24-hour candidates")]:
        st.markdown(f"#### {label}")
        t = test[test.horizon == horizon][display_cols].sort_values("f1",ascending=False).copy()
        if t.empty:
            st.warning(f"No test results available for the {horizon}h horizon.")
        else:
            st.dataframe(
                t.style.format({"precision":"{:.2%}","recall":"{:.2%}","f1":"{:.2%}","false_alarm_rate":"{:.2%}"}),
                use_container_width=True, hide_index=True
            )

    # Generalization evidence is important, but secondary to the deployment decision.
    with st.expander("🧪 Generalization check — unseen Machine 100"):
        unseen = metrics[metrics.split == "unseen_machine"].copy()
        if unseen.empty:
            st.info("No unseen-machine results available.")
        else:
            ucols = ["horizon","model","precision","recall","f1","false_alarm_rate","TP","FP","FN"]
            st.dataframe(
                unseen[ucols].style.format({"precision":"{:.2%}","recall":"{:.2%}","f1":"{:.2%}","false_alarm_rate":"{:.2%}"}),
                use_container_width=True, hide_index=True
            )
            st.caption(
                "Machine 100 was completely excluded from training and validation. "
                "Because it contains relatively few failure events, this is a generalization check rather than a stable fleet-wide performance estimate."
            )
