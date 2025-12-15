import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import math
import re
from datetime import datetime, timedelta, timezone
import io
import os
import numpy as np
from sqlalchemy import create_engine, inspect

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="GeneQuest (IST)", page_icon="🇮🇳")

CROPS = ["Rice", "Cotton", "Maize", "Wheat", "Soybean"]

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; background-color: white; border: 1px solid #ccc; }
    h1 { background: linear-gradient(to right, #1e3c72, #5a3f99, #e67e22); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .credits { font-size: 0.85rem; color: #444; padding: 10px 0; font-weight: 500; }
    .credits b { font-weight: 700; color: #0e76a8; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    section[data-testid="stSidebar"] div.block-container { padding-top: 1rem; }
    .ist-badge { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px; border: 1px solid #c3e6cb; }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None
if 'dataset_name' not in st.session_state: st.session_state.dataset_name = "None"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def get_engine():
    try:
        db_url = st.secrets["database"]["url"]
        return create_engine(db_url)
    except Exception as e:
        return None

def get_ist_time():
    """Calculates Indian Standard Time (UTC + 5:30)."""
    now_utc = datetime.now(timezone.utc)
    return now_utc + timedelta(hours=5, minutes=30)

def format_table_display(table_name):
    """Formats the filename for display. Does NOT change the time (trusts the filename)."""
    try:
        parts = table_name.split('_')
        if len(parts) < 3: return table_name
        time_part = parts[-1]
        date_part = parts[-2]
        user_text = "_".join(parts[1:-2])
        # Display the time exactly as saved in the file
        dt_obj = datetime.strptime(f"{date_part}_{time_part}", "%d-%m-%Y_%H-%M-%S")
        return f"{user_text} ({dt_obj.strftime('%d-%m-%Y %I:%M:%S %p')})"
    except:
        return table_name

def standardize_columns(df):
    new_cols = []
    for i, col in enumerate(df.columns):
        if i == 0: new_cols.append("Sample"); continue
        clean = col.strip()
        if re.search(r'(marler|mkr|marker)', clean, re.IGNORECASE):
            match = re.search(r'\d+$', clean)
            if match: clean = f"Marker-{match.group()}"
        new_cols.append(clean)
    df.columns = new_cols
    return df

def safe_clean_for_display(df):
    if df is None: return pd.DataFrame()
    df_display = df.copy()
    for col in df_display.columns:
        df_display[col] = df_display[col].astype(str)
    return df_display

def audit_clean_dataframe(df, strategy):
    try:
        df = standardize_columns(df)
        for col in df.columns[1:]:
             df[col] = df[col].astype(str)
             df[col] = df[col].apply(lambda x: str(x).upper().strip() if x.lower() != 'nan' else np.nan)
        
        missing_count = df.isnull().sum().sum()
        if strategy == "Missing values: remove the rows":
            if missing_count > 0:
                df.dropna(inplace=True)
                st.toast(f"Removed rows with {missing_count} missing values.")
        else:
            df.fillna("UNKNOWN", inplace=True)
            if missing_count > 0:
                st.toast(f"Filled {missing_count} missing values with 'UNKNOWN'.")
        return df
    except Exception as e:
        st.error(f"Cleaning Error: {e}")
        return df

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=160)
    
    # --- IST VERIFICATION BADGE ---
    ist_now = get_ist_time().strftime("%I:%M %p")
    st.markdown(f"<div class='ist-badge'>✅ <b>System Active</b><br>IST Time: {ist_now}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='credits'><b>Designed by</b><br>Dr. Santosh Kumar Dodda<br>Bioseed Research India</div>", unsafe_allow_html=True)
    st.title("🧬 Controls")

    st.header("1. Input Data")
    uploaded_files = st.file_uploader("Upload CSV", accept_multiple_files=True, type="csv")
    if uploaded_files and st.button("📥 Load CSVs"):
        frames = []
        for f in uploaded_files:
            try:
                d = pd.read_csv(f)
                d = standardize_columns(d)
                d = d.astype(str)
                frames.append(d)
            except: st.error(f"Failed: {f.name}")
        if frames:
            merged_df = pd.concat(frames, ignore_index=True)
            st.session_state.df = merged_df
            st.session_state.dataset_name = "Unsaved_CSV_Data"
            st.success(f"Loaded {len(merged_df)} rows.")

    st.markdown("---")
    st.markdown("###### Data Prep Actions")
    audit_clicked = st.button("🧹 Audit & Clean Data")
    strategy = st.radio("Strategy:", ["Fill UNKNOWN", "Remove Rows"], label_visibility="collapsed")
    
    st.markdown("")
    if st.button("🔄 Reset View"):
        st.session_state.df = None; st.session_state.dataset_name = "None"; st.rerun()

    if audit_clicked and st.session_state.df is not None:
        val = "Missing values: remove the rows" if strategy == "Remove Rows" else "Fill UNKNOWN"
        st.session_state.df = audit_clean_dataframe(st.session_state.df, val)
        st.success("Cleaned!")

    st.markdown("---")
    st.header("2. Save to Cloud")
    with st.form("save_form"):
        target_crop = st.selectbox("Crop Folder", CROPS)
        save_name = st.text_input("Name (e.g. Exp1)")
        if st.form_submit_button("💾 Save to DB"):
            if st.session_state.df is None or not save_name:
                st.error("No data or name!")
            else:
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', save_name)
                
                # --- KEY FIX: FORCE IST TIME INTO FILENAME ---
                ts_obj = get_ist_time()
                ts_str = ts_obj.strftime("%d-%m-%Y_%H-%M-%S")
                table_name = f"{target_crop}_{clean_name}_{ts_str}"
                
                engine = get_engine()
                if engine:
                    try:
                        st.session_state.df.to_sql(table_name, engine, if_exists="fail", index=False)
                        st.session_state.dataset_name = table_name
                        st.success(f"Saved! Time: {ts_obj.strftime('%I:%M %p')}")
                    except Exception as e: st.error(f"Save Error: {e}")
                else:
                    st.error("Check Secrets.")

    st.markdown("---")
    st.header("3. Load from Cloud")
    load_crop = st.selectbox("Browse Folder", CROPS, key="load_crop_box")
    engine = get_engine()
    raw_tables = []
    if engine:
        try:
            insp = inspect(engine)
            all_tables = insp.get_table_names()
            raw_tables = [t for t in all_tables if t.startswith(f"{load_crop}_")]
            raw_tables.sort(reverse=True)
        except: pass
    
    table_options = {format_table_display(t): t for t in raw_tables}
    selected_displays = st.multiselect("Select Dataset(s)", list(table_options.keys()))
    if st.button("📂 Load from Cloud") and selected_displays:
        frames = []
        for dname in selected_displays:
            try:
                d = pd.read_sql_table(table_options[dname], engine)
                d = d.astype(str)
                d = standardize_columns(d)
                frames.append(d)
            except Exception as e: st.error(f"Error: {e}")
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            st.session_state.df = merged
            st.session_state.dataset_name = f"Merged ({len(frames)})"
            st.success(f"Loaded {len(merged)} rows.")

    st.markdown("---")
    st.header("4. Visualization")
    tree_style = st.selectbox("Tree Style", ["Rectangular", "Circular"])

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title("GeneQuest (Cloud Edition)")

if st.session_state.df is None:
    st.info("👈 Use the sidebar to load data.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🧬 Analysis", "📉 Visuals", "📄 Report"])

    with tab1:
        st.subheader(f"Current: {format_table_display(st.session_state.dataset_name)}")
        c1, c2 = st.columns(2)
        c1.metric("Samples", len(st.session_state.df))
        c2.metric("Markers", len(st.session_state.df.columns)-1)
        st.dataframe(safe_clean_for_display(st.session_state.df), use_container_width=True, hide_index=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Diversity")
            divs = {col: st.session_state.df[col].nunique() for col in st.session_state.df.columns[1:]}
            st.bar_chart(pd.Series(divs))
        with c2:
            st.subheader("Matrix")
            if st.button("Calc Matrix"):
                d = st.session_state.df.iloc[:, 1:].apply(lambda x: pd.factorize(x)[0])
                dists = pdist(d, 'cityblock')
                ids = st.session_state.df.iloc[:, 0].values
                sim_df = pd.DataFrame(squareform(dists), index=ids, columns=ids)
                st.dataframe(sim_df, height=300)
                st.download_button("Download CSV", sim_df.to_csv().encode('utf-8'), "matrix.csv")

        st.subheader("QC Analysis")
        qc_rows = []
        for col in st.session_state.df.columns[1:]:
            col_data = st.session_state.df[col].astype(str)
            valid = col_data[~col_data.isin(["UNKNOWN", "nan", "None", "NAN"])]
            rate = (len(valid)/len(st.session_state.df))*100
            
            alleles = []
            for v in valid:
                v = str(v).strip()
                if "/" in v: alleles.extend(v.split("/"))
                else: alleles.extend(list(v))
            
            maf, pic = 0.0, 0.0
            if alleles:
                counts = pd.Series(alleles).value_counts()
                maf = counts.iloc[-1]/counts.sum() if len(counts) > 1 else 0.0
                pi = (counts/counts.sum()).values
                pic = 1 - sum(p**2 for p in pi) - sum(2*(pi[i]**2)*(pi[j]**2) for i in range(len(pi)) for j in range(i+1, len(pi)))
            
            qc_rows.append({"Marker": col, "Call%": f"{rate:.1f}", "MAF": f"{maf:.3f}", "PIC": f"{pic:.3f}"})
        st.dataframe(pd.DataFrame(qc_rows), use_container_width=True, hide_index=True)

    with tab3:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Phylogenetic Tree")
            try:
                d = st.session_state.df.iloc[:, 1:].apply(lambda x: pd.factorize(x)[0])
                linkage = sch.linkage(pdist(d, 'cityblock'), method='ward')
                ids = st.session_state.df.iloc[:,0].values
                
                if tree_style == "Rectangular":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#f0f2f6'); ax.set_facecolor('#f0f2f6')
                    dendro = sch.dendrogram(linkage, labels=ids, ax=ax, leaf_rotation=90)
                    for i, d in zip(dendro['icoord'], dendro['dcoord']):
                        x, y = 0.5 * sum(i[1:3]), d[1]
                        if y > 0: ax.text(x, y, f"{y:.1f}", ha='center', va='bottom', fontsize=8, color='black')
                else:
                    fig = plt.figure(figsize=(8, 8))
                    fig.patch.set_facecolor('#f0f2f6'); ax = fig.add_subplot(111, projection='polar')
                    ax.set_facecolor('#f0f2f6')
                    sch.dendrogram(linkage, labels=ids, ax=ax)
                    ax.set_xticklabels([]); ax.set_yticks([])
                st.pyplot(fig)
            except Exception as e: st.error(f"Tree Error: {e}")

        with c2:
            st.subheader("Allele Freq")
            sel_col = st.selectbox("Marker", st.session_state.df.columns[1:])
            if sel_col:
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#f0f2f6'); ax.set_facecolor('#f0f2f6')
                counts = st.session_state.df[sel_col].value_counts()
                bars = ax.bar(counts.index.astype(str), counts.values, color='teal')
                ax.set_ylim(0, counts.max()*1.3)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                            ha='center', va='bottom', color='black', fontweight='bold')
                st.pyplot(fig)

    with tab4:
        if st.button("Generate PDF Report"):
            buffer = io.BytesIO()
            with PdfPages(buffer) as pdf:
                fig = plt.figure(figsize=(8, 11))
                ax = fig.add_subplot(111); ax.axis('off')
                report_time = get_ist_time().strftime("%d-%m-%Y | %I:%M %p")
                txt = f"GENEQUEST REPORT\n{report_time}\nRows: {len(st.session_state.df)}"
                ax.text(0.1, 0.9, txt, fontsize=12, fontfamily='monospace')
                pdf.savefig(fig)
            st.download_button("Download PDF", buffer.getvalue(), "report.pdf", "application/pdf")
