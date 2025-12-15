import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import math
import os
import re
import numpy as np
import sqlite3
import datetime
import io
from PIL import Image

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="GeneQuest",
    page_icon="🧬"
)

# Global Constants
DB_NAME = "genequest_db.db"
CROPS = ["Rice", "Cotton", "Maize", "Wheat", "Soybean"]

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* 1. Page Background */
    .stApp { background-color: #f0f2f6; }

    /* 2. Button Styling */
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        font-weight: bold; 
        background-color: white; 
        border: 1px solid #ccc;
    }
    
    /* 3. Main Title Gradient */
    h1 { 
        background: linear-gradient(to right, #1e3c72, #5a3f99, #e67e22);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        font-weight: 800;
    }
    
    /* 4. Credits Box Style */
    .credits {
        font-size: 0.85rem;
        color: #444;
        padding: 10px 0;
        margin-bottom: 20px;
        line-height: 1.5;
        font-weight: 500;
    }
    .credits b {
        font-weight: 700;
        color: #0e76a8;
    }
    
    /* 5. Metrics Text Size */
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    
    /* 6. Sidebar Spacing */
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'df' not in st.session_state: st.session_state.df = None
if 'dataset_name' not in st.session_state: st.session_state.dataset_name = "None"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_db_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_db_connection()
    conn.close()

def format_table_display(table_name):
    """Converts DB table names to readable AM/PM format."""
    try:
        parts = table_name.split('_')
        if len(parts) < 3: return table_name
        time_part = parts[-1]
        date_part = parts[-2]
        user_text = "_".join(parts[1:-2])
        dt = datetime.datetime.strptime(f"{date_part}_{time_part}", "%d-%m-%Y_%H-%M-%S")
        nice_time = dt.strftime("%d-%m-%Y %I:%M:%S %p")
        return f"{user_text} ({nice_time})"
    except:
        return table_name

def safe_clean_for_display(df):
    """
    Prevents Minified React Error #185 by forcing EVERYTHING to strings
    and resetting the index before display.
    """
    if df is None: return pd.DataFrame()
    df_clean = df.copy()
    # Force every single cell to be a string
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str)
    # Reset index to ensure uniqueness
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def standardize_columns(df):
    """Fixes column names immediately on load."""
    new_cols = []
    for i, col in enumerate(df.columns):
        if i == 0: 
            new_cols.append("Sample")
            continue
        clean = col.strip()
        if re.search(r'(marler|mkr|marker)', clean, re.IGNORECASE):
            match = re.search(r'\d+$', clean)
            if match: clean = f"Marker-{match.group()}"
        new_cols.append(clean)
    df.columns = new_cols
    return df

def audit_clean_dataframe(df, strategy):
    """Cleans headers and values based on selected strategy."""
    try:
        # 1. Standardize Headers
        df = standardize_columns(df)
        
        # 2. Force Data to Strings (Crucial Step)
        for col in df.columns[1:]:
             df[col] = df[col].astype(str)
             df[col] = df[col].apply(lambda x: x.upper().strip())
             # Convert literal 'nan' strings back to actual numpy nan for filtering
             df[col] = df[col].replace(['NAN', 'NONE', ''], np.nan)
        
        # 3. Handle Missing Strategy
        # We look for actual np.nan now
        missing_count = df.isnull().sum().sum()
        
        if strategy == "Missing values: remove the rows":
            if missing_count > 0:
                df.dropna(inplace=True)
                st.toast(f"Removed rows with {missing_count} missing values.")
        else:
            # Fill with UNKNOWN
            df.fillna("UNKNOWN", inplace=True)
            if missing_count > 0:
                st.toast(f"Filled {missing_count} missing values with 'UNKNOWN'.")
            
        # 4. Final Polish: Ensure everything is string again
        df = df.fillna("UNKNOWN")
        return df
    except Exception as e:
        st.error(f"Cleaning Error: {e}")
        return df

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=160)
    
    st.markdown("""
    <div class="credits">
        <b>Designed by</b><br>
        Dr. Santosh Kumar Dodda<br>
        Bioseed Research India
    </div>
    """, unsafe_allow_html=True)

    st.title("🧬 Controls")

    # --- 1. INPUT DATA ---
    st.header("1. Input Data")
    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True, type="csv")

    if uploaded_files:
        if st.button("📥 Load CSVs"):
            frames = []
            for f in uploaded_files:
                try:
                    d = pd.read_csv(f)
                    d = standardize_columns(d) # Standardize BEFORE merge
                    # Force all to string to prevent type mismatch during concat
                    d = d.astype(str) 
                    frames.append(d)
                except:
                    st.error(f"Failed to read: {f.name}")
            
            if frames:
                # Merge
                merged_df = pd.concat(frames, ignore_index=True)
                st.session_state.df = merged_df
                st.session_state.dataset_name = "Unsaved_CSV_Data"
                st.success(f"Loaded {len(merged_df)} rows.")

    # --- 2. DATA PREP ---
    st.markdown("---")
    st.markdown("###### Data Prep Actions") 
    
    audit_clicked = st.button("🧹 Audit & Clean Data")

    missing_strategy = st.radio(
        "Missing Data Strategy:", 
        ["Missing values: change to unknown", "Missing values: remove the rows"],
        label_visibility="visible"
    )

    st.markdown("") 
    if st.button("🔄 Reset / Clear View"):
        st.session_state.df = None
        st.session_state.dataset_name = "None"
        st.rerun()

    if audit_clicked:
        if st.session_state.df is not None:
            st.session_state.df = audit_clean_dataframe(st.session_state.df, missing_strategy)
            st.success("Data Cleaned!")
        else:
            st.warning("Load data first.")

    # --- 3. SAVE TO DB ---
    st.markdown("---")
    st.header("2. Save to Database")
    with st.form("save_form"):
        target_crop = st.selectbox("Crop Folder", CROPS)
        save_name = st.text_input("Dataset Name (e.g. Experiment_1)")
        submit_save = st.form_submit_button("💾 Save to DB")

        if submit_save:
            if st.session_state.df is None:
                st.error("No data!")
            elif not save_name:
                st.error("Enter a name.")
            else:
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', save_name)
                timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                table_name = f"{target_crop}_{clean_name}_{timestamp}"
                try:
                    conn = get_db_connection()
                    st.session_state.df.to_sql(table_name, conn, if_exists="fail", index=False)
                    conn.close()
                    st.session_state.dataset_name = table_name
                    st.success(f"Saved!")
                except Exception as e:
                    st.error(f"DB Error: {e}")

    # --- 4. LOAD FROM DB ---
    st.markdown("---")
    st.header("3. Load from Database")
    load_crop = st.selectbox("Browse Folder", CROPS, key="load_crop_box")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '{load_crop}_%';")
        raw_tables = [row[0] for row in cursor.fetchall()]
    except:
        raw_tables = []
    conn.close()

    raw_tables.sort(reverse=True)
    table_options = {format_table_display(t): t for t in raw_tables}

    selected_displays = st.multiselect("Select Dataset(s)", list(table_options.keys()))

    if st.button("📂 Load from DB"):
        if not selected_displays:
            st.error("Select a dataset.")
        else:
            frames = []
            conn = get_db_connection()
            for dname in selected_displays:
                real_name = table_options[dname]
                try:
                    d = pd.read_sql(f'SELECT * FROM "{real_name}"', conn)
                    d = d.astype(str) # Force string immediately
                    d = standardize_columns(d)
                    frames.append(d)
                except Exception as e:
                    st.error(f"Error: {e}")
            conn.close()
            
            if frames:
                merged_df = pd.concat(frames, ignore_index=True)
                st.session_state.df = merged_df
                st.session_state.dataset_name = f"Merged ({len(frames)} sets)"
                st.success(f"Loaded {len(st.session_state.df)} total rows.")

    # --- 5. VISUALIZATION ---
    st.markdown("---")
    st.header("4. Visualization")
    tree_style = st.selectbox("Tree Style", ["Rectangular", "Circular"])

# ==========================================
# 4. MAIN DASHBOARD CONTENT
# ==========================================

st.title("GeneQuest")

if st.session_state.df is None:
    st.info("👈 Use the sidebar controls to Load Data.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data View", "🧬 Analysis & QC", "📉 Visualizations", "📄 Report"])

    # --- TAB 1: DATA VIEW ---
    with tab1:
        display_name = format_table_display(st.session_state.dataset_name)
        st.subheader(f"Current Data: {display_name}")
        
        col1, col2 = st.columns(2)
        col1.metric("Total Samples", len(st.session_state.df))
        col2.metric("Total Markers", len(st.session_state.df.columns)-1)
        
        # USE SAFE DISPLAY FUNCTION to avoid React Error #185
        st.dataframe(safe_clean_for_display(st.session_state.df), use_container_width=True, hide_index=True)

    # --- TAB 2: ANALYSIS & QC ---
    with tab2:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Diversity Metrics")
            divs = {col: st.session_state.df[col].nunique() for col in st.session_state.df.columns[1:]}
            avg = sum(divs.values())/len(divs) if divs else 0
            st.info(f"**Average Alleles per Marker:** {avg:.2f}")
            st.bar_chart(pd.Series(divs))
            
        with c2:
            st.subheader("Genetic Distance Matrix")
            if st.button("Calculate Matrix"):
                try:
                    data = st.session_state.df.iloc[:, 1:].copy()
                    for c in data.columns: data[c] = pd.factorize(data[c])[0]
                    dists = pdist(data, 'cityblock')
                    mat = squareform(dists)
                    
                    ids = st.session_state.df.iloc[:, 0].astype(str).values
                    sim_df = pd.DataFrame(mat, index=ids, columns=ids)
                    
                    st.dataframe(sim_df, height=300)
                    csv = sim_df.to_csv().encode('utf-8')
                    st.download_button("Download CSV", csv, "matrix.csv", "text/csv")
                except Exception as e:
                    st.error(f"Matrix Error: {e}")

        st.markdown("---")
        st.subheader("QC Analysis")
        qc_rows = []
        for col in st.session_state.df.columns[1:]:
            # SAFE: Ensure column is string
            col_data = st.session_state.df[col].astype(str)
            # Filter valid
            valid = col_data[~col_data.isin(["UNKNOWN", "nan", "None", "NAN"])]
            
            rate = (len(valid)/len(st.session_state.df))*100
            
            alleles = []
            for v in valid:
                # CRITICAL FIX: Explicit string conversion inside loop
                v_str = str(v).strip()
                if "/" in v_str: alleles.extend(v_str.split("/"))
                else: alleles.extend(list(v_str))
            
            if alleles:
                counts = pd.Series(alleles).value_counts()
                maf = counts.iloc[-1]/counts.sum() if len(counts) > 1 else 0.0
                pi = (counts/counts.sum()).values
                pic = 1 - sum(p**2 for p in pi) - sum(2*(pi[i]**2)*(pi[j]**2) for i in range(len(pi)) for j in range(i+1, len(pi)))
            else: maf, pic = 0.0, 0.0

            qc_rows.append({"Marker": col, "Call Rate %": f"{rate:.1f}", "MAF": f"{maf:.3f}", "PIC": f"{pic:.3f}"})
        st.dataframe(pd.DataFrame(qc_rows), use_container_width=True, hide_index=True)

    # --- TAB 3: VISUALIZATIONS ---
    with tab3:
        c1, c2 = st.columns([2, 1])
        
        # Tree
        with c1:
            st.subheader("Phylogenetic Tree")
            try:
                data = st.session_state.df.iloc[:, 1:].copy()
                for c in data.columns: data[c] = pd.factorize(data[c])[0]
                linkage = sch.linkage(pdist(data, 'cityblock'), method='ward')
                ids = st.session_state.df.iloc[:,0].values
                
                if tree_style == "Rectangular":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#f0f2f6')
                    ax.set_facecolor('#f0f2f6')
                    dendro = sch.dendrogram(linkage, labels=ids, ax=ax, leaf_rotation=90)
                    
                    # Manual Branch Labels (Fixes 'datavalues' error)
                    for i, d in zip(dendro['icoord'], dendro['dcoord']):
                        x = 0.5 * sum(i[1:3])
                        y = d[1]
                        if y > 0:
                            ax.text(x, y, f"{y:.1f}", ha='center', va='bottom', fontsize=8, color='black')
                    ax.set_ylabel("Genetic Distance")
                else:
                    fig = plt.figure(figsize=(8, 8))
                    fig.patch.set_facecolor('#f0f2f6')
                    ax = fig.add_subplot(111, projection='polar')
                    ax.set_facecolor('#f0f2f6')
                    sch.dendrogram(linkage, labels=ids, ax=ax)
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Tree Error: {e}")

        # Allele Frequency
        with c2:
            st.subheader("Allele Frequency")
            cols = st.session_state.df.columns[1:]
            sel_col = st.selectbox("Select Marker", cols)
            if sel_col:
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#f0f2f6')
                ax.set_facecolor('#f0f2f6')
                counts = st.session_state.df[sel_col].value_counts()
                
                # Plot bars
                bars = ax.bar(counts.index.astype(str), counts.values, color='teal')
                
                ax.set_title(sel_col, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylim(0, counts.max() * 1.3)
                
                # CRITICAL FIX: Manual text loop instead of bar_label
                # This works on ALL python versions and fixes "AttributeError"
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', 
                            color='black', fontweight='bold', fontsize=10)
                st.pyplot(fig)

    # --- TAB 4: REPORT ---
    with tab4:
        st.subheader("Download Report")
        if st.button("Generate PDF"):
            buffer = io.BytesIO()
            with PdfPages(buffer) as pdf:
                fig = plt.figure(figsize=(8, 11))
                ax = fig.add_subplot(111); ax.axis('off')
                report_time = datetime.datetime.now().strftime("%d-%m-%Y | %I:%M %p")
                txt = f"GENEQUEST REPORT\n\nDate: {report_time}\nDataset: {st.session_state.dataset_name}\nRows: {len(st.session_state.df)}"
                ax.text(0.1, 0.9, txt, fontsize=12, fontfamily='monospace')
                pdf.savefig(fig)
            buffer.seek(0)
            st.download_button("⬇️ Download PDF Report", buffer, "report.pdf", "application/pdf")

# Init DB on first load
init_db()