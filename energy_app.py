#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Explorer – Interactive Time-Series Analysis Dashboard

This Streamlit application explores global energy and emissions data
from Our World in Data. It allows interactive selection of:

- Country
- Time range
- Energy / emissions metric

and visualizes:

- Time series
- Autocorrelation (ACF)
- Partial autocorrelation (PACF)
- Rolling diagnostics for stationarity

Data backend: DuckDB
Frontend: Streamlit + Plotly
Statistics: NumPy, SciPy, statsmodels

Author: Axel Behrendt
"""

# -----------------------------------------------------------------------
# Importing the required libraries
# -----------------------------------------------------------------------
   
from pathlib import Path
from typing import Optional, Tuple

import requests
import duckdb
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from statsmodels.tsa.stattools import acf, pacf

# -----------------------------------------------------------------------
# Definition of constants and paths (global)
# -----------------------------------------------------------------------

# -----------------------------
# Base directory
# -----------------------------
try:
    BASE_DIR: Path = Path(__file__).resolve().parent
except NameError:
    BASE_DIR: Path = Path.cwd()  

# -----------------------------
# File folder
# -----------------------------
DATA_DIR: Path = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  

# -----------------------------
# URLs for CSV-files
# -----------------------------
DATA_URL: str = "https://github.com/owid/energy-data/raw/master/owid-energy-data.csv"
CODEBOOK_URL: str = "https://github.com/owid/energy-data/raw/master/owid-energy-codebook.csv"

# -----------------------------
# Local paths
# -----------------------------
CSV_PATH: Path = DATA_DIR / "owid-energy-data.csv"
DB_PATH: Path = DATA_DIR / "energy.duckdb"
CODEBOOK_CSV_PATH: Path = DATA_DIR / "owid-energy-codebook.csv"

# -----------------------------
# Tables in DuckDB
# -----------------------------
TABLE_NAME: str = "energy_data"
CODEBOOK_TABLE: str = "energy_codebook"

# -----------------------------
# For Dashboard / Streamlit
# -----------------------------
DEFAULT_DB_PATH: Path = DB_PATH

# -----------------------------------------------------------------------
#
#                 Section 1: Download and Read the Data
#
# -----------------------------------------------------------------------

def setup_database() -> None:
    """
    Builds a local DuckDB database from the OWID energy dataset.

    - Downloads the CSV files if they do not exist
    - Loads the raw data into a temporary DuckDB table
    - Casts all numeric columns using TRY_CAST for robustness
    - Persists the cleaned table as `energy_data`
    - Imports the OWID codebook as `energy_codebook`

    The use of TRY_CAST ensures that non-numeric artifacts
    never break the pipeline and that all valid numeric
    metrics remain usable for analysis.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # CSV download
    # -----------------------------
    if not CSV_PATH.exists():
        print(f"Loading CSV from {DATA_URL} ...")
        response = requests.get(DATA_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(CSV_PATH, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        print("Download complete.")

    if not CODEBOOK_CSV_PATH.exists():
        response = requests.get(CODEBOOK_URL, timeout=60)
        response.raise_for_status()
        CODEBOOK_CSV_PATH.write_bytes(response.content)

    # -----------------------------
    # DuckDB-Connection
    # -----------------------------
    con = duckdb.connect(str(DB_PATH))

    # -----------------------------
    # Read temporary table
    # -----------------------------
    TMP_TABLE = "tmp_csv"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {TMP_TABLE} AS
        SELECT * FROM read_csv_auto('{CSV_PATH}', SAMPLE_SIZE=-1);
    """)

    # -----------------------------    
    # Retrieve column info
    # -----------------------------
    cols = con.execute(f"PRAGMA table_info('{TMP_TABLE}')").fetchdf()

    # -----------------------------
    # Prepare columns for energy data
    # -----------------------------
    select_expr = []
    for _, row in cols.iterrows():
        name = row["name"]
        if name.lower() not in {"country", "year", "iso_code"}:
            select_expr.append(f"TRY_CAST({name} AS DOUBLE) AS {name}")
        else:
            select_expr.append(name)

    # -----------------------------
    # Create table for energy data
    # -----------------------------   
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT {', '.join(select_expr)}
        FROM {TMP_TABLE};
    """)

    # -----------------------------
    # Create codebook table (metadata)
    # -----------------------------  
    con.execute(f"""
        CREATE OR REPLACE TABLE {CODEBOOK_TABLE} AS
        SELECT
            "column"      AS variable,
            title         AS title,
            description   AS description,
            unit          AS unit,
            source        AS source
        FROM read_csv_auto('{CODEBOOK_CSV_PATH}', SAMPLE_SIZE=-1);
        """)

    con.close()
    print(f"Table '{TABLE_NAME}' created. Numerical metrics should be available.")

# -----------------------------------------------------------------------
#
#                        Section 2: Data Analysis
#
# -----------------------------------------------------------------------

def analyze_data() -> pd.DataFrame:
    """
    Opens the DuckDB database, inspects the table,
    and outputs initial information about the data.
    This function is NOT USED by the streamlit dashboard !!

    Returns:
    -------
    pd.DataFrame
        A small preview of the data (e.g., 10 rows)   
    """

    # -----------------------------
    # Open DuckDB connection
    # -----------------------------      
    con = duckdb.connect(str(DB_PATH))
    
    # -----------------------------    
    # Retrieve & display data info
    # -----------------------------
    row_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    col_info = con.execute(f"PRAGMA table_info({TABLE_NAME})").fetchdf()

    print(f"\nThe table '{TABLE_NAME}' contains {row_count:,} rows.")
    print(f"Number of columns: {len(col_info)}\n")
    
    print("Column overview (first 10 cols):")
    print(col_info.head(10).to_string(index=False))
    print()

    sample_df = con.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5").fetchdf()

    print("Sample data rowws:")
    print(sample_df)
    print()
    
    # -----------------------------  
    # Summarizing SQL-query
    # -----------------------------  
    summary_query: str = f"""       
        SELECT 
            COUNT(*) AS total_rows,
            MIN(year) AS first_year,
            MAX(year) AS last_year
            FROM {TABLE_NAME};
    """
    summary: pd.DataFrame = con.execute(summary_query).fetchdf()   
    print("\n--- Table summary ---")
    print(summary)
 
    # ----------------------------- 
    # Example query
    # -----------------------------
    query: str = f"""                 
        SELECT
            year,
            AVG(primary_energy_consumption) AS avg_consumption
            FROM {TABLE_NAME}
            WHERE primary_energy_consumption IS NOT NULL
            GROUP BY year
            ORDER BY year ASC;
            """

    df: pd.DataFrame = con.execute(query).fetchdf()   

    print("\n--- Example query: Average energy consumption per year ---")
    print(df.head())
    
    con.close()
    print("Connection closed.\n")
    
    return sample_df 

# -----------------------------------------------------------------------
#
#          Section 3: Explorative Analysis and Visualization
#
# -----------------------------------------------------------------------

# ----------------------------------------------------------------
#                          Helper functions
# ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_metric_metadata(
    db_path: Path,
    variable: str
) -> Tuple[str, Optional[str]]:
    """
    Provides a description and unit for an OWID variable.
    """
    con = duckdb.connect(str(db_path))
    
    try:
        result = con.execute(
            """
            SELECT description, unit
            FROM energy_codebook
            WHERE variable = ?
            """,
            [variable]
        ).fetchone()
    finally:
        con.close()

    if result is None:
        return variable.replace("_", " ").title(), None

    return result

# ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_table_list(db_path: Path) -> list[str]:
    """
    Returns the list of tables in the DuckDB file.
    Caching: reduces DB accesses during interaction.
    """

    con = duckdb.connect(str(db_path))
    try:
        df = con.execute("SHOW TABLES").fetchdf()
    finally:
        con.close()
    return df["name"].tolist()

# ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_country_list(db_path: Path, table_name: str) -> list[str]:
    """
    Returns the list of countries in the table.
    Expects a ‘country’ column in the table.
    """
    con = duckdb.connect(str(db_path))
    try:
        q = f"SELECT DISTINCT country FROM {table_name} ORDER BY country"
        df = con.execute(q).fetchdf()
    finally:
        con.close()
    return df["country"].dropna().astype(str).tolist()

# ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_numeric_metrics(db_path: Path, table_name: str) -> list[str]:
    """
    Returns all numeric columns of the table for the sidebar.
    Checks DuckDB type and also attempts to convert strings to numbers.
    """

    con = duckdb.connect(str(db_path))
    df_info = con.execute(f"PRAGMA table_info({table_name})").fetchdf()

    numeric_types = {"INTEGER", "BIGINT", "DOUBLE", "FLOAT", "REAL", "DECIMAL"}
    metrics = []

    for _, row in df_info.iterrows():
        col_name = row["name"]
        col_type = row["type"].upper()

        if col_type in numeric_types:
            metrics.append(col_name)
        else:
            # Fallback: check whether the values can be converted to float
            sample_vals = con.execute(f"SELECT {col_name} FROM {table_name} LIMIT 10").fetchall()
            if all(v[0] is None or isinstance(v[0], (int, float)) or _is_number(v[0]) for v in sample_vals):
                metrics.append(col_name)

    con.close()

    # Exclude columns such as ‘year’ or 'iso_code'
    blacklist = {"year", "iso_code"}
    metrics = [m for m in metrics if m not in blacklist]

    return sorted(metrics)

# ----------------------------------------------------------------

def _is_number(val: object) -> bool:
    """Helper function for fallback check"""
    try:
        float(val)
        return True
    except:
        return False

# ----------------------------------------------------------------

@st.cache_data(show_spinner=True)
def query_time_series(
    db_path: Path,
    table_name: str,
    country: Optional[str],
    metric: str,
    year_min: Optional[int],
    year_max: Optional[int],
) -> pd.DataFrame:
    
    """
    Loads a time series (year, metric) from DuckDB.
    If country is None => aggregates globally (mean across countries).
    """

    con = duckdb.connect(str(db_path))

    if country:
        q = f"""
        SELECT year, {metric}
        FROM {table_name}
        WHERE country = ?
        AND {metric} IS NOT NULL
        ORDER BY year ASC
        """
        params = [country]
    else:
        q = f"""
        SELECT year, AVG({metric}) AS {metric}
        FROM {table_name}
        WHERE {metric} IS NOT NULL
        GROUP BY year
        ORDER BY year ASC
        """
        params = None

    try:
        df = con.execute(q, params).fetchdf() if params else con.execute(q).fetchdf()
    finally:
        con.close()

    if year_min is not None:
        df = df[df["year"] >= year_min]
    if year_max is not None:
        df = df[df["year"] <= year_max]

    df = df.dropna(subset=["year", metric])
    df["year"] = df["year"].astype(int)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df.reset_index(drop=True)

# ----------------------------------------------------------------

def prepare_acf_pacf(series: pd.Series, nlags: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ACF and PACF in a robust way.

    Statsmodels requires:
    nlags < len(series) / 2

    This function enforces that constraint automatically
    to prevent runtime errors on short time series.
    """

    vals = series.dropna().astype(float).values
    n = len(vals)

    if n < 5:
        return np.array([]), np.array([])

    safe_nlags = min(nlags, n // 2 - 1)

    acf_vals = acf(vals, nlags=safe_nlags, fft=True)
    pacf_vals = pacf(vals, nlags=safe_nlags, method="yw")

    return acf_vals, pacf_vals

# ----------------------------------------------------------------
#                      Streamlit UI / App
# ----------------------------------------------------------------

def dashboard_main() -> None:
    """
    Main function for the Streamlit dashboard.
    Reads settings from the sidebar, loads data, and draws the panels.
    
    Visualization:

    This dashboard deliberately favors clarity and analytical readability
    over decorative effects.

    All plots use a consistent dark theme and minimal styling in order to:

    - emphasize structure and temporal dynamics
    - keep visual noise low
    - support direct comparison between panels
    - remain suitable for scientific and educational contexts

    The goal is not aesthetic spectacle, but insight:
    every visual element serves an analytical purpose.
    
    """

    # -------------------------
    # Page config
    # -------------------------
    st.set_page_config(
        page_title="Energy Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -------------------------
    # Sidebar: Data source & selection
    # -------------------------
    st.sidebar.title("Data & Selection")

    db_path_input = st.sidebar.text_input(
        "DuckDB Path", value=str(DEFAULT_DB_PATH)
    )
    db_path = Path(db_path_input)

    try:
        tables = load_table_list(db_path)
    except Exception as exc:
        st.sidebar.error(f"Error while reading the DB: {exc}")
        st.stop()

    ignored_tables = {"energy_codebook", "tmp_energy_raw"}

    filtered_tables = [t for t in tables if t not in ignored_tables]

    if not filtered_tables:
        st.sidebar.error("No valid data tables found.")
        st.stop()

    default_index = 0
    if "energy_data" in filtered_tables:
        default_index = filtered_tables.index("energy_data")

    table_name = st.sidebar.selectbox(
        "Table",
        filtered_tables,
        index=default_index
        )

    try:
        countries = load_country_list(db_path, table_name)
    except Exception:
        countries = []

    country_choice = st.sidebar.selectbox(
        "Country (empty = global)", options=[""] + countries, index=0
    )
    country = country_choice if country_choice != "" else None

    # -------------------------
    # Metric selection (automatically from DB)
    # -------------------------
    st.sidebar.markdown("**Metric**")

    metrics = load_numeric_metrics(db_path, table_name)

    if not metrics:
        st.sidebar.error("No numeric metrics found.")
        st.stop()

    metric = st.sidebar.selectbox(
        "Choose a Metric",
        options=metrics,
        index=0
        )
    
    # For the metadata
    description, unit = get_metric_metadata(db_path, metric)

    y_axis_label = unit if unit else "Value"

    # Select range of years
    df_full = query_time_series(db_path, table_name, country, metric, None, None)
    if df_full.empty:
        st.error("No data found for the selected combination.")
        st.stop()

    year_min = int(df_full["year"].min())
    year_max = int(df_full["year"].max())
    year_range = st.sidebar.slider("Years (Range)", min_value=year_min, max_value=year_max,
                                   value=(year_min, year_max), step=1)
    # ACF/PACF Lags
    nlags = st.sidebar.number_input("ACF/PACF Lags", min_value=10, max_value=100, value=40, step=5)

    # Buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("Load data again"):
    
        # Clear cache if necessary
        load_table_list.clear()
        load_country_list.clear()
        query_time_series.clear()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    # -------------------------
    # Load & prepare data
    # -------------------------
    df = query_time_series(db_path, table_name, country, metric, year_range[0], year_range[1])

    series = df.set_index("year")[metric]

    series_transformed = series.copy()

    rolling_mean = series_transformed.rolling(window=5, min_periods=1).mean()
    rolling_std = series_transformed.rolling(window=5, min_periods=1).std()

    # -------------------------
    # Layout: 2 x 2 Panels
    # -------------------------
    st.title("Energy Explorer — Interactive Dashboard")
    st.markdown(f"**Table:** `{table_name}`  •  **Metric:** `{metric}`  •  **Country:** `{country or 'Global'}`")

    col1, col2 = st.columns([1, 1])  

    # ------------------------------------------------
    # Panel 1: Time series plot
    # ------------------------------------------------
    with col1:
        st.subheader("Time Series")
        if series_transformed.dropna().empty:
            st.warning("No data available.")
        else:
            df_plot = series_transformed.reset_index().rename(columns={metric: "value"})
            fig1 = go.Figure()
            
            # ----------------------------------------
            # Main line
            # ----------------------------------------
            fig1.add_trace(go.Scatter(
                x=df_plot["year"],
                y=df_plot["value"],
                mode="lines+markers",
                line=dict(width=1.5, shape="spline", smoothing=1.2),
                marker=dict(size=2),
                name="Value",
                hovertemplate="<b>%{x}</b><br>Value: %{y:.3f} " + (unit or "") + "<extra></extra>",
            ))
            
            # ----------------------------------------
            # Subtle shading under the curve
            # ----------------------------------------
            fig1.add_traces([
                go.Scatter(
                    x=df_plot["year"],
                    y=df_plot["value"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tozeroy",
                    fillcolor="rgba(0, 200, 255, 0.15)",
                    hoverinfo="skip",
                    showlegend=False
                )
            ])
            
            # ----------------------------------------
            # Optional: Rolling mean as trend line
            # ----------------------------------------
            trend = df_plot["value"].rolling(7, min_periods=1).mean()
            fig1.add_trace(go.Scatter(
                x=df_plot["year"],
                y=trend,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot"),
                name="Trend (Rolling Mean)"
            ))
            
            # ----------------------------------------
            # Layout
            # ----------------------------------------
            fig1.update_layout(
                title="Time Series",
                template="plotly_dark",
                height=420,
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    x=0.01,    
                    y=0.99,    
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(0,0,0,0)",   
                    borderwidth=0
                    ),
                margin=dict(l=20, r=20, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            # ----------------------------------------
            # Axes
            # ----------------------------------------
            fig1.update_xaxes(
                title="Year",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
            )
            
            fig1.update_yaxes(
                title=y_axis_label if unit else "Value",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
            )
            
            st.plotly_chart(fig1, use_container_width=True)

    # ------------------------------------------------
    # Panel 2: ACF 
    # ------------------------------------------------
    with col2:
        st.subheader("Autocorrelation (ACF)")
        acf_vals, pacf_vals = prepare_acf_pacf(series_transformed, nlags=nlags)
        if acf_vals.size == 0:
            st.warning("Time series too short for ACF/PACF.")
        else:
            lags = np.arange(len(acf_vals))
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF", marker_color="#00cc96"))
            fig2.add_trace(go.Scatter(x=lags, y=acf_vals, mode="markers+lines",
                                      marker=dict(color="#00ffcc", size=6), line=dict(width=1)))
            fig2.update_layout(title="Autocorrelation (ACF)", template="plotly_dark", height=420)
            st.plotly_chart(fig2, use_container_width=True)
    
    # ------------------------------------------------
    # Panel 3: Rolling diagnostics 
    # ------------------------------------------------
    with col1:
        st.subheader("Rolling Std vs Mean")
        if series_transformed.dropna().empty:
            st.info("No data for Rolling diagnosis.")
        else:
           
            diag_df = pd.DataFrame({"rolling_mean": rolling_mean, "rolling_std": rolling_std}).dropna()
            if diag_df.empty:
                st.info("Insufficient values for rolling diagnosis.")
            else:
                fig3 = px.scatter(diag_df, x="rolling_mean", y="rolling_std", trendline="lowess",
                                  title="Rolling Std vs Rolling Mean (diagnostic)", labels={
                                      "rolling_mean": "Rolling Average",
                                      "rolling_std": "Rolling Standard Deviation"
                                  })
                fig3.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig3, use_container_width=True)
                
    # ------------------------------------------------
    # Panel 4: PACF 
    # ------------------------------------------------
    with col2:
        st.subheader("Partial Autocorrelation (PACF)")
        if pacf_vals.size == 0:
            st.warning("Time series too short for PACF.")
        else:
            lags = np.arange(len(pacf_vals))
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=lags, y=pacf_vals, name="PACF", marker_color="#636efa"))
            fig4.add_trace(go.Scatter(x=lags, y=pacf_vals, mode="markers+lines",
                                      marker=dict(color="#66b2ff", size=6), line=dict(width=1)))
            fig4.update_layout(title="Partial Autocorrelation (PACF)", template="plotly_dark", height=420)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Note: Calculations are performed locally from the DuckDB. "
        "For larger amounts of data, use the DuckDB tables directly and select restrictive filters."
    )

# -----------------------------------------------------------------------
#
#                               MAIN PROGRAM
#
# -----------------------------------------------------------------------

if __name__ == "__main__":
    setup_database()
    dashboard_main()
    # analyze_data()  # Uncomment for local debugging
