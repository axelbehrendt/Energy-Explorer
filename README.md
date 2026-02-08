# Energy Explorer – Interactive Time-Series Analysis Dashboard

**Energy Explorer** is an interactive data analysis dashboard for exploring
global energy consumption, electricity generation, and emissions data
from **Our World in Data**.

The application focuses on **time-series analysis** and helps users
understand trends, temporal dependencies, and the statistical structure
of energy-related indicators across countries.

---

## 🎯 Project Goals

This project was built to:

- Explore long-term developments in global energy and emissions data
- Compare time series across countries and metrics
- Analyze temporal dependencies using **ACF (Autocorrelation Function)** and  
  **PACF (Partial Autocorrelation Function)**
- Assess time-series suitability for modeling approaches (e.g. ARIMA)
- Demonstrate a clean, reproducible, portfolio-ready data application

---

## 🧠 Key Features

- **Interactive selection**
  - Country (or global aggregation)
  - Time range
  - Energy / emissions metric

- **Four analytical panels**
  1. Time series visualization
  2. Autocorrelation (ACF)
  3. Rolling standard deviation vs. rolling mean (stationarity diagnostics)
  4. Partial autocorrelation (PACF)

- **Automatic metadata handling**
  - Units and descriptions loaded from the OWID codebook
  - Dynamic axis labeling based on the selected metric

---

## 🛠️ Technical Stack

| Component        | Technology |
|------------------|------------|
| Data source      | Our World in Data (CSV) |
| Storage & queries| DuckDB |
| Backend logic    | Python |
| Visualization    | Streamlit + Plotly |
| Statistics       | NumPy, statsmodels |
| Deployment       | Streamlit Community Cloud |

---

## 🗂️ Project Structure

├── energy_app.py # Main Streamlit application
├── data/ # Local data (ignored in Git)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

---

## 🚀 How to Run Locally

### 1. Clone the repository
git clone https://github.com/<your-username>/energy-explorer.git
cd energy-explorer

### 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Start the application
streamlit run energy_app.py

## 📊 Data Source
Our World in Data – Energy Dataset
https://github.com/owid/energy-data
The data is downloaded automatically on first run and stored locally.
Metadata (units and descriptions) are read from the official OWID codebook.

## 📄 License
This project is released under the MIT License.

## 👤 Author
Axel Behrendt
Data analysis · Time-series · Scientific Python  
Parts of the development as well as the debugging were supported by ChatGPT 5.2.


