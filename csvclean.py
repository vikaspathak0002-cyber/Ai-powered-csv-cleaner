import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI CSV Cleaner", layout="wide")
st.title("📂 AI-Powered CSV Cleaner & Visualizer (Created by Tech Fusion club of Rama University)")

# --- Sidebar ---
st.sidebar.header("⚙ Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

main_tabs = st.tabs([
    "📝 Raw Data", "🔧 Cleaning", "🤖 ML (Impute + Outliers)",
    "📊 Visualizations", "📈 Dashboard"
])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Tab 1: Raw Data ---
    with main_tabs[0]:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(20))
        st.write("Shape:", df.shape)

    # --- Tab 2: Cleaning ---
    with main_tabs[1]:
        st.subheader("Cleaning Process")

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)

        # Clean string columns
        missing_tokens = {"", "na", "n/a", "null", "none", "nan", "-", "—"}
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(missing_tokens, "", regex=True)

        # Convert booleans
        bool_map = {"true": True, "yes": True, "y": True, "1": True,
                    "false": False, "no": False, "n": False, "0": False}
        for col in df.columns:
            df[col] = df[col].replace(bool_map).infer_objects(copy=False)

        # Try parsing dates
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception:
                pass

        # Clean numeric-like strings
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Drop empty rows/cols & duplicates
        df = df.dropna(how="all").dropna(axis=1, how="all").drop_duplicates()

        st.success("Cleaning done ✅")
        st.dataframe(df.head(20))

    # --- Tab 3: ML ---
    with main_tabs[2]:
        st.subheader("ML-Based Cleaning")

        imputation_strategy = st.sidebar.selectbox(
            "Missing Value Imputation Strategy",
            ["mean", "median", "most_frequent"]
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        if len(numeric_cols) > 0:
            imputer_num = SimpleImputer(strategy=imputation_strategy)
            df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
            st.write(f"Numeric cols imputed with {imputer_num.strategy}")

        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
            st.write("Categorical cols imputed with most frequent value")

        # Outlier handling
        outlier_action = st.sidebar.radio("Outlier Handling", ["mark", "remove", "ignore"])
        if not df[numeric_cols].empty:
            model = IsolationForest(contamination=0.05, random_state=42)
            outlier_labels = model.fit_predict(df[numeric_cols])

            if outlier_action == "mark":
                df["_outlier"] = np.where(outlier_labels == -1, "Outlier", "Normal")
            elif outlier_action == "remove":
                df = df[outlier_labels == 1]

            st.write("Outlier Handling:", outlier_action)

        st.success("ML Cleaning Done ✅")
        st.dataframe(df.head(20))

    # --- Tab 4: Visualizations ---
    with main_tabs[3]:
        st.subheader("Choose Visualization")
        chart_type = st.selectbox("Chart Type", ["Histogram", "Boxplot", "Scatterplot", "Correlation Heatmap"])

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if chart_type == "Histogram":
            if len(numeric_cols) > 0:
                col = st.selectbox("Column", numeric_cols)
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), bins=30, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available for histogram.")

        elif chart_type == "Boxplot":
            if len(numeric_cols) > 0:
                col = st.selectbox("Column", numeric_cols)
                fig, ax = plt.subplots()
                sns.boxplot(y=df[col].dropna(), ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available for boxplot.")

        elif chart_type == "Scatterplot":
            if len(numeric_cols) > 1:
                x = st.selectbox("X-axis", numeric_cols)
                y = st.selectbox("Y-axis", numeric_cols)
                fig, ax = plt.subplots()
                ax.scatter(df[x].dropna(), df[y].dropna(), alpha=0.5)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for scatterplot.")

        elif chart_type == "Correlation Heatmap":
            if len(numeric_cols) > 1:
                corr = df.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for correlation heatmap.")

    # --- Tab 5: Dashboard ---
    with main_tabs[4]:
        st.subheader("📈 Summary Dashboard")

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.histplot(numeric_df.iloc[:, 0].dropna(), bins=30, ax=ax)
                st.pyplot(fig)
            with col2:
                fig, ax
