import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🧹 AI Powered CSV Cleaner & Explorer")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------- Preprocessing ----------
    def preprocess_dataframe(df):
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                # remove % and commas
                df_clean[col] = df_clean[col].astype(str).str.replace('%', '', regex=True).str.replace(',', '', regex=True)
                # try to convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")
        return df_clean

    df = preprocess_dataframe(df)

    # ---------- Sidebar ----------
    st.sidebar.header("🛠 Options")

    # Cleaning
    if st.sidebar.checkbox("🔄 Clean Missing Values"):
        strategy = st.sidebar.radio("Select Strategy:", ["Replace Strings with NaN", "Numeric → Mean", "Drop Rows"])
        if strategy == "Replace Strings with NaN":
            df = df.applymap(lambda x: np.nan if isinstance(x, str) and not x.replace('.', '', 1).isdigit() else x)
        elif strategy == "Numeric → Mean":
            for col in df.select_dtypes(include=np.number).columns:
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == "Drop Rows":
            df = df.dropna()

    # Show Data
    if st.sidebar.checkbox("📊 Show Data"):
        st.dataframe(df.head(20))

    # Info
    if st.sidebar.checkbox("ℹ️ Show Info"):
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    # Describe
    if st.sidebar.checkbox("📈 Describe Data"):
        st.write(df.describe())

    # Columns
    if st.sidebar.checkbox("📋 Show Columns"):
        st.write(list(df.columns))

    # ---------- Graph Visualization ----------
    if st.sidebar.checkbox("📊 Graph Visualization"):
        st.subheader("📊 Data Visualization (Seaborn)")

        # ✅ User choice: Full vs Sample %
        data_choice = st.radio("Select Data for Graph:", ["Use Full Data", "Use Sample %"])
        if data_choice == "Use Sample %":
            sample_size = st.slider("Select sample percentage (%)", 5, 100, 10)  # default 10%
            df_plot = df.sample(frac=sample_size/100, random_state=42)
            st.info(f"Showing graph on **{sample_size}% sample data** ({len(df_plot)} rows out of {len(df)})")
        else:
            df_plot = df

        numeric_cols = df_plot.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            x_axis = st.selectbox("Select X-axis", options=numeric_cols)
            y_axis = st.selectbox("Select Y-axis (Optional)", options=[None] + numeric_cols)

            graph_type = st.radio("Select Graph Type", ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"])

            if graph_type == "Line Plot" and y_axis:
                fig, ax = plt.subplots()
                sns.lineplot(x=df_plot[x_axis], y=df_plot[y_axis], marker="o", color="blue", ax=ax)
                ax.set_title(f"Line Plot of {y_axis} vs {x_axis}")
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                st.pyplot(fig)

            elif graph_type == "Bar Plot" and y_axis:
                fig, ax = plt.subplots()
                sns.barplot(x=df_plot[x_axis], y=df_plot[y_axis], palette="viridis", ax=ax)
                ax.set_title(f"Bar Plot of {y_axis} vs {x_axis}")
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                st.pyplot(fig)

            elif graph_type == "Scatter Plot" and y_axis:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df_plot[x_axis], y=df_plot[y_axis], color="red", ax=ax)
                ax.set_title(f"Scatter Plot of {y_axis} vs {x_axis}")
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                st.pyplot(fig)

            elif graph_type == "Histogram":
                fig, ax = plt.subplots()
                sns.histplot(df_plot[x_axis].dropna(), bins=30, kde=True, color="green", ax=ax)
                ax.set_title(f"Histogram of {x_axis}")
                ax.set_xlabel(x_axis)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            elif graph_type == "Box Plot":
                fig, ax = plt.subplots()
                sns.boxplot(y=df_plot[x_axis], palette="Set2", ax=ax)
                ax.set_title(f"Box Plot of {x_axis}")
                ax.set_ylabel(x_axis)
                st.pyplot(fig)

            elif graph_type == "Correlation Heatmap":
                corr = df_plot.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)

        else:
            st.warning("⚠️ No numeric columns available for plotting.")

    # ---------- Download Cleaned CSV ----------
    st.sidebar.markdown("### 💾 Download")
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

