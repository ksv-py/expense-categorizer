import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import plot
import plotly.express as px

# Add path to access custom exception
sys.path.append(str(Path(__file__).resolve().parent))
from exception import CustomException  # ✅ Your custom exception handler
from logger import logging


def get_summary_stats(df: pd.DataFrame):
    """
    Generate summary statistics from the feedback DataFrame.

    Args:
        df (pd.DataFrame): The feedback dataset as a DataFrame.

    Returns:
        dict: A dictionary containing summary statistics.
    """
    try:
        if df.empty:
            logging.warning("Summary stats: Input DataFrame is empty.")
            return {}

        stats = {
            "total_transactions": len(df),
            "total_amount": round(df["Amount"].sum(), 2),
            "unique_categories": df["Category"].nunique(),
            "top_category": df["Category"].value_counts().idxmax(),
            "top_subcategory": df["Subcategory"].value_counts().idxmax(),
            "top_mode": df["Mode"].value_counts().idxmax()
        }

        logging.info("Summary statistics generated successfully.")
        return stats

    except Exception as e:
        logging.error(f"Error generating summary statistics: {e}")
        raise CustomException(e, sys)


def get_visuals(df: pd.DataFrame):
    """
    Generate static visualizations and save as images using matplotlib/seaborn.

    Args:
        df (pd.DataFrame): The feedback dataset as a DataFrame.

    Returns:
        dict: Paths to the saved image plots.
    """
    try:
        if df.empty:
            logging.warning("Static visuals: Input DataFrame is empty.")
            return {}

        charts = {}
        os.makedirs("static/plots", exist_ok=True)

        # Category Distribution
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x="Category", order=df["Category"].value_counts().index)
        plt.title("Spending by Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        cat_chart = "static/plots/category_plot.png"
        plt.savefig(cat_chart)
        charts["category_chart"] = cat_chart
        plt.close()

        # Mode of Payment
        plt.figure(figsize=(6, 4))
        df["Mode"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90)
        plt.title("Payment Mode Distribution")
        plt.ylabel("")
        mode_chart = "static/plots/mode_plot.png"
        plt.savefig(mode_chart)
        charts["mode_chart"] = mode_chart
        plt.close()

        # Monthly Spending Trend
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.to_period("M")
        monthly = df.groupby("Month")["Amount"].sum().reset_index()
        monthly["Month"] = monthly["Month"].astype(str)
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=monthly, x="Month", y="Amount", marker="o")
        plt.title("Monthly Spending Trend")
        plt.xticks(rotation=45)
        trend_chart = "static/plots/trend_plot.png"
        plt.savefig(trend_chart)
        charts["trend_chart"] = trend_chart
        plt.close()

        logging.info("Static visualizations generated and saved.")
        return charts

    except Exception as e:
        logging.error(f"Error generating static visuals: {e}")
        raise CustomException(e, sys)


def get_plotly_visuals(df: pd.DataFrame):
    """
    Generate dynamic Plotly HTML visualizations.

    Args:
        df (pd.DataFrame): The feedback dataset as a DataFrame.

    Returns:
        dict: HTML div strings of Plotly charts.
    """
    try:
        if df.empty:
            logging.warning("Plotly visuals: Input DataFrame is empty.")
            return {}

        df["Date"] = pd.to_datetime(df["Date"])
        charts = {}

        # Category Distribution (Bar)
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig1 = px.bar(cat_counts, x="Category", y="Count", title="Spending by Category")
        fig1.update_layout(xaxis_tickangle=-45)
        charts["category_chart"] = plot(fig1, output_type="div", include_plotlyjs=False)

        # Payment Mode (Pie)
        mode_counts = df["Mode"].value_counts().reset_index()
        mode_counts.columns = ["Mode", "Count"]
        fig2 = px.pie(mode_counts, names="Mode", values="Count", title="Payment Mode Distribution")
        charts["mode_chart"] = plot(fig2, output_type="div", include_plotlyjs=False)

        # Monthly Spending Trend (Line)
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        monthly = df.groupby("Month")["Amount"].sum().reset_index()
        fig3 = px.line(monthly, x="Month", y="Amount", title="Monthly Spending Trend", markers=True)
        charts["trend_chart"] = plot(fig3, output_type="div", include_plotlyjs=False)

        logging.info("Plotly visualizations generated successfully.")
        return charts

    except Exception as e:
        logging.error(f"Error generating Plotly visuals: {e}")
        raise CustomException(e, sys)
