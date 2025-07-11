import sys
import os
from pathlib import Path
from datetime import datetime

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.services.analytics_service import get_summary_stats, get_visuals, get_plotly_visuals
from src.logger import logging
from src.exception import CustomException

# Load environment variables
load_dotenv()

# Initialize the scheduler
scheduler = BackgroundScheduler()

# Add component path
sys.path.append(str(Path(__file__).resolve().parent / "src" / "components"))

# Flask app setup
app = Flask(__name__)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client[os.getenv("MONGO_DB_NAME", "expense-tracker")]
collection = db[os.getenv("MONGO_COLLECTION_NAME", "feedback")]


@app.route('/')
def home():
    """Render the homepage."""
    logging.info("Homepage accessed")
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request and show feedback form with prediction."""
    try:
        input_data = CustomData(
            mode=request.form.get('Mode'),
            subcategory=request.form.get('Subcategory'),
            amount=float(request.form.get('Amount'))
        )
        input_df = input_data.get_df()
        pred_obj = PredictPipeline()
        prediction = pred_obj.predict(input_df)
        result = prediction[0]

        logging.info(f"Prediction made: {result} for input {input_df.to_dict(orient='records')[0]}")
        return render_template('feedback.html', data=input_df.to_dict(orient='records')[0], prediction=result)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise CustomException(e,sys)


@app.route('/feedback', methods=['POST'])
def feedback():
    """Store user feedback (corrected category) into MongoDB."""
    try:
        actual_category = request.form.get('prediction').title() if request.form.get('feedback_correct') == 'yes' else request.form.get('correct_category').title()

        feedback_doc = {
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Mode': request.form.get('mode'),
            'Category': actual_category,
            'Subcategory': request.form.get('subcategory'),
            'Note': None,
            'Amount': float(request.form.get('amount')),
            'Income/Expense': 'Expense',
            'Currency': 'INR'
        }

        collection.insert_one(feedback_doc)
        logging.info(f"Feedback stored: {feedback_doc}")
        return redirect(url_for('home'))

    except Exception as e:
        logging.error(f"Error during feedback submission: {e}")
        raise CustomException(e,sys)


@app.route('/analytics')
def analytics():
    """Render analytics page with charts and statistics from MongoDB data."""
    try:
        feedback_df = pd.DataFrame(list(collection.find()))
        if feedback_df.empty:
            logging.warning("No feedback data found for analytics.")
            return render_template('analytics.html', stats={}, charts=[], plotly_charts=[])

        stats = get_summary_stats(feedback_df)
        charts = get_visuals(feedback_df)
        plotly_charts = get_plotly_visuals(feedback_df)

        logging.info("Analytics generated successfully.")
        return render_template('analytics.html', stats=stats, charts=charts, plotly_charts=plotly_charts)

    except Exception as e:
        logging.error(f"Error generating analytics: {e}")
        raise CustomException(e,sys)


@app.route('/history', methods=['GET', 'POST'])
def history():
    """Render transaction history with optional filtering by date, mode, and category."""
    try:
        feedback_df = pd.DataFrame(list(collection.find()))
        if feedback_df.empty:
            logging.warning("No feedback data found for history.")
            return render_template('history.html', data=[], filters={})

        feedback_df['Date'] = pd.to_datetime(feedback_df['Date'], errors='coerce')
        feedback_df.dropna(subset=['Date'], inplace=True)
        feedback_df['Date'] = feedback_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        categories = sorted(feedback_df['Category'].dropna().unique())
        modes = sorted(feedback_df['Mode'].dropna().unique())

        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        selected_category = request.form.get('category')
        selected_mode = request.form.get('mode')

        filtered_df = feedback_df.copy()

        if start_date:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Date']) >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Date']) <= pd.to_datetime(end_date)]
        if selected_category and selected_category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        if selected_mode and selected_mode != 'All':
            filtered_df = filtered_df[filtered_df['Mode'] == selected_mode]

        filtered_df = filtered_df.sort_values(by='Date', ascending=False)

        logging.info("History page rendered successfully.")
        return render_template(
            'history.html',
            categories=categories,
            modes=modes,
            data=filtered_df.to_dict(orient='records'),
            filters={
                'categorie': selected_category,
                'mode': selected_mode,
                'start_date': start_date or '',
                'end_date': end_date or '',
                'selected_category': selected_category or 'all',
                'selected_mode': selected_mode or 'all'
            }
        )

    except Exception as e:
        logging.error(f"Error displaying history: {e}")
        raise CustomException(e,sys)

def schedule_model_training():
    """Initiates model retrainig."""
    try:
        logging.info("Scheduled retraining started.")
        TrainPipeline().initiate_train_pipeline()
        logging.info("Scheduled retraining completed.")
    except Exception as e:
        logging.error(f"Scheduled retraining failed: {str(e)}")
        raise CustomException(e,sys)

# Schedule retrainig every 6 hours
scheduler.add_job(schedule_model_training, 'interval', hours = 6)

# Start the scheduler
scheduler.start()

# schedule_model_training()

# Shut down scheduler
atexit.register(lambda: scheduler.shutdown())


# Uncomment if using locally 

# if __name__ == '__main__':
#     logging.info("Starting Flask server...")
#     app.run(debug=True) 
