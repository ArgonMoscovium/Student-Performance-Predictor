from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler 
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__) # Initialize Flask application, this object is an WSGI application (communicates bwn webserver & webapplication)

# Define route for the index page, root route renders an 'index.html' template
@app.route('/')
def index():
    return render_template('index.html') # Then create 'index.html' file under 'templates' dir

# Define route for prediction, '/predictdata' route handles GET & POST requests
@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If it's a GET request, simply render the home page containing data fields
        return render_template('home.html') 
    else:
        # If it's a POST request, collect form data into CustomData object then convert to df, make preds and render results
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )

        # Convert data to a df
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        # Make predictions
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")

        # Render the results on home page
        return render_template('home.html', results=results[0])
    
# Run the Flask app, type http://127.0.0.1:5000/ in the address bar (Or your "IPv4 Address":5000), the feature names were not in proper format(corrected it)
# then type '/predictdata'.. try 'crtrl+C' to quit 
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)        # Put app.run(debug=True).. changes will be instantaneously be visible thru the host link 

