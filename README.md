# Student Exam Performance Predictior

This project is a machine learning-driven web application that predicts a student’s math score based on various features like **gender, parental education, lunch type**, and more. The project demonstrates a complete machine learning workflow, including data ingestion, preprocessing, model training, and deployment, using cutting-edge tools and techniques like CI/CD pipelines, containerization, and automated testing.


## Table of Contents
- Motivation
- Project Architecture
- Tools & Technologies
- Data Pipeline Overview
- Model Pipeline Overview
- Code Snippets
- Installation
- Usage
- Running Tests
- Future Enhancements
- License


## Project Architecture
The project architecture is composed of several critical components:
1. Data Pipeline: Handles ingestion, transformation, and logging.
2. Model Pipeline: Trains multiple models, performs hyperparameter tuning, and selects the best-performing model.
3. Containerized Web App: A Flask web application running inside a Docker container, ensuring consistency across environments.
4. Testing: pytest is used for ensuring the integrity of all pipelines and components.
5. CI/CD Pipeline: Automated workflows with GitHub Actions that test and deploy the model pipeline and web app.


## Tools & Technologies
1. Pandas, Numpy: For data manipulation and numerical computations.
2. Machine Learning Libraries: 
- scikit-learn: For preprocessing, model training, and evaluation. 
- XGBoost, CatBoost: For advanced model building and boosting.
3. Logging & Exception Handling: Custom logging and exception handling to trace issues during execution.
4. Flask: A lightweight web framework to serve predictions via a web interface.
5. Pytest: Test framework for ensuring code reliability through unit tests and functional tests. It will test the entire pipeline including data ingestion, transformation, model training and Flask API endpoints.
6. CI/CD: GitHub Actions for automating the testing and deployment process.
7. Docker: The entire Flask app and model pipelines are containerized for easier deployment and scalability.


## Data Pipeline Overview
1. Data Ingestion:
- Reads raw data from CSV files.
- Splits data into training and testing sets (80/20 split).
- Saves raw, training, and testing datasets.
- Logs each step for debugging and monitoring.
2. Data Transformation:
- Handles missing data using `SimpleImputer`.
- Applies `OneHotEncoder` to categorical columns.
- Standardizes numerical features using `StandardScaler`.
- Combines preprocessing pipelines into a single `ColumnTransformer`.
- Saves the preprocessor object for future use in inference.


## Model Pipeline Overview
1. Model Training:
- Trains multiple machine learning models (Random Forest, Gradient Boosting, XGBoost, etc.).
- Performs hyperparameter tuning to optimize models.
- Evaluates models based on R² score and selects the best-performing model.
- Saves the trained model as a serialized `.pkl` file for future predictions.
2. Model Evaluation:
- Evaluates the models on the test set using performance metrics like R² score.
- Logs all model evaluation results to compare different algorithms.


## Installation
1. Clone the repository:
`git clone <repository_url>`    
`cd <project_folder>`

2. Install dependencies:
`pip install -r requirements.txt`

3. Build and run the Docker container:
`docker build -t exam-predictor .`
`docker run -p 5000:5000 exam-predictor`


## Usage
Once the Flask web application is up and running:
1. Open your browser and navigate to `http://localhost:5000`.
2. Enter the required inputs (e.g., gender, parental education, scores).
3. Click on Predict to see the predicted math score.


## Future Enhancements
1. Advanced Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV for more thorough tuning of models.
2. Model Explainability: Incorporate SHAP or LIME for explainable AI to understand feature importance in predictions.
3. Model Deployment: Deploy the application to cloud services like AWS or Heroku.
4. Frontend Improvements: Enhance the web interface for a more interactive user experience.