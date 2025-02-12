from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp, chisquare
import uvicorn

# Load the trained model
model = joblib.load('tinotenda_mangarai_loan_default_model.pkl')

# Load the baseline statistics and target distribution
baseline_stats = joblib.load('baseline_stats.pkl')
baseline_target_distribution = joblib.load('baseline_target_distribution.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define the features used by the model
class LoanApplication(BaseModel):
    gender: str
    is_employed: str
    job: str
    location: str
    loan_amount: float
    number_of_defaults: int
    outstanding_balance: float
    interest_rate: float
    age: int
    salary: float
    marital_status: str

# Endpoint for the home page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Loan Default Prediction</title>
        </head>
        <body>
            <h1>Loan Default Prediction Form</h1>
            <form action="/predict" method="post">
                <label for="gender">Gender:</label>
                <input type="text" id="gender" name="gender" placeholder="male, female, other"><br><br>
                
                <label for="is_employed">Employment Status:</label>
                <input type="text" id="is_employed" name="is_employed" placeholder="TRUE or FALSE"><br><br>
                
                <label for="job">Job:</label>
                <input type="text" id="job" name="job" placeholder="e.g. Teacher"><br><br>
                
                <label for="location">Location:</label>
                <input type="text" id="location" name="location" placeholder="e.g. Harare"><br><br>
                
                <label for="loan_amount">Loan Amount:</label>
                <input type="text" id="loan_amount" name="loan_amount" placeholder="e.g. 5000"><br><br>
                
                <label for="number_of_defaults">Number of Defaults:</label>
                <input type="text" id="number_of_defaults" name="number_of_defaults" placeholder="e.g. 2"><br><br>
                
                <label for="outstanding_balance">Outstanding Balance:</label>
                <input type="text" id="outstanding_balance" name="outstanding_balance" placeholder="e.g. 2000"><br><br>
                
                <label for="interest_rate">Interest Rate:</label>
                <input type="text" id="interest_rate" name="interest_rate" placeholder="e.g. 0.15"><br><br>
                
                <label for="age">Age:</label>
                <input type="text" id="age" name="age" placeholder="e.g. 30"><br><br>
                
                <label for="salary">Salary:</label>
                <input type="text" id="salary" name="salary" placeholder="e.g. 50000"><br><br>
                
                <label for="marital_status">Marital Status:</label>
                <input type="text" id="marital_status" name="marital_status" placeholder="married, single, divorced"><br><br>
                
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """

# Function to process data similar to Jupyter notebook
def preprocess_data(data):
    try:
        # Replace SoftwareDeveloper to Software Developer
        data['job'] = data['job'].replace(['SoftwareDeveloper'], 'Software Developer')
        data['job'] = data['job'].replace(['Data Scintist'], 'Data Scientist')

        # replacing Missing Values For 'job' and 'location' columns
        imputer = SimpleImputer(strategy='most_frequent')
        # replace nan in location with not specified 
        data['location'].fillna('not specified', inplace=True)

        # replace nan with unemployed 
        data['job'].fillna('unemployed', inplace=True)

        # Convert categorical variables to numeric labels using Label Encoding
        label_encoders = {}
        for column in ['gender', 'is_employed', 'job', 'location', 'marital_status']:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            label_encoders[column] = encoder

        return data, label_encoders
    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        return None, None

# Function to calculate statistics
def calculate_stats(data):
    return data.describe().transpose()

# Function to detect data drift in features
def detect_feature_drift(incoming_data, baseline_stats):
    drift_results = {}
    incoming_stats = calculate_stats(incoming_data)
    
    for col in incoming_data.columns:
        # Kolmogorov-Smirnov test for distribution differences
        stat, p_value = ks_2samp(incoming_data[col], baseline_stats.loc[col])
        drift_results[col] = p_value < 0.05  # True if drift detected (p-value < 0.05)
    
    return drift_results

# Function to calculate target distribution
def calculate_target_distribution(predictions):
    return pd.Series(predictions).value_counts(normalize=True)

# Function to detect data drift in target class
def detect_target_drift(predictions, baseline_target_distribution):
    incoming_target_distribution = calculate_target_distribution(predictions)
    # Align both distributions to have the same index (labels)
    incoming_target_distribution = incoming_target_distribution.reindex(baseline_target_distribution.index, fill_value=0)
    # Chi-square test for distribution differences
    stat, p_value = chisquare(incoming_target_distribution, baseline_target_distribution)
    return p_value < 0.05  # True if drift detected (p-value < 0.05)

# Endpoint for prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    gender: str = Form(...),
    is_employed: str = Form(...),
    job: str = Form(...),
    location: str = Form(...),
    loan_amount: str = Form(...),
    number_of_defaults: str = Form(...),
    outstanding_balance: str = Form(...),
    interest_rate: str = Form(...),
    age: str = Form(...),
    salary: str = Form(...),
    marital_status: str = Form(...)
):
    try:
        # Convert input data to appropriate types
        loan_amount = float(loan_amount)
        number_of_defaults = int(number_of_defaults)
        outstanding_balance = float(outstanding_balance)
        interest_rate = float(interest_rate)
        age = int(age)
        salary = float(salary)

        # Prepare the input data
        input_data = pd.DataFrame([[
            gender, is_employed, job, location, loan_amount, number_of_defaults,
            outstanding_balance, interest_rate, age, salary, marital_status
        ]], columns=[
            'gender', 'is_employed', 'job', 'location', 'loan_amount', 'number_of_defaults',
            'outstanding_balance', 'interest_rate', 'age', 'salary', 'marital_status'
        ])

        # Process the input data
        processed_data, _ = preprocess_data(input_data)
        if processed_data is None:
            return {"error": "Error in data preprocessing"}

        # Select numerical columns for scaling
        numerical_cols = ['loan_amount', 'outstanding_balance', 'salary']

        # Initialize the scaler
        scaler = StandardScaler()

        # Fit and transform the scaler on the numerical columns
        processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])

        # Make a prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)

        # Interpret prediction result
        if prediction[0] == 1:
            result = "Will default on Loan"
        else:
            result = "Will Not Default on Loan"

        # Detect data drift in features
        feature_drift_results = detect_feature_drift(processed_data, baseline_stats)

        # Detect data drift in target class
        target_drift = detect_target_drift(prediction, baseline_target_distribution)

        # Format the drift results for display
        feature_drift_message = "<br>".join(
            [f"{col}: {'Drift detected' if drift else 'No drift'}" for col, drift in feature_drift_results.items()]
        )
        target_drift_message = "Drift detected" if target_drift else "No drift"

        return f"""
        <html>
            <head>
                <title>Loan Default Prediction</title>
            </head>
            <body>
                <h1>Loan Default Prediction Result</h1>
                <div style="border:1px solid black; padding:10px; margin-bottom:10px;">
                    <strong>Prediction:</strong> {result}
                </div>
                <div style="border:1px solid black; padding:10px;">
                    <strong>Probability of Default:</strong> {float(probability[0][1])}
                </div>
                <br><br>
                <h2>Data Drift Detection</h2>
                <div style="border:1px solid black; padding:10px; margin-bottom:10px;">
                    <strong>Feature Drift:</strong><br> {feature_drift_message}
                </div>
                <div style="border:1px solid black; padding:10px;">
                    <strong>Target Drift:</strong> {target_drift_message}
                </div>
                <br><br>
                <a href="/">Back to Form</a>
            </body>
        </html>
        """
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"error": "An error occurred during prediction"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
