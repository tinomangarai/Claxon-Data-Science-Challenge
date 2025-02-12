# Claxon Challenge 

## Overview
This FastAPI application predicts the likelihood of a loan applicant defaulting on their loan. Additionally, it detects data drift in the features and target distribution to monitor the performance and reliability of the model over time.

## Approach
Data Preprocessing
The preprocessing steps are critical to ensure the input data is in the correct format and consistent with the data used during model training.

## Data Cleaning:

Specific job titles like SoftwareDeveloper are replaced with Software Developer.
Missing values in the job and location columns are filled with 'unemployed' and 'not specified', respectively.
## Label Encoding:

Categorical variables (gender, is_employed, job, location, marital_status) are converted to numerical labels using pre-fitted LabelEncoder objects. This ensures consistency with the training phase.

## Scaling:

Numerical features (loan_amount, outstanding_balance, salary) are scaled using a pre-fitted StandardScaler to ensure that the model input is normalized, matching the distribution of the training data.

## Prediction
The processed input data is fed into a pre-trained model to predict the probability of loan default.

## Data Drift Detection
To maintain the model's performance over time, the method monitors data drift in both features and the target distribution.

## Feature Drift Detection:

Uses the Kolmogorov-Smirnov test to compare the incoming data distribution with the baseline distribution.
Drift is detected if the p-value from the test is less than 0.05.

## Target Drift Detection:

Uses the Chi-square test to compare the incoming target distribution with the baseline target distribution.
Drift is detected if the p-value from the test is less than 0.05.

## Assumptions
Consistent Encoding and Scaling:

The LabelEncoder and StandardScaler used during the preprocessing phase are the same as those used during the model training phase to maintain consistency.

Data Quality:

Input data is assumed to be reasonably clean, aside from expected missing values in specific columns (job, location).

Baseline Statistics and Distributions:

The baseline statistics and target distribution used for drift detection are representative of the training data distribution and are assumed to be stored in baseline_stats.pkl and baseline_target_distribution.pkl.

## Reasoning
Form Validation:

Ensuring form inputs are validated prevents type conversion errors and maintains data integrity.
Pre-fitted Encoders and Scalers:

Using pre-fitted LabelEncoder and StandardScaler ensures the preprocessing steps are consistent with those applied during model training, thus preventing discrepancies that could degrade model performance.

## Data Drift Detection:

Monitoring for data drift ensures the model remains accurate and reliable over time by identifying when the input data distribution has changed significantly from the training data.


## Installation
Prerequisites
Python 3.7+
FastAPI
Uvicorn
Joblib
Scipy
Pandas
Scikit-learn

# Running the dependencies 

Download the zipfolder and unzip it in Documents 
Open command prompt or anaconda prompt 
change the working directory to the folder e.g C:\Users\Tinotenda Mangarai\Documents\Tinotenda Mangarai
once in the correct directory, copy and paste the code below to install the packages needed to run the app 

pip install -r requirements.txt


# Running the FastAPI Application
Once dependencies are installed, run the FastAPI application as follows: 


in the same directory used to install the dependencies (C:\Users\Tinotenda Mangarai\Documents\Tinotenda Mangarai), run the following:

copy and paste the code below in command prompt/anaconda prompt 

uvicorn tinotenda_mangarai_app:app --reload

After running the above command, the FastAPI application will be accessible locally at http://127.0.0.1:8000. Open a web browser and navigate to this address to interact with the application.

To test model performance, run the following code in command prompt/anaconda prompt: 

uvicorn tinotenda_model_performance_fastapi:app --reload

Copy and paste http://127.0.0.1:8000  to access the prediction form 


## Usage
Fill out the form with the loan applicant's details:

Gender (male, female, other)
Employment Status (TRUE or FALSE)
Job (e.g., Teacher)
Location (e.g., Harare)
Loan Amount (e.g., 5000)
Number of Defaults (e.g., 2)
Outstanding Balance (e.g., 2000)
Interest Rate (e.g., 0.15)
Age (e.g., 30)
Salary (e.g., 50000)
Marital Status (married, single, divorced)
Submit the form to get the prediction result along with the probability of default.

The page will also display results of data drift detection for features and target distribution.

## File Structure
tinotenda_mangarai_app.py: The main FastAPI application file.
tinotenda_model_performance_fastapi: the endpoint to test model performance 
tinotenda_mangarai_loan_default_model.pkl: The trained model.

scaler.pkl: Pre-fitted StandardScaler.
label_encoders.pkl: Pre-fitted LabelEncoder for categorical features.
baseline_stats.pkl: Baseline statistics for feature drift detection.
baseline_target_distribution.pkl: Baseline target distribution for target drift detection.
requirements.txt: List of dependencies.

## Contact
Tinotenda Mangarai 
0776970289
mangaraitinotenda@gmail.com
