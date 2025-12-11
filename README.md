### ML Capstone Project: Missing Persons Location Estimator
Project Overview
This project is a machine learning-based system designed to predict the likely location of missing persons based on various features such as age, vulnerability, location type, time of day, weather conditions, and other relevant factors. The system includes:

Synthetic dataset generation simulating realistic missing persons cases

Machine learning models to predict distance traveled by missing individuals

Interactive location estimation with search rings and probability heatmaps

Visualization tools including interactive maps with search areas

Search plan generation with actionable recommendations

Key Features
1. Realistic Dataset Generation
Synthetic data generation based on UK Missing Persons Unit statistics

5000+ samples with 19 features including:

Age groups and vulnerability scores

Location types (urban, suburban, rural)

Time and weather conditions

Search parameters and outcomes

Real-world patterns and correlations based on research

2. Machine Learning Pipeline
Multiple regression models implemented:

Random Forest Regressor

Gradient Boosting Regressor

XGBoost

LightGBM

Lasso/Ridge Regression

Automated hyperparameter tuning via GridSearchCV/RandomizedSearchCV

Comprehensive model evaluation metrics (MAE, MSE, R²)

3. Location Estimation System
Search Ring Generation: Concentric rings based on prediction confidence (50%, 80%, 95%)

Probability Grids: Gaussian probability distribution around last known location

Likely Location Identification: Common patterns based on vulnerability and age

Interactive Maps: Folium-based visualization with search areas and heatmaps

4. Search Planning Tools
Priority area calculations (high/medium/low priority)

Terrain-specific search recommendations

Estimated search time and team requirements

Comprehensive search plan reports

Immediate action recommendations based on case details

System Architecture
Core Components:
Dataset Generator (MissingPersonsDataset)

Creates synthetic missing persons data

Adds derived features for ML modeling

Saves datasets and statistics

Location Estimator (LocationEstimator)

Predicts likely locations based on ML model outputs

Generates search maps and probability grids

Creates actionable search plans

ML Model Pipeline

Feature preprocessing (scaling, encoding)

Multiple model training and evaluation

Cross-validation and hyperparameter optimization

Dataset Features
Basic Features:
case_id: Unique identifier

age_group: Categorical age groups (0-11, 12-17, 18-64, 65+)

vulnerability: Special conditions (dementia, autism, mental illness, etc.)

location_type: Urban, suburban, or rural

time_of_day: Morning, afternoon, evening, night

temperature_c: Weather conditions

has_vehicle: Binary indicator

search_party_size: Number of searchers

Derived Features:
age_numeric: Numerical age representation

vulnerability_score: Quantified vulnerability

urban_density: Location density score

time_score: Risk factor based on time

risk_factor: Combined risk assessment

is_weekend: Weekend indicator

Target Variables:
distance_km: Distance traveled (primary prediction target)

hours_until_found: Time until location

is_found: Recovery status

Getting Started
Prerequisites
bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm folium geopy streamlit joblib
Running the Project
Generate Dataset:

python
from dataset_generator import MissingPersonsDataset

dataset_gen = MissingPersonsDataset()
df = dataset_gen.generate_dataset(5000)
dataset_gen.save_dataset()
Train ML Models:

python
# Data preprocessing
    X = df.drop(['distance_km', 'hours_until_found'], axis=1)
    y_distance = df['distance_km']
    y_time = df['hours_until_found']

# Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y_distance, test_size=0.2)
Use Location Estimator:

python
    estimator = LocationEstimator()
    estimator.set_last_known_location(51.5074, -0.1278)  # London coordinates

# Generate search map
search_map = estimator.create_search_map(
    predicted_distance_km=5.0,
    output_file='search_map.html',
    vulnerability='dementia'
)
Model Performance
The system achieves:

R² Score: 0.5436 (Distance prediction)

RMSE: 2.609 km

MAE: Competitive error metrics across models

Best performing models:

Gradient Boosting Regressor

Random Forest Regressor

XGBoost

Visualization Examples
1. Interactive Search Map
Last known location marker

Confidence-based search rings (50%, 80%, 95%)

Probability heatmaps

Likely location markers with descriptions

2. Search Plan Report
json
{
  "case_summary": {...},
  "prediction": {
    "distance_km": 4.2,
    "confidence": "Based on ML model with R²=0.5436",
    "error_margin": "±2.61 km (RMSE)"
  },
  "search_areas": {...},
  "likely_locations": [...],
  "immediate_actions": [...]
}
Streamlit Web Application
The project includes a Streamlit web interface for:

Interactive case input through forms

Real-time predictions and visualizations

Search plan generation with downloadable reports

Case history tracking

To run the web app:

bash
streamlit run app.py
Search Plan Components
Priority Areas:
High Priority (50% confidence radius)

Core search area with highest probability

Rapid response deployment

Medium Priority (80% confidence radius)

Secondary search area

Grid search patterns

Low Priority (95% confidence radius)

Expanded search area

Aerial support consideration

Terrain-Specific Advice:
Urban: Focus on buildings, CCTV, public transport

Suburban: Check parks, trails, community centers

Rural: Search trails, water sources, abandoned buildings

Mixed: Coordinate across different terrain types

Customization Options
1. Dataset Parameters
python
# Adjust generation parameters
dataset_gen.generate_dataset(
    n_samples=10000,
    seed=123,
    age_distribution=[0.2, 0.3, 0.4, 0.1],  # Custom age groups
    vulnerability_weights={...}  # Custom vulnerability probabilities
)
2. Model Configuration
python
# Custom hyperparameter grids
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.2]
}
3. Search Parameters
python
# Adjust confidence levels
rings = estimator.generate_search_rings(
    predicted_distance_km=5.0,
    confidence_levels=[0.3, 0.6, 0.9, 0.99]
)
Research Basis
The project is based on:

UK Missing Persons Unit statistics

Academic research on missing persons behavior

Real-world search and rescue methodologies

Geospatial analysis techniques

Key statistical insights:

Children (0-11): Average distance 0.8km

Teens (12-17): Average distance 2.5km

Adults (18-64): Average distance 5.2km

Seniors (65+): Average distance 1.6km

Dementia patients: 95% found within 24h, average distance 1.6km

Use Cases
1. Law Enforcement
Rapid deployment planning

Resource allocation optimization

Search area prioritization

2. Search and Rescue Teams
Terrain-specific search strategies

Time and team requirement estimation

Evidence-based decision making

3. Community Organizations
Vulnerability-aware search planning

Public awareness campaigns

Preventive measures development

4. Academic Research
Missing persons behavior analysis

Predictive model development

Policy and protocol evaluation

Future Enhancements
Planned Features:
Real-time Data Integration

Live weather data

Traffic patterns

Recent incident reports

Advanced ML Features

Deep learning models

Ensemble methods

Time-series analysis

Mobile Application

Field data collection

Offline functionality

GPS integration

Additional Data Sources

Social media data

CCTV network analysis

Transportation records

Technical Improvements:
Model Explainability: SHAP values, feature importance

Uncertainty Quantification: Bayesian methods

Automated Reporting: PDF generation, email integration

API Development: RESTful endpoints for integration

Contributing
We welcome contributions in:

Model Development: New algorithms and improvements

Data Sources: Additional datasets and features

Visualization: Enhanced mapping and UI components

Documentation: Tutorials and use case examples

Development Setup:
bash
# Clone repository
git clone https://github.com/yourusername/missing-persons-ml.git
cd missing-persons-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/


Acknowledgments
UK Missing Persons Unit for statistical data

Open-source community for ML libraries and tools

Search and rescue organizations for domain expertise

Academic researchers in geospatial analysis and predictive modeling
