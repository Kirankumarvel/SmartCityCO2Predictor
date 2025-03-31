import os
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump
from colorama import Fore, Style

# Define Google API key 
GOOGLE_API_KEY = "Enter Your GOOGLE API KEY"
if not GOOGLE_API_KEY :
    print(Fore.RED + "Missing or invalid Google API key. Please replace 'YOUR_GOOGLE_API_KEY_HERE' with a valid API key." + Style.RESET_ALL)
    exit()

# Get city name from user
city = input("Enter city name: ").strip()
if not city:
    print(Fore.RED + "City name cannot be empty." + Style.RESET_ALL)
    exit()

def get_traffic_data(city):
    """Fetch real-time traffic data for a city using Google Places API."""
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=traffic+in+{city}&key={GOOGLE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            traffic_score = data["results"][0].get("rating", 3)
            review_count = data["results"][0].get("user_ratings_total", 100)
            return traffic_score, review_count
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error fetching traffic data: {e}" + Style.RESET_ALL)
    return None, None

# Get traffic data
traffic_score, review_count = get_traffic_data(city)
if traffic_score is None:
    print(Fore.RED + f"Could not retrieve traffic data for {city}. Using default values." + Style.RESET_ALL)
    traffic_score, review_count = 3, 100

# File handling with pathlib
data_path = Path("C:/Users/kiran/OneDrive/Desktop/python/machine-learning-python/city_co2_emissions.csv")
if not data_path.exists():
    print(Fore.RED + "Dataset file not found. Please ensure the CSV file exists." + Style.RESET_ALL)
    exit()

df = pd.read_csv(data_path)

# Add city if not in dataset
if city not in df['City'].values:
    new_data = pd.DataFrame([{ 'City': city, 'TrafficScore': traffic_score, 'ReviewCount': review_count, 'CO2Emissions': np.nan }])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(data_path, index=False)
    print(Fore.GREEN + f"New city data for {city} added to the dataset." + Style.RESET_ALL)
else:
    print(Fore.YELLOW + f"City {city} already exists in the dataset." + Style.RESET_ALL)

# Prepare data
cdf = df[['TrafficScore', 'ReviewCount', 'CO2Emissions']].dropna()
if len(cdf) < 2:
    print(Fore.RED + "Not enough data to train the model." + Style.RESET_ALL)
    exit()

X = cdf[['TrafficScore', 'ReviewCount']]
y = cdf['CO2Emissions']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

dump(model, "co2_predictor_model.joblib")  # Save model

# Prediction
city_data = np.array([[traffic_score, review_count]])
predicted_co2 = model.predict(city_data)
print(Fore.GREEN + "\nPrediction Results:" + Style.RESET_ALL)
print(Fore.MAGENTA + f"  - City: {city}" + Style.RESET_ALL)
print(Fore.MAGENTA + f"  - Predicted CO2 Emission: {predicted_co2[0]:.2f} g/km" + Style.RESET_ALL)

# Model performance
if len(X_test) > 1:
    test_score = model.score(X_test, y_test)
    print(Fore.BLUE + "\nModel Performance:" + Style.RESET_ALL)
    print(Fore.BLUE + f"  - R^2 Score on Test Data: {test_score:.2f}" + Style.RESET_ALL)
