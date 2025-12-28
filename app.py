from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client

# Load the earthquake data from CSV file
data = pd.read_csv("E:\WebsiteWithEarthquake\data\earthquakes.csv")
data.dropna(subset=["mag"], inplace=True)
# Split the data into input features and target labels
X = data[["latitude", "longitude"]]
y = data["mag"]

# Convert the latitude and longitude values to radians
X.loc[:,"latitude"] = np.radians(X["latitude"])
X.loc[:,"longitude"] = np.radians(X["longitude"])

# Scale the input features to zero mean and unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train a CatBoost model on the entire dataset
catboost = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=3, loss_function="RMSE", random_seed=42)
catboost.fit(X, y)


account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_FROM_NUMBER")
opencage_key = os.getenv("OPENCAGE_API_KEY")


client = Client(account_sid, auth_token)


# Define a Flask web application
app = Flask(__name__)

# Define a function that takes in a location and returns the predicted earthquake risk
def predict_risk(location):
    # Convert the location to latitude and longitude using a geocoding API
    # Here, we'll use the OpenCage Geocoder API (https://opencagedata.com/)
    # You'll need to sign up for an API key and replace "YOUR_API_KEY" below with your actual key
    import opencage.geocoder
    geocoder = opencage.geocoder.OpenCageGeocode("opencage_key")
    results = geocoder.geocode(location)
    if len(results) == 0:
        raise ValueError("Location not found")
    latitude = results[0]['geometry']['lat']
    longitude = results[0]['geometry']['lng']
    # Make a prediction using the trained CatBoost model
    return catboost.predict(scaler.transform([[np.radians(latitude), np.radians(longitude)]]))[0]

# Define a Flask route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Define a Flask route for the prediction result page
@app.route("/predict", methods=["POST"])
def predict():
    # Get the location input from the web form
    location = request.form["location"]
    try:
        # Make a prediction using the input location
        risk = predict_risk(location)
        to_number = request.form["phone"]

        # Send an SMS message with the predicted earthquake risk
        message = 'The predicted earthquake risk for {} is {}.'.format(location, risk)
        client.messages.create(from_=from_number, to=to_number, body=message)
        return render_template("result.html", location=location, risk=risk)
    except ValueError:
        # If the location is not found, display an error message
        return render_template("error.html", message="Location not found")

@app.route("/statistics", methods=["POST"])
def showStat():
    
    
    try:
       
        return render_template("stat.html")
    except ValueError:
        # If the location is not found, display an error message
        return render_template("error.html", message="No Data Found")

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)
