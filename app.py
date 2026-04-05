from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load KMeans model
with open("customer.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Columns
columns = [
    "Year_Birth","Education","Marital_Status","Income","Kidhome","Teenhome",
    "Recency","MntWines","MntFruits","MntMeatProducts","MntFishProducts",
    "MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases",
    "NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth",
    "AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5",
    "Complain","Response","day","month","year"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = [float(data[col]) for col in columns]
        features_array = np.array([features])
        cluster = int(kmeans.predict(features_array)[0])

        # Return cluster prediction as JSON
        return jsonify({"cluster": cluster})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)