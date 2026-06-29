from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

car = pd.read_csv("Cleaned Car.csv")

companies = sorted(car["company"].unique())
car_models = sorted(car["name"].unique())
years = sorted(car["year"].unique(), reverse=True)
fuel_types = car["fuel_type"].unique()


@app.route("/")
def home():
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )


@app.route("/predict", methods=["POST"])
def predict():

    company = request.form["company"]
    name = request.form["car_model"]
    year = int(request.form["year"])
    fuel = request.form["fuel"]
    kms = int(request.form["kms"])

    input_df = pd.DataFrame(
        [[name, company, year, kms, fuel]],
        columns=[
            "name",
            "company",
            "year",
            "kms_driven",
            "fuel_type"
        ]
    )

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
        prediction=round(prediction,2)
    )


if __name__ == "__main__":
    app.run(debug=True)