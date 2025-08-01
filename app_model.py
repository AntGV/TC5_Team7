from flask import Flask, jsonify, request
import numpy as np
import os
import pandas as pd
import pickle


# os.chdir(os.path.dirname(__file__))
app = Flask(__name__)


# Enruta la landing page (endpoint /)
@app.route("/", methods = ["GET"])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    # texto = "Bienvenido a la API predictora del tiempo desarrollada por el Team-7" + "\n" + \
    #     "Las variables necesarias para la predicción son:" + "\n" + \
    #     "temp= hum= winds= prec= cloudc= atmp= uvi= season= vis= loc=" + "\n" + \
    #     "temp (Temperatura - Temperature // float, medido en grados Celsius)" + "\n" + \
    #     "hum (Humedad - Humidity // int, en porcentaje de humedad)" + "\n" + \
    #     "winds (Velocidad del viento - Wind Speed // float, en kilómetros por hora)" + "\n" + \
    #     "prec (Precipitaciones - Precipitation // float, en porcentaje de precipitaciones)" + "\n" + \
    #     "cloudc (Cobertura de nubes - Cloud Cover // str [clear, partly cloudy, cloudy, overcast])" + "\n" + \
    #     "atmp (Presión atmosférica - Atmospheric Pressure // float, en milibares)" + "\n" + \
    #     "uvi (Índice UV - UV Index // int, con valores entre 0 y 14)" + "\n" + \
    #     "season (Estación - Season // str, [Spring, Summer, Autumn, Winter])" + "\n" + \
    #     "vis (Visibilidad - Visibility // float, rango de visibilidad en kilómetros)" + "\n" + \
    #     "loc (Ubicación - Location // str [coastal, inland, mountain])" + "\n"

    return 
        <h1>Bienvenido a la API predictora del tiempo desarrollada por el Team-7</h1><br><br>
        <h2>Las variables necesarias para la predicción son:</h2><br>
        <h3>temp= hum= winds= prec= cloudc= atmp= uvi= season= vis= loc=</h3><br>
        <h4>temp (Temperatura - Temperature // float, medido en grados Celsius)</h4><br>
        <h5>hum (Humedad - Humidity // int, en porcentaje de humedad)</h5><br>
        <h6>winds (Velocidad del viento - Wind Speed // float, en kilómetros por hora)</h6><br>
        <h7>prec (Precipitaciones - Precipitation // float, en porcentaje de precipitaciones)</h7><br>
        <h8>cloudc (Cobertura de nubes - Cloud Cover // str [clear, partly cloudy, cloudy, overcast])</h8><br>
        <h9>atmp (Presión atmosférica - Atmospheric Pressure // float, en milibares)</h9><br>
        <h10>uvi (Índice UV - UV Index // int, con valores entre 0 y 14)</h10><br>
        <h11>season (Estación - Season // str, [Spring, Summer, Autumn, Winter])</h11><br>
        <h12>vis (Visibilidad - Visibility // float, rango de visibilidad en kilómetros)</h12><br>
        <h13>loc (Ubicación - Location // str [coastal, inland, mountain])</h13><br>

# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods = ["GET"])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open("modelo_clasificador.pkl", "rb") as f:
        model = pickle.load(f)

    Temperature = request.args.get("Temperature", None)
    Humidity = request.args.get("Humidity", None)
    Wind_Speed = request.args.get("Wind_Speed", None)
    Precipitation = request.args.get("Precipitation", None)
    Cloud_Cover = request.args.get("Cloud_Cover", None)
    Atm_Press = request.args.get("Atm_Press", None)
    UV_I = request.args.get("UV_I", None)
    Season = request.args.get("Season", None)
    Visibility = request.args.get("Visibility", None)
    Location = request.args.get("Location", None)

    prediction = model.predict([[float(Temperature),float(Humidity),float(Wind_Speed),float(Precipitation),float(Cloud_Cover),float(Atm_Press),float(UV_I),float(Season),float(Visibility),float(Location)]])
    return jsonify({'predictions': prediction[0]})
    # if jsonify({'predictions': prediction[0]}) == 0:
    #     etiqueta = "Sunny, thank you for the sunshine bouquet"
    # elif jsonify({'predictions': prediction[0]}) == 1:
    #     etiqueta = "Cloudy, the sky is gray and white and cloudy"
    # elif jsonify({'predictions': prediction[0]}) == 2:
    #     etiqueta = "On and on the rain will fall"
    # elif jsonify({'predictions': prediction[0]}) == 2:
    #     etiqueta = "Let it snow!, Let it snow!, Let it snow!"
    
    # return etiqueta


if __name__ == '__main__':
    app.run(debug=True)
