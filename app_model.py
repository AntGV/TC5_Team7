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

    return '''
        <h1>Bienvenido a la API predictora del tiempo desarrollada por el Team-7</h1>
        <h2>Las variables necesarias para la predicción son:</h2>
        <h3>temp= hum= winds= prec= cloudc= atmp= uvi= season= vis= loc=</h3>
        <h4>temp (Temperatura - Temperature // float, medido en grados Celsius)</h4>
        <h4>hum (Humedad - Humidity // int, en porcentaje de humedad)</h4>
        <h4>winds (Velocidad del viento - Wind Speed // float, en kilómetros por hora)</h4>
        <h4>prec (Precipitaciones - Precipitation // float, en porcentaje de precipitaciones)</h4>
        <h4>cloudc (Cobertura de nubes - Cloud Cover // str [clear, partly cloudy, cloudy, overcast])</h4>
        <h4>atmp (Presión atmosférica - Atmospheric Pressure // float, en milibares)</h4>
        <h4>uvi (Índice UV - UV Index // int, con valores entre 0 y 14)</h4>
        <h4>season (Estación - Season // str, [Spring, Summer, Autumn, Winter])</h4>
        <h4>vis (Visibilidad - Visibility // float, rango de visibilidad en kilómetros)</h4>
        <h4>loc (Ubicación - Location // str [coastal, inland, mountain])</h4>
        '''

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

    predict_df = pd.DataFrame({
        'Temperature':float(Temperature),
        'Humidity':int(Humidity),
        'Wind_Speed':float(Wind_Speed),
        'Precipitation':float(Precipitation),
        'Cloud_Cover':str(Cloud_Cover),
        'Atm_Press':float(Atm_Press),
        'UV_I':int(UV_I),
        'Season':str(Season),
        'Visibility':float(Visibility),
        'Location':str(Location),
    }, index=[0])

    print(predict_df)

    prediction = model.predict(predict_df)

    #return jsonify({'predictions': prediction[0]})

    if prediction[0] == 0:
        return "Sunny, thank you for the sunshine bouquet"
    elif prediction[0] == 1:
        return "Cloudy, the sky is gray and white and cloudy"
    elif prediction[0] == 2:
        return "On and on the rain will fall"
    elif prediction[0] == 3:
        return "Let it snow!, Let it snow!, Let it snow!"


if __name__ == '__main__':
    app.run(debug=True)
