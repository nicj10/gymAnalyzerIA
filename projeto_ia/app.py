from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Carregar modelo
model = joblib.load("logistic_model_cbun.pkl")

@app.route("/")
def form():
    prediction = request.args.get("prediction")
    error = request.args.get("error")
    return render_template("form.html", prediction=prediction, error=error)

@app.route('/sobre.html')
def sobre():
    return render_template("sobre.html") 

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Coletar dados do formulário
        months_as_member = int(request.form["months_as_member"])
        weight = float(request.form["weight"])
        days_before = float(request.form["days_before"])

        # Dias da semana
        day_of_week = request.form["day_of_week"]
        days = ["Fri", "Mon", "Sat", "Sun", "Thu", "Tue", "Wed"]
        day_values = [1 if day_of_week == day else 0 for day in days]

        # Horário
        time = request.form["time"]
        time_AM = 1 if time == "AM" else 0
        time_PM = 1 if time == "PM" else 0

        # Categoria
        category = request.form["category"]
        categories = ["Aqua", "Cycling", "HIIT", "Strength", "Yoga"]
        category_values = [1 if category == cat else 0 for cat in categories]

        # Montar vetor de entrada
        input_data = [months_as_member, weight, days_before] + day_values + [time_AM, time_PM] + category_values
        input_array = np.array(input_data).reshape(1, -1)

        # Fazer a previsão
        prob = model.predict_proba(input_array)[0][1]
        prediction = f"{prob * 100:.2f}"  # Resultado em percentual

        # Redirecionar com o resultado
        return redirect(url_for("form", prediction=prediction))

    except Exception as e:
        return redirect(url_for("form", error=str(e)))


if __name__ == "__main__":
    app.run(debug=True)
