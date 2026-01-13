from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

app = Flask(__name__)

# Load model & scaler
model = load_model("model_ann.h5")
scaler = load("scaler.pkl")

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

image_map = {
    'Iris-setosa': 'images/setosa.jpg',
    'Iris-versicolor': 'images/versicolor.jpg',
    'Iris-virginica': 'images/virginica.jpg'
}

# =========================
# HALAMAN HOME
# =========================
@app.route('/')
def home():
    return render_template('home.html')

# =========================
# HALAMAN PREDIKSI
# =========================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    probs = None
    image = None
    error = None

    if request.method == 'POST':
        try:
            sl = float(request.form['sepal_length'])
            sw = float(request.form['sepal_width'])
            pl = float(request.form['petal_length'])
            pw = float(request.form['petal_width'])

            if not (
                4.0 <= sl <= 8.0 and
                2.0 <= sw <= 4.5 and
                1.0 <= pl <= 7.0 and
                0.1 <= pw <= 2.5
            ):
                error = "Input berada di luar rentang dataset Iris"
            else:
                data = np.array([[sl, sw, pl, pw]])
                data_scaled = scaler.transform(data)

                output = model.predict(data_scaled)[0]

                prediction = classes[np.argmax(output)]
                probs = [round(p * 100, 2) for p in output]
                image = image_map[prediction]

        except:
            error = "Input tidak valid"

    return render_template(
        'predict.html',
        prediction=prediction,
        probs=probs,
        image=image,
        error=error
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

