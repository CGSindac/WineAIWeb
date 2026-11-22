from flask import Flask, render_template, redirect, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    try:
        # Get inputs
        f_acid = request.form.get('f_acidity')
        v_acid = request.form.get('v_acidity')
        c_acid = request.form.get('c_acid')
        r_sugar = request.form.get('r_sugar')
        chlorides = request.form.get('chlorides')
        f_sulfur_d  = request.form.get('f_sulfur_d')
        t_sulfur_d = request.form.get('t_sulfur_d')
        density = request.form.get('density')
        pH = request.form.get('pH')
        sulphates = request.form.get('sulphates')
        alcohol = request.form.get('alcohol')

        new_val = list(map(float, [f_acid, v_acid, c_acid, r_sugar, chlorides, f_sulfur_d, t_sulfur_d, density, pH, sulphates, alcohol]))

        # Compute prediciton
        print(new_val)
        return render_template('result.html', result=100.00)
    except:
        return render_template('predict.html')