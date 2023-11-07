import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_C=0.01.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    user = request.get_json()

    X = dv.transform([user])
    y_pred = model.predict_proba(X)[0, 1]
    revenue = y_pred >= 0.5

    result = {
        'revenue_probability': float(y_pred),
        'revenue': bool(revenue)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)