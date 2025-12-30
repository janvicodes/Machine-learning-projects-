from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('logistic_regression.pkl', 'rb'))
feature_extraction = pickle.load(open('feature_extraction.pkl', 'rb'))

def predict_mail(input_mail):
    input_user_mail = [input_mail]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model.predict(input_data_features)
    return prediction[0]   # return single value

@app.route('/', methods=['GET', 'POST'])
def analyze_mail():
    if request.method == 'POST':
        mail = request.form.get('mail')
        predicted_mail = predict_mail(mail)
        return render_template('index.html', classify=predicted_mail)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
