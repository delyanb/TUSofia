from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#========================loading the save files==================================================
model1 = pickle.load(open('gradient_boosting_classifier.pkl','rb'))
model2 = pickle.load(open('random_forest_classifier.pkl','rb'))
model3 = pickle.load(open('ada_boost_classifier.pkl','rb'))
model4 = pickle.load(open('meta_model.pkl','rb'))
feature_extraction = pickle.load(open('feature_extraction.pkl','rb'))


def predict_mail_gradient(input_text):
    input_user_mail  = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model1.predict(input_data_features)
    return prediction

def predict_mail_forest(input_text):
    input_user_mail  = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model2.predict(input_data_features)
    return prediction

def predict_mail_ada(input_text):
    input_user_mail  = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model3.predict(input_data_features)
    return prediction

def predict_mail_stacking(input_text):
    input_user_mail  = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model4.predict(input_data_features)
    return prediction



@app.route('/', methods=['GET', 'POST'])
def analyze_mail():
    if request.method == 'POST':
        mail = request.form.get('mail')
        predicted_mail_gradient = predict_mail_gradient(input_text=mail)
        predicted_mail_forest = predict_mail_forest(input_text=mail)
        predicted_mail_ada = predict_mail_ada(input_text=mail)
        predicted_mail_meta = predict_mail_stacking(input_text=mail)
        return render_template('index.html', classify=predicted_mail_gradient,classify1 = predicted_mail_forest, classify2=predicted_mail_ada,classify3=predicted_mail_meta)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
