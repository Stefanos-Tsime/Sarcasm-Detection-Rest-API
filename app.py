from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import re



classifier = pickle.load(open('classifier.pickle','rb'))
count_vectorizer = pickle.load(open('count_vectorizer.pickle','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    #inhouse data preparation
    text = request.form['News_Headline_text']
    text = re.sub('[\W]+', ' ',re.sub('\'', '',text.lower()))
    text = np.array([text])
    
    #apply imported vectorizer and classifier
    vectors = count_vectorizer.transform(text)
    prediction = classifier.predict(vectors)
    
    return render_template('index.html',prediction_text=f'the prediction is: {prediction}')


if __name__ == "__main__":
    app.run(port=8005, debug=True)