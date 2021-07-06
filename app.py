from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
from termcolor import colored

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    final_features=request.form['All']
    final=final_features.split(',')
    feature=[]
    for i in final:
        if '.' in i:
            feature.append(float(i))
        else:
            feature.append(int(i))
    feature_array=np.array(feature)
    feature1=np.reshape(feature_array,(1,-1))
    prediction = model.predict(feature1)
    output=prediction[0]

    if output==0:
        print("Not Diabetic")
    else:
        print("Diabetic")

    return  render_template('index.html',prediction_text='1:You might have diabeties. Please visit doctor\n''0:Chance of diabeties are low\n' ',' 'The model prediction are "{}"'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__=="__main__":
    app.run(debug=True)

