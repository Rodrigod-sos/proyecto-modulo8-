from flask import Flask,request,render_template
import joblib
import numpy as np
import sklearn

model = joblib.load('./model/Insurance.pkl')
sc_x = joblib.load('./model/Scaler_X.pkl')
sc_y = joblib.load('./model/Scaler_Y.pkl')

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    predictionresult = 0
    edadrequest = 0
    
    if request.method == 'POST':
        edadrequest = int(request.form['edad'])
        edadsc = sc_x.transform(np.array([[edadrequest]]))
        prediction = model.predict(edadsc)
        predictionsc = sc_y.inversetransform(prediction)
        predictionresult = round(predictionsc[0][0],2)
        
    return render_template('index.html',edad=edadrequest, prediction=predictionresult)

app.run(debug=True)