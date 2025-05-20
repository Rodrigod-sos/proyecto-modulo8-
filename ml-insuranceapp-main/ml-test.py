import joblib
import numpy as np
import sklearn

model = joblib.load('./model/Insurance.pkl')
scx = joblib.load('./model/Scaler_X.pkl')
scy = joblib.load('./model/Scaler_Y.pkl')

edad = int(input('Ingrese la edad: '))
edadsc = scx.transform(np.array([[edad]]))

prediction = model.predict(edadsc)

predictionsc = scy.inverse_transform(prediction)
print(f'Los gastos para un paciente con {edad} años resulta: S/ {predictionsc[0][0]:.2f}')