#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def resultado_precio(dataTesting, fila_observacion):
    res = joblib.load(os.path.join(os.path.dirname(__file__), 'prediccion_carros.pkl'))
    conversion = joblib.load(os.path.join(os.path.dirname(__file__), 'conversion_datos_dummies.pkl'))
    
    data = dataTesting.iloc[[fila_observacion]]
    
    data_dummie = conversion(data)

    prediccion_resul = res.predict(data_dummie)

    return prediccion_resul

if __name__ == "__main__":

    dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0) 

    if len(sys.argv) == 1:
        print('Código del carro al que desea predecir el precio')
    else:
        fila_observacion = int(sys.argv[1])
        resultado_1 = resultado_precio(dataTesting, fila_observacion)
        
        print(f'La predicción del precio del carro es: {resultado_1[0]}')


        
        
