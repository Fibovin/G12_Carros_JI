from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='3.0',
    title='Proyecto 1: predicción precios de vehiculos (G12)',
    description='Con esta API podra identificar la predicción que puede tener el carro que sea de su interes revisar.'
)

ns = api.namespace('predicción_ID', description='En esta se contempla la predicción del vehiculo')

parametros = api.parser()
parametros.add_argument(
    'Observacion',
    type=int,
    required=True,
    help='Digite el Cód. ID asigando al vehiculo que desea identificar la predicción del precio',
    location='args'
)

archivos = api.model('Resource', {
    'Resultado': fields.Float,
})

def resultado_precio(dataTesting, fila_observacion):
    res = joblib.load(os.path.join(os.path.dirname(__file__), 'prediccion_carros.pkl'))
    
    data = dataTesting.iloc[[fila_observacion]]
    
    data_dummie = conversion_datos_dummies(data)

    prediccion_resul = res.predict(data_dummie)

    return prediccion_resul

@ns.route('/')
class CarPriceApi(Resource):
    @api.doc(parser=parametros)
    @api.marshal_with(archivos)
    def get(self):
        args = parametros.parse_args()
        fila_observacion = args['Observacion']
        
        dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)  

        try:
            res_prediccion = resultado_precio(dataTesting, fila_observacion)
            return {"Resultado": res_prediccion[0]}, 200
        except Exception as e:
            api.abort(404, f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
