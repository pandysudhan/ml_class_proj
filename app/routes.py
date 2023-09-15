from app import app
from flask_restful import Resource, Api
from app.ml_integration import model_loader
api= Api(app)

class iris_predict(Resource):
    def get(self, features):
        features = features.split(',')
        for i in range(len(features)):
            print(features)
            features[i] = float(features[i])

        print(features)
        return {'sepal_length':features[0],
                'sepal_width':features[1],
                'sepal_length':features[2],
                'petal_width':features[3],
            'flower type':model_loader.find_type_iris(features)}
    

class home(Resource):
    def get(self):
        return "HEllo world"
    
api.add_resource(home, '/')
api.add_resource(iris_predict, '/<string:features>')