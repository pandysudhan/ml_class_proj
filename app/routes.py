from app import app
from flask_restful import Resource, Api
from app.ml_integration import model_loader
api= Api(app)

class iris_predict(Resource):
    def get(self, features):
        features = features.split(',')
        for i in range(len(features)):
            features[i] = float(features[i])

        print(features)
        return model_loader.find_type_iris(features)
    

class home(Resource):
    def get(self):
        return "HEllo world"
    
api.add_resource(home, '/')
api.add_resource(iris_predict, '/<string:features>')