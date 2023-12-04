from app import app
from flask_restful import Resource, Api
from ml_integration import model_loader

api= Api(app)

class titanic_classify(Resource):
    def get(self, features):
        features = features.split(',')

        print(features)
        return {
            'Result':model_loader.classify_survived(features)}
    

class home(Resource):
    def get(self):
        return "Hello world"
    
api.add_resource(home, '/')
api.add_resource(titanic_classify, '/<string:features>')