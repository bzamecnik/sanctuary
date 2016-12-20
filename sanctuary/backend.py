from flask import Flask
from flask import jsonify
import json
from bson.json_util import dumps as bson_dumps
# from flask_cors import CORS

import sacred_mongo

app = Flask(__name__)

# enable CORS since the back-end is on a different place than th front-end
# CORS(app)

app.debug = True

@app.route("/experiments", methods=['GET'])
def experiments():
    return jsonify({'experiments': sacred_mongo.list_experiments()})

@app.route("/runs", methods=['GET'])
def list_runs():
    return jsonify({'runs': [str(r['_id']) for r in sacred_mongo.list_runs()]})

@app.route("/runs/by-experiment/<experiment_id>", methods=['GET'])
def list_runs_by_experiment(experiment_id):
    return jsonify({'runs': [str(r['_id']) for r in sacred_mongo.list_runs_by_experiment(experiment_id)]})

@app.route("/runs/<run_id>", methods=['GET'])
def run_details(run_id):
    return jsonify(json.loads(bson_dumps(sacred_mongo.get_run(run_id))))

@app.route("/files", methods=['GET'])
def list_files():
    return jsonify(json.loads(bson_dumps({'files': [f for f in sacred_mongo.list_files()]})))

if __name__ == "__main__":
    # http://stackoverflow.com/questions/23639355/extremely-long-wait-time-when-loading-rest-resource-from-angularjs
    # app.run(host="0.0.0.0", threaded=True)
    app.run()
