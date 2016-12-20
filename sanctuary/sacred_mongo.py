"""
A simple Python API for loading Keras models from MongoDB run artifact stored
by the Sacred library.

Usage:

run_id = '5843062c4e60f9a60c9db41f'
model = get_model(get_run(run_id))
"""

from bson.objectid import ObjectId
import pymongo
from tempfile import TemporaryDirectory

mongo_client = pymongo.MongoClient()
db = mongo_client['sacred']
runs_collection = db['default.runs']
chunks_collection = db['default.chunks']
files_collection = db['default.files']

def list_runs():
    return runs_collection.find({}, {'_id': True})

def list_runs_by_experiment(exp_name):
    return runs_collection.find(
        {'experiment.name': exp_name},
        {'_id': True})

def list_experiments():
    return runs_collection.distinct('experiment.name')

def list_files():
    return files_collection.find()

def get_run(run_id):
    if isinstance(run_id, str):
        run_id = ObjectId(run_id)
    return runs_collection.find_one(run_id)

def get_file_chunks(file_id):
    for chunk in chunks_collection.find({'files_id': file_id}) \
        .sort([('n', pymongo.ASCENDING)]):
        yield chunk['data']

def find_model_artifact(run, suffix='model.h5'):
    for artifact_id in run['artifacts']:
        artifact = files_collection.find_one(artifact_id)
        if artifact['filename'].endswith(suffix):
            return artifact

# We stored the model as a run artifact. The artifact file is stored in
# multiple chunks which we need to put together.
def model_from_chunks(model_chunks):
    import keras
    # Since h5py doesn't allow reading from in-memory file-like objects,
    # let's store it to a temporary file.
    # http://stackoverflow.com/questions/16654251/can-h5py-load-a-file-from-a-byte-array-in-memory
    # We use TemporaryDirectory instead of NamedTemporaryFile since after
    # closing such file it gets deleted before Keras can load it.
    with TemporaryDirectory() as temp_dir:
        model_file = temp_dir + '/model.h5'
        with open(model_file, 'wb') as f:
            for chunk_data in model_chunks:
                f.write(chunk_data)
        return keras.models.load_model(model_file)

def get_model(run):
    model_artifact = find_model_artifact(run)
    print('Loading model:', model_artifact['filename'])
    model_chunks = get_file_chunks(model_artifact['_id'])
    model = model_from_chunks(model_chunks)

    return model
