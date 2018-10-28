import flask
import numpy as np
from run import predict as run_predict
from run import get_staff_dict, train
from flask import request, jsonify
from keras import utils as np_utils

app = flask.Flask(__name__)


@app.route("/train", methods=["GET"])
def get():
    return jsonify(score=train()), 200


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":

        # return the data dictionary as a JSON response
        req = request.get_json(force=True)
        col1 = get_staff_dict()[req['staff_id']]
        col2 = np_utils.to_categorical(range(7), 7)[req['weekday']]
        col3 = int(req['yesterday'])
        col4 = int(req['before_yesterday'])
        col5 = int(req['three_days_ago'])
        features = np.concatenate((col1, col2, col3, col4, col5), axis=None)
        features = np.reshape(np.array(features), (1, -1))
        try:
            res = run_predict(features)
        except Exception as e:
            return jsonify(error=str(e)), 400

        return jsonify(chance=str(res[0][0])), 200


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=80, threaded=False)
