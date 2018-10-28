import flask


app = flask.Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        # return the data dictionary as a JSON response
        return "Hello"


app.run(host='0.0.0.0', debug=True, port=80)
