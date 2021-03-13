from flask import Flask, jsonify, request
from model import get_prediction, batch_prediction
from service_streamer import ThreadedStreamer


app = Flask(__name__)
streamer = ThreadedStreamer(batch_prediction, batch_size=64)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        low, mlow, middle, mhigh, high, other = get_prediction(image_bytes=img_bytes)
        return jsonify({'低收入': low, '中低收入': mlow, '中等收入': middle, '中高收入': mhigh, '高收入': high, '其他': other,})


@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        low, mlow, middle, mhigh, high, other = streamer.predict([img_bytes])[0]
        return jsonify({'低收入': low, '中低收入': mlow, '中等收入': middle, '中高收入': mhigh, '高收入': high, '其他': other,})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
