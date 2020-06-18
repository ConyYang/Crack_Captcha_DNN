from flask import Flask, request
from datasetPrepare.createCaptcha import create_captcha
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return 'welcome'


if __name__ == '__main__':
    app.run()