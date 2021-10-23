from flask import Flask, request as freq
from log.log import logger
from bert_corrector import BertCorrector

app = Flask(__name__)

d = BertCorrector()

@app.route('/correcting', methods=['GET', 'POST'])
def corrector():
    string = dict(freq.args)['data']
    corrected_string = d.bert_correct(string)
    print(string, corrected_string)
    return corrected_string

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)
