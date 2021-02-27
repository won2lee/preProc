#app.py
from flask import Flask, render_template, request

app = Flask(__name__)

from app_utils import to_start, rid_blank, preproc_num, to_normal
from trns.get_model import trns_model

pre_en, pre_ko, trns = trns_model()

def nmt(X, to_start, pre_ko, pre_en, trns):

    enko_count = sum([1 if ord(c) in range(65,123) else -1 for c in X])
    X = to_start(X)
    
    if enko_count > 0:
        X = pre_en(X)
        X = preproc_num(X)
        X = [s.split(' ') for s in X if s.strip() !="''"] #'"'
        X = trns.translate(X,'ko')
        X = to_normal(X)
        
    else:
        X = pre_ko.forward(X)
        X = preproc_num(X)
        X = [s.split(' ') for s in X if s.strip() !="''"]
        X = trns.translate(X,'en')
        X = rid_blank(X)    

    return X

@app.route('/')
def root():
    return render_template('nmt.html') #'Welcome to Translation!'

#@main.route('/test')
#@app.route('/nmt')
#def test():
#    return render_template('nmt.html')

@app.route('/nmt', methods=['POST'])
def post():
    X = request.form['nmt']
    Y = nmt(X, to_start, pre_ko, pre_en, trns)
    return render_template('nmt.html',to_test = X, tested = Y)



if __name__ == '__main__':
    app.run(debug=True)
