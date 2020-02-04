from flask import Flask
import pandas as pd
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/AAPL',methods=['GET','POST']) 

def dtoj():
    df1=pd.read_csv('AAPL.csv')
    re=df1.to_json(orient='records')
    return re
#     return df1.to_html(header="true", table_id="table")

@app.route('/GOOG',methods=['GET','POST']) 

def dtj():
    df2=pd.read_csv('GOOG.csv')
    re=df2.to_json(orient='records')
    return re

if __name__=='__main__':
    app.run()
