from flask import Flask,render_template,request
from app.utils import prediction 
import numpy as np
import CONFIG
app=Flask(__name__)

@app.route('/')
def default():


    return render_template("admission.html")
@app.route('/admission/form',methods=['POST','GET'])
def get_data():
    data=request.form
    pred_obj=prediction()
    chance=pred_obj.predict_admission(data)
    chance_percent=np.around(chance*100,2)
    return str(chance_percent[0])

if __name__=="__main__":
    app.run(CONFIG.host,CONFIG.port,CONFIG.debug)