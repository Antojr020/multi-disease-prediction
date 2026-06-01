from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Loading Models
try:
    diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
    print("✓ Diabetes model loaded")
except Exception as e:
    print(f"✗ Failed to load diabetes model: {e}")
    diabetes_model = None

try:
    heart_model = pickle.load(open('models/heart_disease.sav', 'rb'))
    print("✓ Heart disease model loaded")
except FileNotFoundError:
    print("✗ Heart disease model file not found")
    heart_model = None
except Exception as e:
    print(f"✗ Failed to load heart disease model: {e}")
    heart_model = None
    
try:
    parkinson_model = pickle.load(open('models/parkinson_trained_model.sav','rb'))
    print("✓ Parkinson model loaded")
except Exception as e:
    print(f"✗ Failed to load parkinson model: {e}")
    parkinson_model = None

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('diabetes.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')


@app.route('/parkinson')
def parkinson():
    return render_template('parkinson.html')


########################### Result Functions ########################################


@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetespedigree = float(request.form['diabetespedigree'])
        age = float(request.form['age'])
        skinthickness = float(request.form['skin'])
        pred = diabetes_model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)



@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

@app.route('/resultyy', methods=['GET', 'POST'])
def resultyy():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        mdvp_fo_hz= float(request.form['mdvp_fo_hz'])
        mdvp_fhi_hz = float(request.form['mdvp_fhi_hz'])
        mdvp_flo_hz= float(request.form['mdvp_flo_hz'])
        mdvp_jitter= float(request.form['mdvp_jitter'])
        mdvp_jitter_abs = float(request.form['mdvp_jitter_abs'])
        mdvp_rap = float(request.form['mdvp_rap'])
        mdvp_ppq  = float(request.form['mdvp_ppq'])
        jitter_ddp   = float(request.form['jitter_ddp'])
        mdvp_shimmer= float(request.form['mdvp_shimmer'])
        mdvp_shimmer_db = float(request.form['mdvp_shimmer_db'])
        shimmer_apq3= float(request.form['shimmer_apq3'])
        shimmer_apq5 = float(request.form['shimmer_apq5'])
        mdvp_apq= float(request.form['mdvp_apq'])
        shimmer_dda= float(request.form['shimmer_dda'])
        nhr= float(request.form['nhr'])
        hnr= float(request.form['hnr'])
        rpde= float(request.form['rpde'])
        dfa= float(request.form['dfa'])
        spread1= float(request.form['spread1'])
        spread2= float(request.form['spread2'])
        d2 = float(request.form['d2'])
        ppe= float(request.form['ppe'])
        pred = parkinson_model.predict(
            np.array([mdvp_fo_hz, mdvp_fhi_hz, mdvp_flo_hz, mdvp_jitter, mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Parkinson test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultyy.html', n=name, age=age, r=pred , gender=gender)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
