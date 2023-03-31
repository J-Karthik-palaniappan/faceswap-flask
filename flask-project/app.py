from flask import Flask,request,render_template,flash,redirect,url_for,session,Response
from flask_session import Session
import os
import logindatabase
import final_face_swap

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def main():
    session.clear()
    final_face_swap.close_camera()
    return render_template("index.html")

@app.route("/model")
def model():
    if not session.get("uname"):
        return redirect(url_for('main'))
    return render_template("model.html")

@app.route('/video_feed')
def video_feed():
    final_face_swap.open_camera()
    img = session.get('img')
    return Response(final_face_swap.gen_frames(img), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/",methods=['POST'])
def login():
    uname = request.form['uname']
    psw = request.form['psw']
    session["uname"] = uname
    if logindatabase.loginuser(uname, psw):
        return redirect(url_for('model'))
    return render_template("index.html",loginstatus=False)
    
@app.route("/",methods=['Post'])
def signup():
    email = request.form['email']
    uname = request.form['uname']
    psw = request.form['psw']
    session["uname"] = uname
    
    if logindatabase.newuser(email, uname, psw):
        return redirect(url_for('model'))
    return render_template("index.html",signupstatus=False)

@app.route('/imageshow',methods=['GET','POST'])
def image_upload():
    path = None
    if request.method == 'POST':
        if 'img' not in request.files:
            return redirect(url_for('main'))
        file = request.files['img']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        session['img'] = path
        return render_template('model.html', img=path)
    
if __name__=="__main__":
    app.run(host='localhost',port=5000,debug=True)