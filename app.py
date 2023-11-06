from flask import Flask, render_template, request, session
import os

app = Flask(__name__, template_folder='./src/templates', static_folder='./src/resources', static_url_path='')
app.config['SECRET_KEY'] = os.urandom(24)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/performance')
def performance():
    return render_template('performance.html')


@app.route('/introduction')
def introduction():
    return render_template('introduction.html')


@app.route('/model_1')
def model_1():
    return render_template('model_1.html')


@app.route('/registration')
def registration():
    return render_template('registration.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


if __name__ == '__main__':
    from src.server.registor import registor

    app.register_blueprint(registor)

    from src.server.detect import detect

    app.register_blueprint(detect)

    from src.server.login import login

    app.register_blueprint(login)

    app.run(debug=True)
    # app.run(debut=False, host='', port=80)
