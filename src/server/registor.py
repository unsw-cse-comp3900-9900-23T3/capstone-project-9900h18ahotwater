from flask import Blueprint, request, session
from flask import jsonify

registor = Blueprint('registor', __name__)

@registor.route('/reg', methods=['POST'])
def regist():
    data = request.get_json()
    email = data['email']
    phone = data['phone']
    password = data['password']
    session['email'] = email
    session['phone'] = phone
    session["isLogin"] = True
    #write to database
    #to do
    return jsonify({'status': 'success'})

@registor.route('/ecode', methods=['POST'])
def ecode():
    data = request.get_json()
    email = data['email']
    session['email'] = email
    # session["ecode"] = 
    return jsonify({'status': 'success'})

def send_email():
    pass
    