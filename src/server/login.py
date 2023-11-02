from flask import Blueprint, request, session
from flask import jsonify
from src.server.sql import dbsession, User

login = Blueprint('login', __name__)


@login.route('/dologin', methods=['POST'])
def dologin():
    data = request.get_json()
    email = data['email']
    password = data['password']
    session['email'] = email
    session["isLogin"] = True
    res = dbsession.query(User).filter(User.email == email).first()
    if res:
        if res.password == password:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'fail'})
    else:
        return jsonify({'status': 'fail'})


@login.route('/dologout', methods=['GET'])
def dologout():
    session.pop('email', None)
    session.pop('isLogin', None)
    return jsonify({'status': 'success'})
