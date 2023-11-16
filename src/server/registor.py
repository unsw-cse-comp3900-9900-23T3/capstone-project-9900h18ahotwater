from flask import Blueprint, request, session
from flask import jsonify, redirect, url_for
from src.server.sql import dbsession, User
from sqlalchemy import or_

registor = Blueprint('registor', __name__)


#/email
@registor.route('/reg', methods=['POST'])
def regist():
    data = request.get_json()
    email = data['email']
    phone = data['phone']
    password = data['password']
    session['email'] = email
    session['phone'] = phone
    session["isLogin"] = True

    res = dbsession.query(User).filter(or_(User.email == email, User.phone==phone)).first()
    if res:
        # phone or email exist
        return jsonify({'status': 'exist'})
    else:
        # phone or email not exist
        try:
            user = User(nickname= email,email=email, phone=phone, password=password, username=email)
            dbsession.add(user)
            dbsession.commit()
            return jsonify({'status': 'success'})
            # return redirect(url_for('index'))
        except:
            return jsonify({'status': 'fail'})


@registor.route('/code', methods=['POST'])
def ecode():
    data = request.get_json()
    email = data['email']
    session['email'] = email
    # session['code'] = '123456'
    # session["ecode"] = 
    return jsonify({'status': 'success'})

def send_email():
    pass
    