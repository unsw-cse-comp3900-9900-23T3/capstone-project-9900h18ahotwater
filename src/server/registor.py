from flask import Blueprint, request
from flask import jsonify

bp = Blueprint('registor', __name__)

@bp.route('/reg', methods=['POST'])
def sendEmail():
    email = request.get_json()['email']
    # print(email)
    # print("hahahahah")
    return jsonify({'status': 'success', 'email': email})
    