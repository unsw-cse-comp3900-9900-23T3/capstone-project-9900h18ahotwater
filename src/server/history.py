from src.server.sql import dbsession, Data, History, User
from flask import Blueprint, request, session, jsonify

history = Blueprint('history', __name__)

@history.route('/gethistory', methods=['GET'])
def gethistory():
    if not session.get('isLogin'):
        return jsonify({'status': 'not login'})
    else:
        user_eamil = session.get('email')
        user_id = User().find_by_email(user_eamil).user_id
        res = History().find_by_user_id(user_id)
        data = []
        for i in res:
            data.append((i.data_id, i.history_model, i.classes, i.probability))
        response = {"num_of_history":len(data),
                    "num_of_image":[],
                    "image1":[],
                    "image2":[],
                    "text":[],
                    "model":[],
                    "classes":[],
                    "probability":[]}

        for i in range(len(data)):
            response["num_of_image"].append(Data().find_by_data_id(data[i][0]).num_img)
            response["image1"].append(Data().find_by_data_id(data[i][0]).img1)
            response["image2"].append(Data().find_by_data_id(data[i][0]).img2)
            response["text"].append(Data().find_by_data_id(data[i][0]).text)
            response["model"].append(data[i][1])
            response["classes"].append(data[i][2])
            response["probability"].append(data[i][3])
        return jsonify(response)


