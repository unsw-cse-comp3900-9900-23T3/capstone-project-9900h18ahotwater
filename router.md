/reg           | post    | {"email":"", "password"="", "phone" = ""}         |  return: {"status":""}                 |  status: "success", "fail", "exist"
/detect        | post    | {"num_data":int, "img_path":[], "text":"", "model":"model1"} model 可选项（model1，model2...model4）        |  return: {"classes":[], "prob":[]}     |
/dologin       | post    | {"email":"", "password"=""}                       |  return: {"status":""}                 |  status: "success", "fail", "code-error"
/uploadphoto   | post
/getcode       | get
/dologout      | get