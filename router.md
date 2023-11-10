/reg           | post    | {"email":"", "password"="", "phone" = ""}         |  return: {"status":""}                 |  status: "success", "fail", "exist"
/detect        | post    | {"num_data":int, "img_path":[], "text":""}        |  return: {"classes":[], "prob":[]}     |
/dologin       | post    | {"email":"", "password"=""}                       |  return: {"status":""}                 |  status: "success", "fail", "code-error"
/uploadphoto   | post
/getcode       | get