from flask import Flask, redirect, url_for, request, jsonify, make_response
import ml_models

app = Flask(__name__)

@app.route("/")
def hello():
    return "Post json to http://18.236.65.205/postjson to get results!"

@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    if request.is_json:
        json_data = request.get_json()
        res1,res2,res3,res4=ml_models.predictor(json_data)
        response_body={
        "1": res1,
        "2": res2,
        "3": res3,
        "4": res4
        }
        res=make_response(jsonify(response_body),200)
    else:
        res='No json found!'
        
    return res
  
if __name__ == "__main__":
    app.run(debug=True)