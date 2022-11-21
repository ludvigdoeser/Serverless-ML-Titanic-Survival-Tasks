import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

def titanic_predict(pclass,age,sibSp,parch,sex_male):
    input_list = []
    input_list.append(pclass)
    input_list.append(age)
    input_list.append(sibSp)
    input_list.append(parch)
    input_list.append(sex_male)
    
    # To get rid off the warning: make a pandas df where feature values AND feature names are present
    # .... 
    
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want the first element.
    if res[0] == 0:
        png = 'drowning'
    else:
        png = 'survived'
        
    titanic_im = "https://raw.githubusercontent.com/ludvigdoeser/Serverless-ML-Titanic-Survival-Tasks/main/figures/" + png + ".png"
    
    img = Image.open(requests.get(titanic_im, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic_predict,
    title="Titatnic Survival Predictive Analytics",
    description="Experiment with passenger feature to predict whether he/she survived.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Pclass (Ticket Class) [1 = 1st, 2 = 2nd, 3 = 3rd]"),
        gr.inputs.Number(default=30.0, label="Age"),
        gr.inputs.Number(default=0., label="SibSp (Number of siblings/spouses aboard)"),
        gr.inputs.Number(default=0., label="Parch (Number of parents/children aboard)"),
        gr.inputs.Number(default=0., label="Sex [0 = female, 1 = male]"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

