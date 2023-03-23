import pickle 
import json 
import pandas as pd
import numpy as np 
import CONFIG
class prediction():
    def __init__(self):
         print("admission model")
        
    def load_raw(self):
        with open(CONFIG.model_path,"rb") as file:
               self.model=pickle.load(file)
        with open(CONFIG.asset_path,'r') as file:
               self.columns_name=json.load(file)
        
    def predict_admission(self,data):
        self.load_raw()
        self.data=data
        user_input=np.zeros(len(self.columns_name["columns"]))
        user_input[0]=data["gre_score"]
        user_input[1]=data["t_score"]
        user_input[2]=data["uni_rating"]
        user_input[3]=data["sop"]
        user_input[4]=data["lor"]
        user_input[5]=data["cgpa"]
        user_input[6]=data["research"]
        
        adm_chance=self.model.predict([user_input])

        return adm_chance

        
