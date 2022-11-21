import os
import modal
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ludvigdoeser/Serverless-ML-Titanic-Survival-Tasks/main/data/titanic.csv")

#primary_key=["Pclass","Sex","Age","Sibsp","Parch","Fare","Embarked","Title","Deck"], 
titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["pclass","age","sibSp","parch","sex_male"],
    description="Titanic Survival dataset")

titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

print('Done')