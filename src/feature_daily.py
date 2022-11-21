import os
import modal

LOCAL=True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 
    #image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib", "scikit-learn"])
    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalableML"))
    def f():
        g()


def empirical_family_members(feature):
    """
    These distributions are valid for both passengers that survived and for the group of passengers that died.
    """ 
    
    random_nr = random.random(0,1)
    
    if feature == 'sibSp':
    
        # Empirically we know:
        if random_nr >= 0 and random_nr <= 0.659664:
            return 0
        if random_nr >= 0.659664 and random_nr <= 0.915966:
            return 1
        if random_nr >= 0.915966 and random_nr <= 0.950980:
            return 2
        if random_nr >= 0.950980 and random_nr <= 0.967787:
            return 3
        if random_nr >= 0.967787 and random_nr <= 0.992997:
            return 4
        if random_nr >= 0.992997 and random_nr <= 1.0:
            return 5
    
    elif feature == 'parch':
        
        # Empirically we know:
        if random_nr >= 0 and random_nr <= 0.729692:
            return 0
        if random_nr >= 0.729692 and random_nr <= 0.883754:
            return 1
        if random_nr >= 0.883754 and random_nr <= 0.978992:
            return 2
        if random_nr >= 0.978992 and random_nr <= 0.985994:
            return 3
        if random_nr >= 0.985994 and random_nr <= 0.991597:
            return 4
        if random_nr >= 0.991597 and random_nr <= 0.998599:
            return 5
        if random_nr >= 0.998599 and random_nr <= 1.0:
            return 6
        
def generate_passenger(low_class, higher_class, age_min=0, age_max=160):
    """
    Returns a synthetic passenger
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"pclass": [random.randint(low_class,higher_class)],
                       "age": [random.randint(age_min,age_max)/2],
                       "sibSp": [empirical_family_members('sibSp')],
                       "parch": [empirical_family_members('parch')],
                       "sex_male": [random.randint(0,1)]
                      })
    
    if name == 'survived':
        df['survived'] = 1
    else:
        df['survived'] = 0
        
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    survived_df = pd.DataFrame({"pclass": [random.randint(1,3)],
                       "age": [random.randint(age_min,age_max)/2],
                       "sibSp": [empirical_family_members('sibSp')],
                       "parch": [empirical_family_members('parch')],
                       "sex_male": [random.randint(0,1)]
                      })
    
    drowned_df = pd.DataFrame({"pclass": [random.randint(low_class,higher_class)],
                       "age": [random.randint(age_min,age_max)/2],
                       "sibSp": [empirical_family_members('sibSp')],
                       "parch": [empirical_family_members('parch')],
                       "sex_male": [random.randint(0,1)]
                      })

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,2)
    elif pick_random >= 1:
        titanic_df = survived_df
        print("Survived passenger added")
    else:
        titanic_df = drowned_df
        print("Drowned passenger added")

    return titanic_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
