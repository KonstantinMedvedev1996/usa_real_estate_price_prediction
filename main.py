import streamlit as st
import pandas as pd 
import seaborn as sns
import json
import joblib

st.header('USA Real Estate Price Prediction')

path_data = 'data/zip_mean_price.pickle'
# path_unique_values = 'data/unique_values.json'
# path_model = 'models/lr_pipeline.sav'

# df = pd.read_csv('data/russiarealestate.zip')

@st.cache_data
def load_data(path):
    "Load data from path"
    data = pd.read_pickle(path)
    #for demonstraion
    # data = data.sample(5000)
    return data

# @st.cache_data
# def load_model(path):
#     "Load model from path"
#     model = joblib.load(path)
#     return model

@st.cache_data
def transform(data):
    "Transform data"
    colors = sns.color_palette("coolwarm").as_hex()
    n_colors = len(colors)
    
    data = data.reset_index(drop=True)
    # data["norm_price"] = data["price"]/data["area"]
    
    data["label_colors"] = pd.qcut(data["mean_price_sqft"], n_colors, labels=colors)
    data["label_colors"] = data["label_colors"].astype("str")
    return data

df = load_data(path_data)
df = transform(df)
st.write(df[:10])

st.map(data=df, latitude="latitude",longitude="longitude", color="label_colors")

# with open(path_unique_values) as file:
#     dict_unique = json.load(file)


# #features
# building_type = st.sidebar.selectbox('Building type',(dict_unique['building_type']))
# object_type = st.sidebar.selectbox('Object type',(dict_unique['object_type']))
# level = st.sidebar.slider ("Level", min_value=min(dict_unique["level"]), max_value=max(dict_unique["level"]))
# levels = st.sidebar.slider ("Levels", min_value= min(dict_unique["levels"]), max_value = max(dict_unique["levels"]))
# rooms = st.sidebar.selectbox('Rooms',(dict_unique['rooms']))
# area = st.sidebar.slider ("Area", min_value=min(dict_unique["area"]), max_value = max(dict_unique["area"]))
# kitchen_area = st.sidebar.slider("Kitchen area", min_value=min(dict_unique["kitchen_area"]), max_value = max(dict_unique["kitchen_area"])) 

# dict_data = {
    
#     "building_type": building_type,
#     "object_type": object_type,
#     "level": level,
#     "levels": levels,
#     "rooms" : rooms,
#     "area": area,
#     "kitchen_area":kitchen_area
# }

# data_predict = pd.DataFrame([dict_data])
# model = load_model(path_model)

# button_1 = st.button("Hello")
# # button_2 = st.button("Predict_price")
# button_3 = st.button("Predict")

# if button_1:
#     st.write('Why hello there')
    
# # if button_2:
# #     model.predict(data_predict)

# if button_3:
#     output = model.predict(data_predict)
#     st.success(f"{round(output[0])} rub")
    
    
    
    
st.markdown(
    
   """
    ### Описание полей
    
        - date - date of publication of the announcement;
        - time - the time when the ad was published;
        - geo_lat - Latitude
        - geo_lon - Longitude
        - region - Region of Russia. There are 85 subjects in the country in total.
        - building_type - Facade type. 0 - Other. 1 - Panel. 2 - Monolithic. 3 - Brick. 4 - Blocky. 5 - Wooden
        - object_type - Apartment type. 1 - Secondary real estate market; 2 - New building;
        - level - Apartment floor
        - levels - Number of storeys
        - rooms - the number of living rooms. If the value is "-1", then it means "studio apartment"
        - area - the total area of ​​the apartment
        - kitchen_area - Kitchen area
        - price - Price. in rubles

   """ 


)