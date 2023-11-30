import streamlit as st
import pandas as pd 
import seaborn as sns
import json
import joblib
# import sklearn

st.header('USA Real Estate Price Prediction')

path_data = 'data/zip_mean_price.pickle'
path_unique_values = 'data/unique_values.json'
path_model = 'models/model.sav'

# df = pd.read_csv('data/russiarealestate.zip')

@st.cache_data
def load_data(path):
    "Load data from path"
    data = pd.read_pickle(path)
    #for demonstraion
    # data = data.sample(5000)
    return data

@st.cache_data
def load_model(path):
    "Load model from path"
    model = joblib.load(path)
    return model

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
# st.write(df[:10])

st.map(data=df, latitude="latitude",longitude="longitude", color="label_colors")

with open(path_unique_values) as file:
    dict_unique = json.load(file)


# #features
status_type = st.sidebar.selectbox('Statusctype',(dict_unique['status']))
object_type = st.sidebar.selectbox('Property type',(dict_unique['propertyType']))
street_type = st.sidebar.selectbox('Street type',(dict_unique['street']))
baths = st.sidebar.slider ("Bathrooms", min_value=min(dict_unique["baths"]), max_value=max(dict_unique["baths"]))
fireplace = st.sidebar.selectbox('Fireplace',(dict_unique['fireplace']))
# fireplace = st.sidebar.slider ("Fireplace", min_value= min(dict_unique["fireplace"]), max_value = max(dict_unique["fireplace"]))
sqft = st.sidebar.slider ("Living space in sqft", min_value= min(dict_unique["sqft"]), max_value = max(dict_unique["sqft"]))
beds = st.sidebar.slider ("Beds", min_value=min(dict_unique["beds"]), max_value=max(dict_unique["beds"]))
# fireplace = st.sidebar.slider ("Fireplace", min_value= min(dict_unique["fireplace"]), max_value = max(dict_unique["fireplace"]))
stories = st.sidebar.slider ("Stories", min_value=min(dict_unique["stories"]), max_value=max(dict_unique["stories"]))
private_pool = st.sidebar.selectbox('Private Pool',(dict_unique['private_pool']))
year_built = st.sidebar.slider ("Year built", min_value=min(dict_unique["year_built"]), max_value=max(dict_unique["year_built"]))
heating_type = st.sidebar.selectbox('heating',(dict_unique['heating']))
cooling = st.sidebar.selectbox('Cooling system',(dict_unique['cooling']))
lot_size = st.sidebar.slider ("Lot size in sqft", min_value= min(dict_unique["lot_size"]), max_value = max(dict_unique["lot_size"]))
school_rate_mean = st.sidebar.slider ("Schools' score", min_value= min(dict_unique["school_rate_mean"]), max_value = max(dict_unique["school_rate_mean"]))
school_distance_mean = st.sidebar.slider ("Schools' distance", min_value= min(dict_unique["school_distance_mean"]), max_value = max(dict_unique["school_distance_mean"]))
has_parking = st.sidebar.selectbox('Has parking?',(dict_unique['has_parking']))
has_garage = st.sidebar.selectbox('Has garage?',(dict_unique['garage']))
parking_spaces = st.sidebar.slider ("Parking spaces", min_value= min(dict_unique["parking_spaces"]), max_value = max(dict_unique["parking_spaces"]))
states_listed = st.sidebar.selectbox('State',(dict_unique['states_shoted']))
cities_listed = st.sidebar.selectbox('City',(dict_unique['cities_shorted']))
# rooms = st.sidebar.selectbox('Rooms',(dict_unique['rooms']))
# area = st.sidebar.slider ("Area", min_value=min(dict_unique["area"]), max_value = max(dict_unique["area"]))
# kitchen_area = st.sidebar.slider("Kitchen area", min_value=min(dict_unique["kitchen_area"]), max_value = max(dict_unique["kitchen_area"])) 

dict_data = {
    
    "status": status_type,
    "propertyType": object_type,
    "street": street_type,
    "baths": baths,
    "fireplace" : fireplace,
    "sqft": sqft,
    "beds":beds,
    "stories": stories,
    "private_pool": private_pool,
    "year_built": year_built,
    "heating": heating_type,
    "cooling": cooling,
    "lot_size": lot_size,
    "school_rate_mean": school_rate_mean,
    "school_distance_mean": school_distance_mean,
    "has_parking": has_parking,
    "garage": has_garage,
    "parking_spaces": parking_spaces,
    "states_shoted": states_listed,
    "cities_shorted": cities_listed
       
}

data_predict = pd.DataFrame([dict_data])
model = load_model(path_model)

button_1 = st.button("Hello")
# button_2 = st.button("Predict_price")
button_3 = st.button("Predict")

if button_1:
    st.write('Why hello there')
    
# if button_2:
#     model.predict(data_predict)

if button_3:
    output = model.predict(data_predict)
    st.success(f"{round(output[0])} rub")
    
    
    
    
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