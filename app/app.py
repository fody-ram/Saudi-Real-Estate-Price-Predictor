import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
import json

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Saudi Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ============================================================
# LOAD MODEL & ENCODERS
# ============================================================

@st.cache_resource
def load_model():
    import os
    # Works both locally and on Streamlit Cloud
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model       = joblib.load(os.path.join(base, 'models', 'xgb_model.pkl'))
    le_city     = joblib.load(os.path.join(base, 'models', 'le_city.pkl'))
    le_district = joblib.load(os.path.join(base, 'models', 'le_district.pkl'))
    le_type     = joblib.load(os.path.join(base, 'models', 'le_type.pkl'))
    features    = joblib.load(os.path.join(base, 'models', 'features.pkl'))
    return model, le_city, le_district, le_type, features

# ============================================================
# LOAD CITY & DISTRICT OPTIONS FROM DB
# ============================================================

@st.cache_data
def load_options():
    with open('app/cities.json', encoding='utf-8') as f:
        cities = json.load(f)
    with open('app/districts.json', encoding='utf-8') as f:
        districts = json.load(f)
    return cities, districts

cities, districts = load_options()

# ============================================================
# CITY → LAT/LNG LOOKUP (approximate city centers)
# ============================================================

city_coords = {
    'الرياض':        (24.7136,  46.6753),
    'جدة':           (21.4858,  39.1925),
    'مكة المكرمة':   (21.3891,  39.8579),
    'المدينة المنورة':(24.5247, 39.5692),
    'الدمام':        (26.4207,  50.0888),
    'الخبر':         (26.2172,  50.1971),
    'بريدة':         (26.3292,  43.9750),
    'الهفوف':        (25.3647,  49.5876),
    'الطائف':        (21.2703,  40.4158),
    'جازان':         (16.8892,  42.5511),
}

# ============================================================
# UI
# ============================================================

st.title("🏠 Saudi Real Estate Price Predictor")
st.markdown("Predict property prices across Saudi Arabia using machine learning.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("🌆 City", cities)
    district = st.selectbox("📍 District", districts)
    property_type = st.selectbox("🏗️ Property Type", ['Villa', 'Apartment', 'Land', 'Commercial'])
    area = st.number_input("📐 Area (sqm)", min_value=10, max_value=10000, value=300)

with col2:
    beds         = st.number_input("🛏️ Bedrooms",      min_value=0, max_value=20, value=3)
    wc           = st.number_input("🚿 Bathrooms",     min_value=0, max_value=20, value=2)
    livings      = st.number_input("🛋️ Living Rooms",  min_value=0, max_value=10, value=1)
    street_width = st.number_input("🛣️ Street Width (m)", min_value=0, max_value=60, value=15)
    age          = st.number_input("🏚️ Property Age (years)", min_value=0, max_value=100, value=5)

st.divider()

col3, col4, col5 = st.columns(3)
with col3:
    furnished = st.checkbox("🛋 Furnished")
with col4:
    ac = st.checkbox("❄️ AC")
with col5:
    ketchen = st.checkbox("🍳 Kitchen")

st.divider()

# ============================================================
# PREDICTION
# ============================================================

if st.button("💰 Predict Price", type="primary", use_container_width=True):

    # 1. Map UI English Property Types to the original Arabic strings used in training
    type_mapping = {
        'Villa': 'فيلا',
        'Apartment': 'شقة',
        'Land': 'أرض',
        'Commercial': 'تجاري'
    }
    mapped_property_type = type_mapping.get(property_type, property_type)
    model, le_city, le_district, le_type, features = load_model()

    # Encode categorical inputs
    city_enc     = le_city.transform([city])[0] if city in le_city.classes_ else 0
    district_enc = le_district.transform([district])[0] if district in le_district.classes_ else 0
    type_enc     = le_type.transform([property_type])[0] if property_type in le_type.classes_ else 0

    # 3. Get approximate lat/lng coordinates
    lat, lng = city_coords.get(city, (24.7136, 46.6753))

    # 4. Build input row using the EXACT column names from the notebook training phase
    input_data = pd.DataFrame([{
        'size'            : area,              # Notebook used 'size', not 'area'
        'beds'            : beds,
        'v_bathrooms'     : wc,                # Notebook used 'v_bathrooms', not 'wc'
        'v_living_rooms'  : livings,           # Notebook used 'v_living_rooms', not 'livings'
        'street_width'    : street_width,
        'age'             : age,
        'is_furnished'    : int(furnished),    # Notebook used 'is_furnished'
        'has_ac'          : int(ac),           # Notebook used 'has_ac'
        'has_kitchen'     : int(ketchen),      # Notebook used 'has_kitchen'
        'lat'             : lat,
        'lng'             : lng,
        'city_encoded'    : city_enc,          # Notebook used 'city_encoded'
        'district_encoded': district_enc,      # Notebook used 'district_encoded'
        'type_encoded'    : type_enc           # Notebook used 'type_encoded'
    }])

    # 5. Strictly align feature order to match the training matrices layout
    final_input = input_data[features]

    # 6. Predict and inverse-transform target log to real SAR currency
    log_pred   = model.predict(final_input)[0]
    price_pred = np.expm1(log_pred)

    # Display result safely
    st.success(f"### 💰 Estimated Price: {price_pred:,.0f} SAR")

    # Price range (±15%)
    low  = price_pred * 0.85
    high = price_pred * 1.15
    st.info(f"📊 Likely range: **{low:,.0f} SAR** → **{high:,.0f} SAR**")

    # Context display
    st.caption(f"Based on {property_type} in {city} | {area} sqm | {beds} beds | Age: {age} years")