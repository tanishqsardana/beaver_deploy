import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import matplotlib.pyplot as plt
import geemap.foliumap as geemap
from streamlit_folium import folium_static
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, accuracy_score
import ee 
import os
import numpy as np
import pandas as pd
# import gdal
import tempfile
import rasterio

import json
from google.oauth2 import service_account


SERVICE_ACCOUNT_FILE = "project-beaver-415818-a4eb6a21a9a8.json"
EE_SCOPES = ['https://www.googleapis.com/auth/earthengine']

if not os.path.exists(SERVICE_ACCOUNT_FILE):
    raise FileNotFoundError("Service account JSON file not found in /app/")

with open(SERVICE_ACCOUNT_FILE, "r") as f:
    credentials_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    credentials_info, scopes=EE_SCOPES
)

ee.Initialize(credentials)


def set_id_year_property_GEE_Collection(feature):
    # Get the feature id
    feature_id = feature.id()
    short_id = feature_id.slice(-2)  # Extract the last two characters
    date = ee.Date(feature.get('date'))
    year = date.get('year')
    # Set the feature id as a property
    return feature.set('id_property', feature_id).set('year', year).set('DamID', short_id)


def set_id_negatives(idx):
    idx = ee.Number(idx)
    feature = ee.Feature(features_list.get(idx))
    # Cast idx.add(1) to an integer and then format as a string without decimals.
    labeled_feature = feature.set(
        'id_property', ee.String('N').cat(idx.add(1).int().format())
    )
    return labeled_feature

def add_dam_buffer_and_standardize_date(feature):
    # Add Dam property and other metadata
    dam_status = feature.get("Dam")
    date = feature.get("date")
    formatted_date = ee.Date(date).format('YYYYMMdd')
    
    # Buffer geometry while retaining properties
    buffered_geometry = feature.geometry().buffer(buffer_radius)
    
    # Create a new feature with buffered geometry and updated properties
    return ee.Feature(buffered_geometry).set({
        "Dam": dam_status,
        "Survey_Date": ee.Date(date),
        "Damdate": ee.String("DamDate_").cat(formatted_date),
        "Point_geo": feature.geometry(),
        "id_property": feature.get("id_property")
    })