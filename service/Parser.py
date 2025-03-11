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


SERVICE_ACCOUNT_FILE = "ee-beaver-lab-37f45e7fed2e.json"
EE_SCOPES = ['https://www.googleapis.com/auth/earthengine']

if not os.path.exists(SERVICE_ACCOUNT_FILE):
    raise FileNotFoundError("Service account JSON file not found in /app/")

with open(SERVICE_ACCOUNT_FILE, "r") as f:
    credentials_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    credentials_info, scopes=EE_SCOPES
)

ee.Initialize(credentials)


def set_id_year_property(feature):
    try:
        # Ensure feature has an ID; default to "unknown" if not present
        feature_id = feature.id() if feature.id() else "unknown"

        # Convert Earth Engine String to Python string for processing
        feature_id = feature_id.getInfo() if isinstance(feature_id, ee.ComputedObject) else feature_id

        # Extract the last two characters safely
        short_id = feature_id[-2:] if isinstance(feature_id, str) and len(feature_id) >= 2 else "NA"

        # Safely get the year from the date property
        date = feature.get("date")
        year = ee.Date(date).get("year").getInfo() if date else None

        # Add the new properties
        return feature.set("id_property", feature_id).set("year", year).set("DamID", short_id)
    except Exception as e:
        st.error(f"An error occurred during standardization: {e}")
        return feature  # Return the original feature if an error occurs

#### The format is slightly different in Jupyter notebook because it doesn't deal with streamlit syntax
def upload_points_to_ee(file):
    try:
        if file.name.endswith(".csv"):
            # Read CSV file
            df = pd.read_csv(file)

            # Debug: Check the DataFrame structure
            st.write("Uploaded CSV preview:")
            st.dataframe(df.head())  # Display the first few rows for debugging

            # Ensure required columns exist
            required_columns = {"longitude", "latitude", "date", "DamID"}
            if not required_columns.issubset(df.columns):
                st.error(f"CSV must have the following columns: {', '.join(required_columns)}.")
                return None


            if not pd.to_datetime(df["date"], errors="coerce").notnull().all():
                st.error("The 'date' column must be in a valid date format (e.g., YYYY-MM-DD).")
                return None

            # Convert to a list of Earth Engine points with standardization
            def standardize_feature(row):
                # Explicitly extract required values from the row
                longitude = float(row["longitude"])
                latitude = float(row["latitude"])
                dam_date = str(row["date"])
                dam_id = str(row["DamID"])

                # Include only required properties in the feature
                properties = {
                    "date": dam_date,
                    "DamID": dam_id,
                }

                # Create an Earth Engine feature
                feature = ee.Feature(ee.Geometry.Point([longitude, latitude]), properties)
                return set_id_year_property(feature)

            # Apply standardization to each row
            standardized_features = df.apply(standardize_feature, axis=1).tolist()
            feature_collection = ee.FeatureCollection(standardized_features)

            st.success("CSV successfully uploaded and standardized.")
            return feature_collection

        elif file.name.endswith(".geojson"):
            # Read GeoJSON file
            geojson = json.load(file)

            # Convert GeoJSON features to Earth Engine Features
            features = [
                ee.Feature(
                    ee.Geometry(geom["geometry"]),
                    geom.get("properties", {"id": i}),
                )
                for i, geom in enumerate(geojson["features"])
            ]
            feature_collection = ee.FeatureCollection(features)

            st.success("GeoJSON successfully uploaded and converted.")
            return feature_collection

        else:
            st.error("Unsupported file format. Please upload a CSV or GeoJSON file.")
            return None

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None



