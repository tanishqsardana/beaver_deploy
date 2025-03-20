import dateutil
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


credentials_info = {
    "type": st.secrets["gcp_service_account"]["type"],
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
    "private_key": st.secrets["gcp_service_account"]["private_key"],
    "client_email": st.secrets["gcp_service_account"]["client_email"],
    "client_id": st.secrets["gcp_service_account"]["client_id"],
    "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
    "token_uri": st.secrets["gcp_service_account"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
    "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
}

credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/earthengine"]
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

# TODO: Add a function to easily upload points to Earth Engine
#The format is slightly different in Jupyter notebook because it doesn't deal with streamlit syntax

def clean_coordinate(value):
    """Cleans and converts a coordinate value into a valid float."""
    try:
        value = str(value).strip().replace("°", "").replace(",", ".")
        value = value.replace("N", "").replace("S", "").replace("E", "").replace("W", "")
        return float(value)
    except ValueError:
        return None  # Return None if the value cannot be converted

def parse_date(value, date_format):
    """Parses a date value into a standardized YYYY-MM-DD format."""
    try:
        if date_format == "Auto Detect":
            return dateutil.parser.parse(str(value)).strftime("%Y-%m-%d")
        elif date_format == "Unix Timestamp":
            return pd.to_datetime(int(value), unit="s").strftime("%Y-%m-%d")
        else:
            return pd.to_datetime(value, format=date_format).strftime("%Y-%m-%d")
    except Exception:
        return None  # Return None if the date cannot be parsed

def upload_points_to_ee(file):
    """Handles CSV and GeoJSON file uploads, standardizes data, and converts them into an Earth Engine FeatureCollection."""
    try:
        if file.name.endswith(".csv"):
            # Reset file pointer before re-reading
            file.seek(0)

            # Allow the user to specify delimiter
            delimiter = st.selectbox("Select delimiter used in CSV:", [",", ";", "\t"], index=0)

            try:
                # Read CSV file with user-defined delimiter
                df = pd.read_csv(file, delimiter=delimiter, dtype=str, encoding="utf-8")

                if df.empty:
                    st.error("❌ No data found in the file. Please check the format.")
                    return None

            except pd.errors.EmptyDataError:
                st.error("❌ The file is empty or improperly formatted.")
                return None
            except pd.errors.ParserError:
                st.error("❌ Error parsing the file. Please check the delimiter and format.")
                return None
            except UnicodeDecodeError:
                st.error("❌ Encoding issue detected. Try saving the file with UTF-8 encoding.")
                return None

            # Display preview of the uploaded file
            st.write("📋 **Uploaded CSV preview:**")
            st.dataframe(df.head())

            # Reset file pointer again before re-reading
            file.seek(0)

            # Allow user to confirm header presence
            header_option = st.radio("Does the file contain headers?", ["Yes", "No"], index=0)

            # Reload the file based on user selection
            df = pd.read_csv(file, delimiter=delimiter, header=0 if header_option == "Yes" else None, encoding="utf-8")

            # Ensure that column selection is interactive
            longitude_col = st.selectbox("Select the **Longitude** column:", df.columns)
            latitude_col = st.selectbox("Select the **Latitude** column:", df.columns)
            date_col = st.selectbox("Select the **Date** column (optional):", ["None"] + list(df.columns))
            damid_col = st.selectbox("Select the **DamID** column (optional):", ["None"] + list(df.columns))

            # Date format selection
            date_format = st.selectbox(
                "Select the **Date format**:",
                ["Auto Detect", "YYYY-MM-DD", "MM/DD/YYYY", "DD-MM-YYYY", "Unix Timestamp"]
            )

            # Validate column selections
            if not longitude_col or not latitude_col:
                st.error("❌ Longitude and Latitude columns must be selected.")
                return None

            # Convert DataFrame rows into Earth Engine Features
            def standardize_feature(row):
                longitude = clean_coordinate(row[longitude_col])
                latitude = clean_coordinate(row[latitude_col])

                if longitude is None or latitude is None:
                    return None  # Skip rows with invalid coordinates

                # Handle date column
                dam_date = None
                if date_col != "None":
                    dam_date = parse_date(row[date_col], date_format)

                # Handle DamID column
                dam_id = row[damid_col] if damid_col != "None" else "unknown"

                # Create properties dictionary
                properties = {"DamID": dam_id}
                if dam_date:
                    properties["date"] = dam_date

                # Convert to Earth Engine feature
                return ee.Feature(ee.Geometry.Point([longitude, latitude]), properties)

            # Apply standardization and filter out invalid rows
            standardized_features = list(filter(None, df.apply(standardize_feature, axis=1).tolist()))
            feature_collection = ee.FeatureCollection(standardized_features)

            st.success("✅ CSV successfully uploaded and standardized.")
            return feature_collection

        elif file.name.endswith(".geojson"):
            # Reset file pointer before reading GeoJSON
            file.seek(0)

            # Read GeoJSON file
            try:
                geojson = json.load(file)
            except json.JSONDecodeError:
                st.error("❌ Invalid GeoJSON file format.")
                return None

            # Validate GeoJSON structure
            if "features" not in geojson or not isinstance(geojson["features"], list):
                st.error("❌ Invalid GeoJSON format: missing 'features' key.")
                return None

            # Convert GeoJSON features to Earth Engine Features
            features = []
            for i, feature_obj in enumerate(geojson["features"]):
                try:
                    geom = feature_obj.get("geometry")
                    props = feature_obj.get("properties", {"id": i})
                    features.append(ee.Feature(ee.Geometry(geom), props))
                except Exception as e:
                    st.warning(f"⚠️ Skipped feature {i} due to an error: {e}")

            feature_collection = ee.FeatureCollection(features)

            st.success("✅ GeoJSON successfully uploaded and converted.")
            return feature_collection

        else:
            st.error("❌ Unsupported file format. Please upload a CSV or GeoJSON file.")
            return None

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the file: {e}")
        return None
