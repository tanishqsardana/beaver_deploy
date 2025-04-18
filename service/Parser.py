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
import csv
import io
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


####### Original chris parser- doesnt include widgit 
# def upload_points_to_ee(file):
#     """
#     Handles CSV and GeoJSON uploads, standardizes data, and converts them into an
#     Earth Engine FeatureCollection. Allows the user to assign a fixed date based on selected year.
#     """
#     if not file:
#         return None

#     try:
#         if file.name.endswith(".csv"):
#             file.seek(0)

#             delimiter_display = {",": "Comma (,)", ";": "Semicolon (;)", "\t": "Tab (\\t)"}
#             delimiter_key = st.selectbox("Select delimiter used in CSV:", list(delimiter_display.values()), index=0)
#             delimiter = [k for k, v in delimiter_display.items() if v == delimiter_key][0]

#             df_preview = pd.read_csv(file, delimiter=delimiter, dtype=str, encoding="utf-8", nrows=5, header=None)
#             st.write("**Preview of the uploaded file:**")
#             st.dataframe(df_preview)

#             file.seek(0)

#             header_option = st.radio("Does the file contain headers?", ["Yes", "No"], index=0)
#             if header_option == "Yes":
#                 df = pd.read_csv(file, delimiter=delimiter, header=0, encoding="utf-8")
#             else:
#                 df = pd.read_csv(file, delimiter=delimiter, header=None, encoding="utf-8")
#                 df.columns = [f"column{i}" for i in range(len(df.columns))]

#             longitude_col = st.selectbox("Select the **Longitude** column:", df.columns)
#             latitude_col = st.selectbox("Select the **Latitude** column:", df.columns)

#             # Year selection & default prompt
#             selected_year = st.selectbox("Select a year (default date will be July 1 of selected year):", list(range(2017, 2025)), index=3)
#             selected_date = f"{selected_year}-07-01"

#             # ET availability warning
#             if selected_year < 2020 or selected_year > 2025:
#                 st.warning("You may proceed to next steps, but ET data may not be available for the selected year.")

#             if st.button("Confirm and Process Data"):
#                 def standardize_feature(row):
#                     longitude = clean_coordinate(row[longitude_col])
#                     latitude = clean_coordinate(row[latitude_col])

#                     if longitude is None or latitude is None:
#                         return None

#                     properties = {"date": selected_date}
#                     return ee.Feature(ee.Geometry.Point([longitude, latitude]), properties)

#                 standardized_features = list(filter(None, df.apply(standardize_feature, axis=1).tolist()))
#                 feature_collection = ee.FeatureCollection(standardized_features)

#                 st.success("CSV successfully uploaded and standardized.")
#                 return feature_collection

#         elif file.name.endswith(".geojson"):
#             file.seek(0)
#             try:
#                 geojson = json.load(file)
#             except json.JSONDecodeError:
#                 st.error("Invalid GeoJSON file format.")
#                 return None

#             if "features" not in geojson or not isinstance(geojson["features"], list):
#                 st.error("Invalid GeoJSON format: missing 'features' key.")
#                 return None

#             # year selection & default prompt
#             selected_year = st.selectbox("Select a year (default date will be July 1 of selected year):", list(range(2017, 2025)), index=3)
#             selected_date = f"{selected_year}-07-01"

#             if selected_year < 2020 or selected_year > 2025:
#                 st.warning("You may proceed to next steps, but ET data may not be available for the selected year.")

#             if st.button("Confirm and Process GeoJSON"):
#                 features = []
#                 for i, feature_obj in enumerate(geojson["features"]):
#                     try:
#                         geom = feature_obj.get("geometry")
#                         props = feature_obj.get("properties", {"id": i})
#                         props["date"] = selected_date  # Add the selected date to the properties
#                         features.append(ee.Feature(ee.Geometry(geom), props))
#                     except Exception as e:
#                         st.warning(f"Skipped feature {i} due to an error: {e}")

#                 feature_collection = ee.FeatureCollection(features)
#                 st.success("GeoJSON successfully uploaded and converted.")
#                 return feature_collection

#         else:
#             st.error("Unsupported file format. Please upload a CSV or GeoJSON file.")
#             return None

#     except Exception as e:
#         st.error(f"An error occurred while processing the file: {e}")
#         return None

##### Modified parser to include widgit_prefix and autodetect header
def upload_points_to_ee(file, widget_prefix=""):
    if not file:
        return None

    try:
        if file.name.endswith(".csv"):
            file.seek(0)

            # Let the user select a delimiter
            delimiter_display = {",": "Comma (,)", ";": "Semicolon (;)", "\t": "Tab (\\t)"}
            delimiter_key = st.selectbox(
                "Select delimiter used in CSV:",
                list(delimiter_display.values()),
                index=0,
                key=f"{widget_prefix}_delimiter_selectbox"
            )
            delimiter = [k for k, v in delimiter_display.items() if v == delimiter_key][0]

            # Read a sample for header detection using csv.Sniffer
            sample = file.read(1024)
            try:
                sample_str = sample.decode("utf-8")
            except AttributeError:
                sample_str = sample  # already a string
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(sample_str)
            file.seek(0)  # Reset pointer

            # Read the CSV with or without headers
            if has_header:
                df = pd.read_csv(file, delimiter=delimiter, header=0, encoding="utf-8")
            else:
                df = pd.read_csv(file, delimiter=delimiter, header=None, encoding="utf-8")
                df.columns = [f"column{i}" for i in range(len(df.columns))]

            st.write("**Preview of the uploaded file:**")
            st.dataframe(df.head(5))

            # Auto-select Latitude and Longitude columns (case-insensitive)
            columns_lower = [col.lower() for col in df.columns]
            if "longitude" in columns_lower:
                default_longitude = list(df.columns).index(df.columns[columns_lower.index("longitude")])
            else:
                default_longitude = 0

            if "latitude" in columns_lower:
                default_latitude = list(df.columns).index(df.columns[columns_lower.index("latitude")])
            else:
                default_latitude = 1 if len(df.columns) > 1 else 0

            longitude_col = st.selectbox(
                "Select the **Longitude** column:",
                options=df.columns,
                index=default_longitude,
                key=f"{widget_prefix}_longitude_selectbox"
            )
            latitude_col = st.selectbox(
                "Select the **Latitude** column:",
                options=df.columns,
                index=default_latitude,
                key=f"{widget_prefix}_latitude_selectbox"
            )

            # Year selection & default prompt
            selected_year = st.selectbox(
                "Select a year:",
                list(range(2017, 2025)),
                index=3,
                key=f"{widget_prefix}_year_selectbox"
            )
            selected_date = f"{selected_year}-07-01"

            if selected_year < 2020 or selected_year > 2025:
                st.warning("You may proceed to next steps, but ET data may not be available for the selected year.")

            if st.button("Confirm and Process Data", key=f"{widget_prefix}_process_data_button"):
                def standardize_feature(row):
                    longitude = clean_coordinate(row[longitude_col])
                    latitude = clean_coordinate(row[latitude_col])
                    if longitude is None or latitude is None:
                        return None
                    properties = {"date": selected_date}
                    return ee.Feature(ee.Geometry.Point([longitude, latitude]), properties)

                standardized_features = list(filter(None, df.apply(standardize_feature, axis=1).tolist()))
                feature_collection = ee.FeatureCollection(standardized_features)

                st.success("CSV successfully uploaded and standardized. Preview the data on the map below.")
                return feature_collection

        elif file.name.endswith(".geojson"):
            file.seek(0)
            try:
                geojson = json.load(file)
            except json.JSONDecodeError:
                st.error("Invalid GeoJSON file format.")
                return None

            if "features" not in geojson or not isinstance(geojson["features"], list):
                st.error("Invalid GeoJSON format: missing 'features' key.")
                return None

            # Year selection & default prompt for GeoJSON
            selected_year = st.selectbox(
                "Select a year:",
                list(range(2017, 2025)),
                index=3,
                key=f"{widget_prefix}_geojson_year_selectbox"
            )
            selected_date = f"{selected_year}-07-01"

            if selected_year < 2020 or selected_year > 2025:
                st.warning("You may proceed to next steps, but ET data may not be available for the selected year.")

            if st.button("Confirm and Process GeoJSON", key=f"{widget_prefix}_geojson_process_button"):
                features = []
                for i, feature_obj in enumerate(geojson["features"]):
                    try:
                        geom = feature_obj.get("geometry")
                        props = feature_obj.get("properties", {"id": i})
                        props["date"] = selected_date  # Add the selected date to the properties
                        features.append(ee.Feature(ee.Geometry(geom), props))
                    except Exception as e:
                        st.warning(f"Skipped feature {i} due to an error: {e}")

                feature_collection = ee.FeatureCollection(features)
                st.success("GeoJSON successfully uploaded and converted.")
                return feature_collection

        else:
            st.error("Unsupported file format. Please upload a CSV or GeoJSON file.")
            return None

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None



# Older Version of the function
# def upload_points_to_ee(file):
#     try:
#         if file.name.endswith(".csv"):
#             # Read CSV file
#             df = pd.read_csv(file)

#             # Debug: Check the DataFrame structure
#             st.write("Uploaded CSV preview:")
#             st.dataframe(df.head())  # Display the first few rows for debugging

#             # Ensure required columns exist
#             required_columns = {"longitude", "latitude", "date", "DamID"}
#             if not required_columns.issubset(df.columns):
#                 st.error(f"CSV must have the following columns: {', '.join(required_columns)}.")
#                 return None


#             if not pd.to_datetime(df["date"], errors="coerce").notnull().all():
#                 st.error("The 'date' column must be in a valid date format (e.g., YYYY-MM-DD).")
#                 return None

#             # Convert to a list of Earth Engine points with standardization
#             def standardize_feature(row):
#                 # Explicitly extract required values from the row
#                 longitude = float(row["longitude"])
#                 latitude = float(row["latitude"])
#                 dam_date = str(row["date"])
#                 dam_id = str(row["DamID"])

#                 # Include only required properties in the feature
#                 properties = {
#                     "date": dam_date,
#                     "DamID": dam_id,
#                 }

#                 # Create an Earth Engine feature
#                 feature = ee.Feature(ee.Geometry.Point([longitude, latitude]), properties)
#                 return set_id_year_property(feature)

#             # Apply standardization to each row
#             standardized_features = df.apply(standardize_feature, axis=1).tolist()
#             feature_collection = ee.FeatureCollection(standardized_features)

#             st.success("CSV successfully uploaded and standardized.")
#             return feature_collection

#         elif file.name.endswith(".geojson"):
#             # Read GeoJSON file
#             geojson = json.load(file)

#             # Convert GeoJSON features to Earth Engine Features
#             features = [
#                 ee.Feature(
#                     ee.Geometry(geom["geometry"]),
#                     geom.get("properties", {"id": i}),
#                 )
#                 for i, geom in enumerate(geojson["features"])
#             ]
#             feature_collection = ee.FeatureCollection(features)

#             st.success("GeoJSON successfully uploaded and converted.")
#             return feature_collection

#         else:
#             st.error("Unsupported file format. Please upload a CSV or GeoJSON file.")
#             return None

#     except Exception as e:
#         st.error(f"An error occurred while processing the file: {e}")
#         return None