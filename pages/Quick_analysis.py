import streamlit as st
import seaborn as sns
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import matplotlib.pyplot as plt
import geemap.foliumap as geemap
from streamlit_folium import folium_static
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, accuracy_score
from service.Sentinel2_functions import process_Sentinel2_with_cloud_coverage
from service.Export_dam_imagery import  S2_PixelExtraction_Export,Sentinel_Only_Export
from service.Negative_sample_functions import deduplicate_locations, prepareHydro,sampleNegativePoints
from service.Parser import upload_points_to_ee, set_id_year_property
from service.Visualize_trends import S2_Export_for_visual, compute_all_metrics,compute_lst,add_landsat_lst, add_landsat_lst_et,compute_all_metrics_up_downstream, S2_Export_for_visual_flowdir, compute_all_metrics_LST_ET
from service.Data_management import set_id_negatives, add_dam_buffer_and_standardize_date
import ee 
import csv
import io
import os
import numpy as np
import pandas as pd
# import gdal
import tempfile
import rasterio
from service.Validation_service import (
    validate_dam_waterway_distance,
    check_waterway_intersection,
    generate_validation_report,
    visualize_validation_results
)

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

ee.Initialize(credentials, project="ee-beaver-lab")

st.title("Quick Analysis")

# Step 1: Data upload
uploaded_file = st.file_uploader("Upload Dam Locations (CSV or GeoJSON)", type=["csv", "geojson"])

if uploaded_file:
    with st.spinner("Uploading data..."):
        try:
            pos_fc = upload_points_to_ee(uploaded_file, widget_prefix="Dam")
            st.success("Dam locations uploaded successfully!")
            if pos_fc:
                # Preview uploaded points
                preview_map = geemap.Map()
                preview_map.add_basemap("SATELLITE")
                preview_map.addLayer(pos_fc, {'color': 'blue'}, 'Uploaded Dams')
                preview_map.centerObject(pos_fc)
                preview_map.to_streamlit(height=400)

                data_confirmed = True

        except Exception as e:
            st.error(f"Error uploading data: {e}")
            data_confirmed = False
else:
    data_confirmed = False