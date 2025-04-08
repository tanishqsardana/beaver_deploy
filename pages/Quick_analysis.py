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

if "Dam_data" not in st.session_state:
    st.session_state.Dam_data = None
if "Buffer_map" not in st.session_state:
    st.session_state.Buffer_map = None
# Step 1: Data upload   
uploaded_file = st.file_uploader("Upload Dam Locations (CSV or GeoJSON)", type=["csv", "geojson"])

if uploaded_file:
    with st.spinner("Uploading data..."):
        try:
            feature_collection = upload_points_to_ee(uploaded_file, widget_prefix="Dam")
            st.success("Dam locations uploaded successfully!")
            if feature_collection:
                # Preview uploaded points
                st.session_state.Positive_collection = feature_collection  # Save to session state
                st.session_state['Full_positive'] = st.session_state.Positive_collection
                # Positives_map = geemap.Map()
                # Positives_map.add_basemap("SATELLITE")
                # Positives_map.centerObject(st.session_state['Full_positive'])
                # Positives_map.addLayer(st.session_state['Full_positive'],{'color': 'blue'},'Dams')

                # st.write("Dam Locations (blue points):")
                # Positives_map.to_streamlit(width=1200, height=700)

        except Exception as e:
            st.error(f"Error uploading data: {e}")
            data_confirmed = False
else:
    data_confirmed = False



if 'Positive_collection' in st.session_state:
    st.header("Select Buffer Parameters")

    innerRadius = st.number_input("Inner Radius (meters)", value=200, min_value=0, step=50)
    outerRadius = st.number_input("Outer Radius (meters)", value=2000, min_value=0, step=100)
    buffer_radius = st.number_input("Buffer Radius (meters)", value=200, min_value=1, step=10)

    if st.button("Generate Buffer Map"):
        with st.spinner("Processing data..."):
            try:
                # Filter states by Dam bounding box
                positive_dam_bounds = st.session_state['Full_positive'].geometry().bounds()
                states_dataset = ee.FeatureCollection("TIGER/2018/States")
                states_with_dams = states_dataset.filterBounds(positive_dam_bounds)
                st.session_state['Positive_dam_state'] = states_with_dams
                states_geo = st.session_state['Positive_dam_state']
                state_names = states_geo.aggregate_array("NAME").getInfo()
                state_initials = {
                    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
                    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
                    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
                    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
                    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
                    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
                    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
                    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
                    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
                    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
                    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
                }

                nhd_collections = []
                for state in state_names:
                    state_initial = state_initials.get(state)
                    if state_initial:
                        nhd_dataset = ee.FeatureCollection(
                            f'projects/sat-io/open-datasets/NHD/NHD_{state_initial}/NHDFlowline'
                        )
                        nhd_collections.append(nhd_dataset)

                merged_nhd = ee.FeatureCollection(nhd_collections).flatten()
                st.session_state.selected_waterway = merged_nhd
                st.session_state['Waterway'] = merged_nhd

                st.session_state.Combined_collection = None

                first_pos = st.session_state.Positive_collection.first()
                date = ee.Date(first_pos.get('date'))
                year_string = date.format('YYYY')
                full_date = ee.String(year_string).cat('-07-01')

                positive_dams_fc = deduplicate_locations(st.session_state.Positive_collection)

                # Convert waterway feature collection to raster
                waterway_fc = st.session_state.selected_waterway
                hydroRaster = prepareHydro(waterway_fc)

                # Sample negative points
                negativePoints = sampleNegativePoints(
                    positive_dams_fc, 
                    hydroRaster, 
                    innerRadius, 
                    outerRadius, 
                    10
                )
                negativePoints = negativePoints.map(
                    lambda feature: feature.set('Dam', 'negative').set("date", full_date)
                )

                fc = negativePoints
                features_list = fc.toList(fc.size())
                indices = ee.List.sequence(0, fc.size().subtract(1))

                def set_id_negatives2(idx):
                    idx = ee.Number(idx)
                    feature = ee.Feature(features_list.get(idx))
                    return feature.set(
                        'id_property', ee.String('N').cat(idx.add(1).int().format())
                    )

                Neg_points_id = ee.FeatureCollection(indices.map(set_id_negatives2))

                Pos_collection = st.session_state.Positive_collection.map(
                    lambda feature: feature.set('Dam', 'positive')
                )

                pos_features_list = Pos_collection.toList(Pos_collection.size())
                pos_indices = ee.List.sequence(0, Pos_collection.size().subtract(1))

                def set_id_positives(idx):
                    idx = ee.Number(idx)
                    feature = ee.Feature(pos_features_list.get(idx))
                    return feature.set(
                        'id_property', ee.String('P').cat(idx.add(1).int().format())
                    )

                Positive_dam_id = ee.FeatureCollection(pos_indices.map(set_id_positives))
                Merged_collection = Positive_dam_id.merge(Neg_points_id)
                st.session_state['Merged_collection'] = Merged_collection

                def add_dam_buffer_and_standardize_date(feature):
                    dam_status = feature.get("Dam")
                    standardized_date = date  # forced date
                    formatted_date = date.format('YYYYMMdd')
                    buffered_geometry = feature.geometry().buffer(buffer_radius)
                    return ee.Feature(buffered_geometry).set({
                        "Dam": dam_status,
                        "Survey_Date": standardized_date,
                        "Damdate": ee.String("DamDate_").cat(formatted_date),
                        "Point_geo": feature.geometry(),
                        "id_property": feature.get("id_property")
                    })

                Buffered_collection = Merged_collection.map(add_dam_buffer_and_standardize_date)
                Dam_data = Buffered_collection.select(['id_property', 'Dam', 'Survey_Date', 'Damdate', 'Point_geo'])
                st.session_state['Dam_data'] = Dam_data

                dam_bounds = Dam_data.geometry().bounds()
                states_with_dams = states_dataset.filterBounds(dam_bounds)
                st.session_state['Dam_state'] = states_with_dams

                Negative = Dam_data.filter(ee.Filter.eq('Dam', 'negative'))
                Positive = Dam_data.filter(ee.Filter.eq('Dam', 'positive'))

                # Create and store map in session_state
                Buffer_map = geemap.Map()
                Buffer_map.add_basemap("SATELLITE")
                Buffer_map.addLayer(Negative, {'color': 'red'}, 'Negative')
                Buffer_map.addLayer(Positive, {'color': 'blue'}, 'Positive')
                Buffer_map.centerObject(Dam_data)
                
                # Store the map object so it won't disappear
                st.session_state.Buffer_map = Buffer_map

                st.success("Buffer map generated successfully!")
            except Exception as e:
                st.error(f"Error generating buffer map: {e}")


# -------------------- Always Show the Buffer Map (if we have it) --------------------
if st.session_state.Buffer_map:
    st.subheader("Buffered Dam Map")
    st.session_state.Buffer_map.to_streamlit(width=800, height=600)


# -------------------- Visualization Buttons --------------------
if st.session_state['Dam_data']:
    col1, col2 = st.columns(2)
    with col1:
        clicked_all_area = st.button("Visualize All Area")
    with col2:
        clicked_up_down = st.button("Visualize Upstream & Downstream")


    if clicked_all_area:
        with st.spinner("Processing visualization for all areas..."):
            try:
   
                Dam_data = st.session_state['Dam_data']
                waterway_fc = st.session_state['Waterway']
                total_count = Dam_data.size().getInfo()
                batch_size = 10
                num_batches = (total_count + batch_size - 1) // batch_size
                dam_list = Dam_data.toList(total_count)
                df_list = []

                progress_bar = st.progress(0)
                for i in range(num_batches):
                    batch = dam_list.slice(i * batch_size, min(total_count, (i + 1) * batch_size))
                    dam_batch = ee.FeatureCollection(batch)
                    S2_IC_batch = S2_Export_for_visual(dam_batch)
                    results_batch = S2_IC_batch.map(add_landsat_lst_et).map(compute_all_metrics_LST_ET)
                    df_batch = geemap.ee_to_df(ee.FeatureCollection(results_batch))
                    df_list.append(df_batch)
                    progress_bar.progress((i+1)/num_batches)

                final_df = pd.concat(df_list)
                st.session_state.final_df = final_df

                # Convert to DataFrame
                # df_lst = geemap.ee_to_df(final_df)
                final_df['Image_month'] = pd.to_numeric(final_df['Image_month'])
                final_df['Image_year'] = pd.to_numeric(final_df['Image_year'])
                final_df['Dam_status'] = final_df['Dam_status'].replace({'positive': 'Dam', 'negative': 'Non-dam'})

                # --- Produce some charts ---
                # (Below is just your original logic; be sure to call st.pyplot(fig)!)
                import seaborn as sns

                fig, axes = plt.subplots(4, 1, figsize=(12, 18))

                metrics = ['NDVI', 'NDWI_Green', 'LST', 'ET']
                titles = ['NDVI', 'NDWI Green', 'LST (°C)', 'ET']

                for ax, metric, title in zip(axes, metrics, titles):
                    sns.lineplot(
                        data=final_df, x="Image_month", y=metric, 
                        hue="Dam_status", style="Dam_status",
                        markers=True, dashes=False, ax=ax
                    )
                    ax.set_title(f'{title} by Month', fontsize=14)
                    ax.set_xticks(range(1, 13))

                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Visualization error: {e}")

    if clicked_up_down:
        with st.spinner("Processing upstream and downstream visualization..."):
            try:
                Dam_data = st.session_state['Dam_data']
                waterway_fc = st.session_state['Waterway']
                total_count = Dam_data.size().getInfo()
                batch_size = 10
                num_batches = (total_count + batch_size - 1) // batch_size
                dam_list = Dam_data.toList(total_count)
                df_list = []

                progress_bar = st.progress(0)
                for i in range(num_batches):
                    batch = dam_list.slice(i * batch_size, min(total_count, (i + 1) * batch_size))
                    dam_batch = ee.FeatureCollection(batch)
                    S2_IC_batch = S2_Export_for_visual_flowdir(dam_batch, waterway_fc)
                    results_batch = S2_IC_batch.map(add_landsat_lst_et).map(compute_all_metrics_up_downstream)
                    df_batch = geemap.ee_to_df(ee.FeatureCollection(results_batch))
                    df_list.append(df_batch)
                    progress_bar.progress((i+1)/num_batches)

                final_df = pd.concat(df_list)
                st.session_state.final_df = final_df

                import seaborn as sns

                fig2, axes2 = plt.subplots(4, 1, figsize=(12, 20))

                def melt_and_plot(df, metric, ax):
                    up_col = f"{metric}_up"
                    down_col = f"{metric}_down"
                    melted = df.melt(
                        ['Image_year', 'Image_month', 'Dam_status'],
                        [up_col, down_col],
                        'Flow', metric
                    )
                    melted['Flow'].replace({
                        up_col: 'Upstream',
                        down_col: 'Downstream'
                    }, inplace=True)
                    sns.lineplot(
                        data=melted, x='Image_month', y=metric, 
                        hue='Dam_status', style='Flow',
                        markers=True, dashes=False, ax=ax
                    )
                    ax.set_title(f"{metric.upper()} by Month (Upstream vs Downstream)")
                    ax.set_xticks(range(1, 13))

                for ax, met in zip(axes2, ['NDVI', 'NDWI', 'LST', 'ET']):
                    melt_and_plot(final_df, met, ax)

                plt.tight_layout()
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Upstream/downstream visualization error: {e}")