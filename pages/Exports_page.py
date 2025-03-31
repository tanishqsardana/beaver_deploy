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

# st.set_page_config(layout="wide")

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



# Initialize session state variables
if "Positive_collection" not in st.session_state:
    st.session_state.Positive_collection = None



# Streamlit UI
st.title("Analyzing the Impact of Beaver Dams")

# Section: Upload Points
st.header("Upload Dam Locations")
st.write(
    "The points must contain the following properties: longitude, latitude, Dam (positive or negative), date (YYYY-MM-DD), DamID."
)

# File upload
uploaded_file = st.file_uploader("Choose a CSV or GeoJSON file", type=["csv", "geojson"],key="Dam_file_uploader")

if uploaded_file:
    # feature_collection = upload_points_to_ee(uploaded_file)
    feature_collection = upload_points_to_ee(uploaded_file, widget_prefix="Dam")
    if feature_collection:
        st.session_state.Positive_collection = feature_collection  # Save to session state
        st.session_state['Full_positive'] = st.session_state.Positive_collection
    else:
        st.error("Please upload a dataset and match the columns.")

if 'Full_positive' in st.session_state:
    try:
        # Check if data is empty
        if st.session_state['Full_positive'].size().getInfo() == 0:
            st.error("Uploaded data is empty. Please upload data again.")
        else:
            Positives_map = geemap.Map()
            Positives_map.add_basemap("SATELLITE")
            
            # Add error handling
            try:
                Positives_map.centerObject(st.session_state['Full_positive'])
                Positives_map.addLayer(st.session_state['Full_positive'],{'color': 'blue'},'Dams')
                st.write("Dam Locations (blue points):")
                Positives_map.to_streamlit(width=1200, height=700)
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")
                st.write("Debug information:")
                st.write(f"Data size: {st.session_state['Full_positive'].size().getInfo()}")
                st.write(f"Data type: {type(st.session_state['Full_positive'])}")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")


# Section: Draw Points
# st.subheader("Draw Points (optional)")
# enable_drawing = st.checkbox("Enable drawing on the map")

# # Initialize map
# map_center = [39.7538, -98.4439]
# draw_map = folium.Map(location=map_center, zoom_start=4)

# folium.TileLayer(
#     tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#     attr="Esri",
#     name="Esri Satellite",
#     overlay=False,
#     control=True,
# ).add_to(draw_map)

# # Add uploaded points to the map
# if st.session_state.Positive_collection:
#     geojson_layer = geemap.ee_to_geojson(st.session_state.Positive_collection)
#     folium.GeoJson(geojson_layer, name="Uploaded Points").add_to(draw_map)

# # Add drawing functionality if enabled
# if enable_drawing:
#     draw = Draw(
#         export=True,
#         filename="points.geojson",
#         draw_options={
#             "rectangle": False,
#             "polygon": False,
#             "circle": False,
#             "polyline": False,
#             "marker": True,  # Enable marker tool for points
#         },
#         edit_options={"remove": True},
#     )
#     draw.add_to(draw_map)

# folium.LayerControl().add_to(draw_map)
# st_data = st_folium(draw_map, width=1200, height=700, key="main_map")

# # Process drawn points and append to points list
# points_list = []
# if enable_drawing and st_data and "all_drawings" in st_data:
#     geojson_list = st_data["all_drawings"]
#     if geojson_list:
#         for geojson in geojson_list:
#             if geojson and "geometry" in geojson:
#                 coordinates = geojson["geometry"]["coordinates"]
#                 points_list.append(coordinates)




# if points_list:
#     ee_points = ee.FeatureCollection(
#         [ee.Feature(ee.Geometry.Point(coord), {"id": idx}) for idx, coord in enumerate(points_list)]
#     )
#     if st.session_state.Positive_collection:
#         st.session_state.Positive_collection = st.session_state.Positive_collection.merge(ee_points)
#     else:
#         st.session_state.Positive_collection = ee_points
if "selected_waterway" not in st.session_state:
    st.session_state.selected_waterway = None  # Selected hydro dataset
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False  # Track if a waterway dataset is loaded



if 'Full_positive' in st.session_state:
    st.header("Select Waterway")
    upload_own_checkbox = st.checkbox("Upload Own Dataset")
    choose_existing_checkbox = st.checkbox("Choose an Existing Dataset")

    if upload_own_checkbox:
        asset_id = st.text_input("Enter the GEE Asset Table ID for your dataset (e.g., projects/ee-beaver-lab/assets/Hydro/MA_Hydro_arc):")
        if st.button("Load Uploaded Dataset"):
            try:
                waterway_own = ee.FeatureCollection(asset_id)
                st.session_state.selected_waterway = waterway_own
                st.session_state.dataset_loaded = True
                st.success("Uploaded dataset loaded and added to the map.")
            except Exception as e:
                st.error(f"Failed to load the dataset. Error: {e}")

    if choose_existing_checkbox:
        positive_dam_bounds = st.session_state['Full_positive'].geometry().bounds()
        states_dataset = ee.FeatureCollection("TIGER/2018/States")  # US States boundaries dataset
        states_with_dams = states_dataset.filterBounds(positive_dam_bounds)
        st.session_state['Positive_dam_state'] = states_with_dams
        states_geo = st.session_state['Positive_dam_state']
        state_names = states_geo.aggregate_array("NAME").getInfo()

        if not state_names:
            st.error("No states found within the Dam data bounds.")
        else:
            st.write(f"States within Dam data bounds: {state_names}")

        # Dropdown for dataset options
        dataset_option = st.selectbox(
            "Choose a dataset for waterways:",
            ["Choose", "WWF Free Flowing Rivers", "NHD by State"]
        )

        # Button to confirm dataset selection
        if st.button("Load Existing Dataset"):
            try:
                if dataset_option == "WWF Free Flowing Rivers":
                    wwf_dataset = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
                    clipped_wwf = wwf_dataset.filterBounds(states_with_dams)
                    st.session_state.selected_waterway = clipped_wwf
                    st.session_state.dataset_loaded = True
                    st.success("WWF dataset loaded and added to the map.")

                elif dataset_option == "NHD by State":
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

                    # Merge all NHD datasets
                    if nhd_collections:
                        merged_nhd = ee.FeatureCollection(nhd_collections).flatten()
                        st.session_state.selected_waterway = merged_nhd
                        st.session_state['Waterway'] = st.session_state.selected_waterway
                        st.session_state.dataset_loaded = True
                        st.success("NHD datasets for selected states loaded and added to the map.")
                    else:
                        st.error("No NHD datasets found for the selected states.")
            except Exception as e:
                st.error(f"Failed to load the dataset. Error: {e}")

    # Display the map
    if 'Waterway' in st.session_state:
        Waterway_map = geemap.Map()
        Waterway_map.add_basemap("SATELLITE")
        Waterway_map.centerObject(st.session_state['Full_positive'])
        Waterway_map.addLayer(st.session_state['Full_positive'],{'color': 'FF0000'},'Dams')
        Waterway_map.addLayer(st.session_state.selected_waterway, {"color": "blue"}, "Selected Waterway")
        Waterway_map.to_streamlit(width=1200, height=700)



# Combined Dam Status and Buffering Section
if "Combined_collection" not in st.session_state:
    st.session_state.Combined_collection = None

if "buffer_radius" not in st.session_state:
    st.session_state.buffer_radius = 200
if "year_selection" not in st.session_state:
    st.session_state.year_selection = 2020

# After waterway selection
if 'Waterway' in st.session_state:
    # Initialize validation state if not exists
    if 'validation_complete' not in st.session_state:
        st.session_state['validation_complete'] = False
    if 'use_all_dams' not in st.session_state:
        st.session_state['use_all_dams'] = True
    if 'validation_step' not in st.session_state:
        st.session_state['validation_step'] = 'initial'
    if 'show_non_dam_section' not in st.session_state:
        st.session_state['show_non_dam_section'] = False
    if 'buffer_complete' not in st.session_state:
        st.session_state['buffer_complete'] = False
        
    # Only show validation section if validation is not complete
    if not st.session_state['validation_complete']:
        st.header("Validate Dam Locations")
        
        # Add validation parameters
        max_distance = st.number_input(
            "Maximum allowed distance from waterway (meters):",
            min_value=0,
            value=100,
            step=10,
            key="max_distance_input"
        )
        
        # Validate positive dams
        if 'Full_positive' in st.session_state:
            if st.button("Validate Dam Locations"):
                with st.spinner("Validating dam locations..."):
                    try:
                        # Check if inputs are valid
                        if not st.session_state['Full_positive'] or not st.session_state['Waterway']:
                            st.error("Invalid input data: feature collection or waterway data is missing")
                        else:
                            # Perform distance validation
                            distance_validation = validate_dam_waterway_distance(
                                st.session_state['Full_positive'],
                                st.session_state['Waterway'],
                                max_distance
                            )
                            
                            # Perform intersection validation
                            intersection_validation = check_waterway_intersection(
                                st.session_state['Full_positive'],
                                st.session_state['Waterway']
                            )
                            
                            # Combine validation results
                            validation_results = {
                                **distance_validation,
                                **intersection_validation
                            }
                            
                            # Display validation report
                            st.subheader("Validation Report")
                            st.text(generate_validation_report(validation_results))
                            
                            # Display validation map
                            st.subheader("Validation Map")
                            validation_map = visualize_validation_results(
                                st.session_state['Full_positive'],
                                st.session_state['Waterway'],
                                validation_results
                            )
                            validation_map.to_streamlit(width=1200, height=700)
                            
                            # Store validation results in session state
                            st.session_state['validation_results'] = validation_results
                            st.session_state['validation_step'] = 'show_options'
                            
                    except Exception as e:
                        st.error(f"Error during validation: {str(e)}")
                        st.write("Debug information:")
                        st.write(f"Data size: {st.session_state['Full_positive'].size().getInfo()}")
                        st.write(f"Data type: {type(st.session_state['Full_positive'])}")
                        st.write(f"Waterway data size: {st.session_state['Waterway'].size().getInfo()}")
                        st.write(f"Waterway data type: {type(st.session_state['Waterway'])}")
            
            # Show options after validation is complete
            if st.session_state['validation_step'] == 'show_options':
                if st.session_state['validation_results']['invalid_count'].getInfo() > 0:
                    st.warning("Some dam locations have been identified as potentially invalid.")
                    st.write("Please review the validation report and map above.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Continue with all dams", key="use_all_dams_btn"):
                            st.session_state['validation_complete'] = True
                            st.session_state['use_all_dams'] = True
                            st.session_state['Dam_data'] = st.session_state['Full_positive']
                            st.session_state['show_non_dam_section'] = True
                            st.rerun()
                    with col2:
                        if st.button("Use only valid dams", key="use_valid_dams_btn"):
                            st.session_state['validation_complete'] = True
                            st.session_state['use_all_dams'] = False
                            # Ensure only valid dams are used
                            valid_dams = st.session_state['validation_results']['valid_dams']
                            # Update all related dam data
                            st.session_state['Dam_data'] = valid_dams
                            st.session_state['Full_positive'] = valid_dams
                            st.session_state['Positive_collection'] = valid_dams  # update Positive_collection
                            st.session_state['show_non_dam_section'] = True
                            st.rerun()
                else:
                    st.success("All dam locations are valid!")
                    st.session_state['validation_complete'] = True
                    st.session_state['use_all_dams'] = True
                    st.session_state['Dam_data'] = st.session_state['Full_positive']
                    st.session_state['show_non_dam_section'] = True
                    st.rerun()
    else:
        # Display validation results
        st.header("Validate Dam Locations")
        st.success("✅ Validation completed successfully!")
        if st.session_state['use_all_dams']:
            st.info("Using all dam locations for analysis.")
        else:
            st.info("Using only valid dam locations for analysis.")
        if 'validation_results' in st.session_state:
            st.subheader("Validation Report")
            st.text(generate_validation_report(st.session_state['validation_results']))
            st.subheader("Validation Map")
            validation_map = visualize_validation_results(
                st.session_state['Full_positive'],
                st.session_state['Waterway'],
                st.session_state['validation_results']
            )
            validation_map.to_streamlit(width=1200, height=700)

    # Show non-dam section after validation is complete
    if st.session_state['validation_complete'] and st.session_state['show_non_dam_section']:
        st.header("Upload or Generate Non-Dam Locations")
        upload_negatives_checkbox = st.checkbox("Upload Non-Dam Dataset (must be on a waterbody)")
        generate_negatives_checkbox = st.checkbox("Generate Non-Dam Locations")

        if upload_negatives_checkbox:
            uploaded_negatives_file = st.file_uploader("Choose a CSV or GeoJSON file", type=["csv", "geojson"],key="Non_Dam_file_uploader")
            if uploaded_negatives_file:
                negative_feature_collection = upload_points_to_ee(uploaded_negatives_file,widget_prefix="NonDam")
                if negative_feature_collection:
                    st.session_state.Negative_upload_collection = negative_feature_collection
                    st.session_state['Full_negative'] = st.session_state.Negative_upload_collection

                    first_pos = st.session_state.Positive_collection.first()
                    date = ee.Date(first_pos.get('date'))
                    year_string = date.format('YYYY')
                    full_date = ee.String(year_string).cat('-07-01')
                
                    negativePoints = negative_feature_collection.map(lambda feature: feature.set('Dam', 'negative').set("date",full_date))
                    
                    fc = negativePoints
                    features_list = fc.toList(fc.size())
                    indices = ee.List.sequence(0, fc.size().subtract(1))
                    
                    def set_id_negatives2(idx):
                        idx = ee.Number(idx)
                        feature = ee.Feature(features_list.get(idx))
                        labeled_feature = feature.set(
                            'id_property', ee.String('N').cat(idx.add(1).int().format())
                        )
                        return labeled_feature
                    Neg_points_id = ee.FeatureCollection(indices.map(set_id_negatives2))

                    Pos_collection = st.session_state.Positive_collection
                    Pos_collection = Pos_collection.map(lambda feature: feature.set('Dam', 'positive'))

                    pos_features_list = Pos_collection.toList(Pos_collection.size())
                    pos_indices = ee.List.sequence(0, Pos_collection.size().subtract(1))

                    def set_id_positives(idx):
                        idx = ee.Number(idx)
                        feature = ee.Feature(pos_features_list.get(idx))
                        labeled_feature = feature.set(
                            'id_property', ee.String('P').cat(idx.add(1).int().format())
                        )
                        return labeled_feature

                    Positive_dam_id = ee.FeatureCollection(pos_indices.map(set_id_positives))
                    Merged_collection = Positive_dam_id.merge(Neg_points_id)
                    st.session_state['Merged_collection'] = Merged_collection
                    st.session_state['buffer_complete'] = True

                    Negatives_map = geemap.Map()
                    Negatives_map.add_basemap("SATELLITE")
                    Negatives_map.addLayer(negativePoints,{'color': 'red', 'width': 2},'Negative')
                    Negatives_map.addLayer(Positive_dam_id,{'color': 'blue'},'Positive')
                    Negatives_map.centerObject(Merged_collection)
                    Negatives_map.to_streamlit(width=1200, height=700)

        if generate_negatives_checkbox:
            st.subheader("Specify the parameters for negative point generation:")

            innerRadius = st.number_input("Inner Radius (meters)", value=500, min_value=0, step=50, key="inner_radius_input")
            outerRadius = st.number_input("Outer Radius (meters)", value=5000, min_value=0, step=100, key="outer_radius_input")
            samplingScale = st.number_input("Sampling Scale (meters)", value=10, min_value=1, step=1, key="sampling_scale_input")
            
            if st.button("Generate Negative Points"):
                with st.spinner("Generating negative points..."):
                    first_pos = st.session_state.Positive_collection.first()
                    date = ee.Date(first_pos.get('date'))
                    year_string = date.format('YYYY')
                    full_date = ee.String(year_string).cat('-07-01')
                
                    positive_dams_fc = deduplicate_locations(st.session_state.Positive_collection)
                    waterway_fc = st.session_state.selected_waterway
                    hydroRaster = prepareHydro(waterway_fc)
                    
                    negativePoints = sampleNegativePoints(
                        positive_dams_fc, 
                        hydroRaster, 
                        innerRadius, 
                        outerRadius, 
                        samplingScale
                    )
                    negativePoints = negativePoints.map(lambda feature: feature.set('Dam', 'negative').set("date",full_date))
                    
                    fc = negativePoints
                    features_list = fc.toList(fc.size())
                    indices = ee.List.sequence(0, fc.size().subtract(1))

                    def set_id_negatives2(idx):
                        idx = ee.Number(idx)
                        feature = ee.Feature(features_list.get(idx))
                        labeled_feature = feature.set(
                            'id_property', ee.String('N').cat(idx.add(1).int().format())
                        )
                        return labeled_feature
                    
                    Neg_points_id = ee.FeatureCollection(indices.map(set_id_negatives2))
                    
                    Pos_collection = st.session_state.Positive_collection
                    Pos_collection = Pos_collection.map(lambda feature: feature.set('Dam', 'positive'))

                    pos_features_list = Pos_collection.toList(Pos_collection.size())
                    pos_indices = ee.List.sequence(0, Pos_collection.size().subtract(1))

                    def set_id_positives(idx):
                        idx = ee.Number(idx)
                        feature = ee.Feature(pos_features_list.get(idx))
                        labeled_feature = feature.set(
                            'id_property', ee.String('P').cat(idx.add(1).int().format())
                        )
                        return labeled_feature

                    Positive_dam_id = ee.FeatureCollection(pos_indices.map(set_id_positives))
                    Merged_collection = Positive_dam_id.merge(Neg_points_id)
                    st.session_state['Merged_collection'] = Merged_collection
                    st.session_state['buffer_complete'] = True

                    Negative_points = geemap.Map()
                    Negative_points.add_basemap("SATELLITE")
                    Negative_points.addLayer(negativePoints,{'color': 'red', 'width': 2},'Negative')
                    Negative_points.addLayer(Positive_dam_id,{'color': 'blue'},'Positive')
                    Negative_points.centerObject(Merged_collection)
                    Negative_points.to_streamlit(width=1200, height=700)

    # Show buffer section after non-dam section is complete
    if 'Merged_collection' in st.session_state and st.session_state['buffer_complete']:
        st.header("Merge and Buffer Dam and Non Dam locations:")

        # User inputs for Dam status and buffer radius
        buffer_radius = st.number_input(
            "Enter buffer radius (meters):", 
            min_value=1, 
            step=1, 
            value=st.session_state.buffer_radius,
            key="buffer_radius_input"
        )

        # Button to apply Dam status and create buffers
        if st.button("Apply Dam Status and Create Buffers"):
            # Apply the function to the feature collection

            first_pos = st.session_state.Positive_collection.first()
            date = ee.Date(first_pos.get('date'))
            year_string = date.format('YYYY')
            full_date = ee.String(year_string).cat('-07-01')


            def add_dam_buffer_and_standardize_date(feature):
                # Add Dam property and other metadata
                dam_status = feature.get("Dam")
                
                # Force the date to July 1st of the specified year
                standardized_date = date
                formatted_date = date.format('YYYYMMdd')
                
                # Buffer geometry while retaining properties
                buffered_geometry = feature.geometry().buffer(buffer_radius)
                
                # Create a new feature with buffered geometry and updated properties
                return ee.Feature(buffered_geometry).set({
                    "Dam": dam_status,
                    "Survey_Date": standardized_date,  # Set survey date to July 1st
                    "Damdate": ee.String("DamDate_").cat(formatted_date),  # Updated date format
                    "Point_geo": feature.geometry(),
                    "id_property": feature.get("id_property")
                })


            Buffered_collection = st.session_state['Merged_collection'].map(add_dam_buffer_and_standardize_date)
            

            # Select only relevant properties
            Dam_data = Buffered_collection.select(['id_property', 'Dam', 'Survey_Date', 'Damdate', 'Point_geo'])

            # Save to session state
            st.session_state['Dam_data'] = Dam_data
            st.session_state['buffers_created'] = True  # add new state variable to mark buffers created

            st.success(f"Buffers created with radius {buffer_radius} meters.")

    # Show Buffered Feature Collection and Visualization sections only after buffers are created
    if 'Dam_data' in st.session_state and st.session_state.get('buffers_created', False):
        st.write("Buffered Feature Collection:")
        dam_bounds = st.session_state['Dam_data'].geometry().bounds()
        states_dataset = ee.FeatureCollection("TIGER/2018/States")  # US States boundaries dataset
        states_with_dams = states_dataset.filterBounds(dam_bounds)
        st.session_state['Dam_state'] = states_with_dams
        Dam_data = st.session_state['Dam_data']
        Negative = Dam_data.filter(ee.Filter.eq('Dam', 'negative'))
        Positive = Dam_data.filter(ee.Filter.eq('Dam', 'positive'))

        Buffer_map = geemap.Map()
        Buffer_map.add_basemap("SATELLITE")
        Buffer_map.addLayer(Negative,{'color': 'red'},'Negative')
        Buffer_map.addLayer(Positive,{'color': 'blue'},'Positive')
        Buffer_map.centerObject(st.session_state['Dam_data'])
        Buffer_map.to_streamlit(width=800, height=600)

        # Add visualization button at the end
        st.header("Visualize trends")
        if st.button("Visualize Trends", key="visualize_trends_btn"):
            Initiate_visualize = 'visualize' 
            st.session_state['Visualize_trends'] = Initiate_visualize
            st.rerun()

# Move visualization section to the very end
if 'Visualize_trends' in st.session_state:
    with st.spinner("Processing... this may take some time."):
        # Filter Imagery
        Dam_data = st.session_state['Dam_data']

        S2_cloud_mask_export = ee.ImageCollection(S2_Export_for_visual(Dam_data))
        S2_ImageCollection = ee.ImageCollection(S2_cloud_mask_export)

        S2_with_LST = S2_ImageCollection.map(add_landsat_lst_et)
        results_fc_lst = S2_with_LST.map(compute_all_metrics_LST_ET)       

        results_fcc_lst = ee.FeatureCollection(results_fc_lst)       

        ## Practice
        df_lst = geemap.ee_to_df(results_fcc_lst)
        st.success("Dataframe with NDVI, NDWI, LST, and ET generated!")

        # 4) Convert columns to numeric
        df_lst['Image_month'] = pd.to_numeric(df_lst['Image_month'])
        df_lst['Image_year'] = pd.to_numeric(df_lst['Image_year'])
        df_lst['Dam_status'] = df_lst['Dam_status'].replace({'positive': 'Dam', 'negative': 'Non-dam'})
        df_lst['LST'] = pd.to_numeric(df_lst['LST'])
        df_lst['ET'] = pd.to_numeric(df_lst['ET'])  # <--- NEW: Ensure ET is numeric

        # 5) Set up your plotting style (using a white background)
        sns.set(style="whitegrid", palette="muted")
        fig = plt.figure(figsize=(12, 18), facecolor='white', edgecolor='white')

        # 6) Sort the DataFrame by year, then month
        df_lst = df_lst.sort_values(by=['Image_year', 'Image_month'])

        ############################################
        # 1) Plot NDVI
        ############################################
        ax1 = fig.add_subplot(4, 1, 1)
        sns.lineplot(
            data=df_lst, 
            x="Image_month", 
            y="NDVI", 
            hue="Dam_status", 
            style="Dam_status",
            markers=True, 
            dashes=False,
            ax=ax1
        )
        ax1.set_title('NDVI by Month for Dam and Non-Dam Sites', fontsize=14, color='black')
        ax1.set_xlabel('Month', fontsize=12, color='black')
        ax1.set_ylabel('Mean NDVI', fontsize=12, color='black')
        ax1.legend(title='Dam Status', loc='upper right')
        ax1.set_xticks(range(1, 13))
        ax1.tick_params(axis='x', colors='black')
        ax1.tick_params(axis='y', colors='black')

        ############################################
        # 2) Plot NDWI_Green
        ############################################
        ax2 = fig.add_subplot(4, 1, 2)
        sns.lineplot(
            data=df_lst, 
            x="Image_month", 
            y="NDWI_Green", 
            hue="Dam_status", 
            style="Dam_status",
            markers=True, 
            dashes=False,
            ax=ax2
        )
        ax2.set_title('NDWI Green by Month for Dam and Non-Dam Sites', fontsize=14, color='black')
        ax2.set_xlabel('Month', fontsize=12, color='black')
        ax2.set_ylabel('Mean NDWI Green', fontsize=12, color='black')
        ax2.legend(title='Dam Status', loc='upper right')
        ax2.set_xticks(range(1, 13))
        ax2.tick_params(axis='x', colors='black')
        ax2.tick_params(axis='y', colors='black')

        ############################################
        # 3) Plot LST
        ############################################
        ax3 = fig.add_subplot(4, 1, 3)
        sns.lineplot(
            data=df_lst,
            x="Image_month",
            y="LST",
            hue="Dam_status",
            style="Dam_status",
            markers=True,
            dashes=False,
            ax=ax3
        )
        ax3.set_title('LST by Month for Dam and Non-Dam Sites', fontsize=14, color='black')
        ax3.set_xlabel('Month', fontsize=12, color='black')
        ax3.set_ylabel('Mean LST (°C)', fontsize=12, color='black')
        ax3.legend(title='Dam Status', loc='upper right')
        ax3.set_xticks(range(1, 13))
        ax3.tick_params(axis='x', colors='black')
        ax3.tick_params(axis='y', colors='black')

        ############################################
        # 4) Plot ET
        ############################################
        ax4 = fig.add_subplot(4, 1, 4)
        sns.lineplot(
            data=df_lst,
            x="Image_month",
            y="ET",
            hue="Dam_status",
            style="Dam_status",
            markers=True,
            dashes=False,
            ax=ax4
        )
        ax4.set_title('ET by Month for Dam and Non-Dam Sites', fontsize=14, color='black')
        ax4.set_xlabel('Month', fontsize=12, color='black')
        ax4.set_ylabel('Mean ET', fontsize=12, color='black')
        ax4.legend(title='Dam Status', loc='upper right')
        ax4.set_xticks(range(1, 13))
        ax4.tick_params(axis='x', colors='black')
        ax4.tick_params(axis='y', colors='black')

        # 7) Final layout adjustments and display
        fig.tight_layout()
        st.pyplot(fig)

        # 8) Provide a download button to export the figure as a PNG image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        buf.seek(0)
        st.download_button(
            label="Download Figure",
            data=buf,
            file_name="trends_figure.png",
            mime="image/png"
        )