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

# Initialize session state for questionnaire
if 'questionnaire_shown' not in st.session_state:
    st.session_state.questionnaire_shown = False
if 'survey_clicked' not in st.session_state:
    st.session_state.survey_clicked = False

# Show questionnaire toast if not shown before
if not st.session_state.questionnaire_shown:
    st.toast(
        "Please help us improve by completing our survey!",
        icon="📝"
    )
    
    st.title("Beaver Impacts Feedback Survey")
    st.markdown("""
    Thank you for being a beta tester for the Beaver Impacts web tool! We really value your input and appreciate you taking the time to fill out this form.
    
    Please click [here](https://docs.google.com/forms/d/e/1FAIpQLSeE1GP7OptA4-z8Melz2AHxNsddtL9ZgJVXdVVtxLsrljJ10Q/viewform?usp=sharing) to start the survey.
    """)
    
    if st.button("I have started the survey and will fill it out during analysis.", type="primary"):
        st.session_state.questionnaire_shown = True
        st.rerun()

# Continue with the rest of the application if questionnaire is shown
if st.session_state.questionnaire_shown:
    # Initialize session state variables
    if "Positive_collection" not in st.session_state:
        st.session_state.Positive_collection = None
    if "buffer_radius" not in st.session_state:
        st.session_state.buffer_radius = 200  # Default buffer radius
    if "validation_complete" not in st.session_state:
        st.session_state.validation_complete = False
    if "use_all_dams" not in st.session_state:
        st.session_state.use_all_dams = True
    if "validation_step" not in st.session_state:
        st.session_state.validation_step = 'initial'
    if "show_non_dam_section" not in st.session_state:
        st.session_state.show_non_dam_section = False
    if "buffer_complete" not in st.session_state:
        st.session_state.buffer_complete = False
    if "Dam_data" not in st.session_state:
        st.session_state.Dam_data = None
    if "Full_positive" not in st.session_state:
        st.session_state.Full_positive = None
    if "selected_waterway" not in st.session_state:
        st.session_state.selected_waterway = None
    if "dataset_loaded" not in st.session_state:
        st.session_state.dataset_loaded = False
    if "validation_results" not in st.session_state:
        st.session_state.validation_results = None
    if "buffers_created" not in st.session_state:
        st.session_state.buffers_created = False
    if "Merged_collection" not in st.session_state:
        st.session_state.Merged_collection = None
    if "visualization_complete" not in st.session_state:
        st.session_state.visualization_complete = False
    if "df_lst" not in st.session_state:
        st.session_state.df_lst = None
    if "fig" not in st.session_state:
        st.session_state.fig = None

    # Streamlit UI
    st.title("Analyzing the Impact of Beaver Dams")

    # Initialize session state for step control
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 1
    if 'total_steps' not in st.session_state:
        st.session_state['total_steps'] = 6  # Total number of steps

    # Display content based on current step
    if st.session_state['current_step'] == 1:
        st.warning("Please note that the Evapotranspiration data is not available for beaver dam locations in the east half of US. See which states are not available on OpenET website: [Link](https://explore.etdata.org/#5/39.665/-110.396).")
        st.header("Step 1: Upload Dam Locations")
        uploaded_file = st.file_uploader("Choose a CSV or GeoJSON file", type=["csv", "geojson"], key="Dam_file_uploader")
        if uploaded_file:
            with st.spinner("Processing uploaded file..."):
                try:
                    feature_collection = upload_points_to_ee(uploaded_file, widget_prefix="Dam")
                    if feature_collection:
                        st.session_state.Positive_collection = feature_collection
                        st.session_state.Full_positive = feature_collection
                        # Display data preview
                        st.subheader("Data Preview")
                        preview_map = geemap.Map()
                        preview_map.add_basemap("SATELLITE")
                        preview_map.addLayer(feature_collection, {'color': 'blue'}, 'Dam Locations')
                        preview_map.centerObject(feature_collection)
                        preview_map.to_streamlit(width=800, height=600)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    elif st.session_state['current_step'] == 2:
        st.header("Step 2: Select Waterway")
        if "selected_waterway" not in st.session_state:
            st.session_state.selected_waterway = None
        if "dataset_loaded" not in st.session_state:
            st.session_state.dataset_loaded = False

        if 'Full_positive' in st.session_state:
            # Show loading message
            st.info("Automatically loading NHD dataset and preparing map...")
            
            try:
                # Get dam bounds
                positive_dam_bounds = st.session_state['Full_positive'].geometry().bounds()
                states_dataset = ee.FeatureCollection("TIGER/2018/States")
                states_with_dams = states_dataset.filterBounds(positive_dam_bounds)
                st.session_state['Positive_dam_state'] = states_with_dams
                states_geo = st.session_state['Positive_dam_state']
                state_names = states_geo.aggregate_array("NAME").getInfo()

                if not state_names:
                    st.error("No states found within the dam data bounds.")
                else:
                    st.write(f"States within dam data bounds: {state_names}")

                    # Automatically load NHD dataset
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

                    if nhd_collections:
                        merged_nhd = ee.FeatureCollection(nhd_collections).flatten()
                        st.session_state.selected_waterway = merged_nhd
                        st.session_state['Waterway'] = st.session_state.selected_waterway
                        st.session_state.dataset_loaded = True
                        
                        # Display map
                        Waterway_map = geemap.Map()
                        Waterway_map.add_basemap("SATELLITE")
                        Waterway_map.centerObject(st.session_state['Full_positive'])
                        # Add waterway layer first
                        Waterway_map.addLayer(st.session_state.selected_waterway, {"color": "blue"}, "Selected Waterway")
                        # Then add dam points layer
                        Waterway_map.addLayer(st.session_state['Full_positive'],{'color': 'red'},'Dams')
                        Waterway_map.to_streamlit(width=1200, height=700)
                                                
                        # Provide additional options
                        st.subheader("Additional Options")
                        upload_own_checkbox = st.checkbox("Use Custom Dataset")
                        choose_other_checkbox = st.checkbox("Use Alternative Dataset")

                        if upload_own_checkbox:
                            asset_id = st.text_input("Enter GEE Asset Table ID (e.g., projects/ee-beaver-lab/assets/Hydro/MA_Hydro_arc):")
                            if st.button("Load Custom Dataset"):
                                try:
                                    waterway_own = ee.FeatureCollection(asset_id)
                                    st.session_state.selected_waterway = waterway_own
                                    st.session_state.dataset_loaded = True
                                    st.success("Custom dataset successfully loaded.")
                                except Exception as e:
                                    st.error(f"Failed to load dataset: {e}")

                        if choose_other_checkbox:
                            dataset_option = st.selectbox(
                                "Select alternative dataset:",
                                ["WWF Free Flowing Rivers"]
                            )

                            if st.button("Load Alternative Dataset"):
                                try:
                                    if dataset_option == "WWF Free Flowing Rivers":
                                        wwf_dataset = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
                                        clipped_wwf = wwf_dataset.filterBounds(states_with_dams)
                                        st.session_state.selected_waterway = clipped_wwf
                                        st.session_state.dataset_loaded = True
                                        st.success("WWF dataset successfully loaded.")
                                except Exception as e:
                                    st.error(f"Failed to load dataset: {e}")
                    else:
                        st.error("No NHD datasets found for the selected states.")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    elif st.session_state['current_step'] == 3:
        st.header("Step 3: Validate Dam Locations")
        
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
            # Add validation parameters
            max_distance = st.number_input(
                "Maximum allowed distance from waterway (meters):",
                min_value=0,
                value=100,
                step=10,
                key="max_distance_input"
            )
            
            # Validate positive dams
            if 'Full_positive' in st.session_state and 'Waterway' in st.session_state:
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
                                
                                # Add debug information
                                # st.write("Debug: Validation Results")
                                # st.write(f"Total dams: {st.session_state['Full_positive'].size().getInfo()}")
                                # st.write(f"Distance valid dams: {distance_validation['valid_count'].getInfo()}")
                                # st.write(f"Distance invalid dams: {distance_validation['invalid_count'].getInfo()}")
                                # st.write(f"Intersecting dams: {intersection_validation['intersecting_count'].getInfo()}")
                                # st.write(f"Using max_distance: {max_distance} meters")
                                
                                # Combine validation results
                                validation_results = {
                                    'valid_dams': distance_validation['valid_dams'],
                                    'invalid_dams': distance_validation['invalid_dams'],
                                    'invalid_dams_info': distance_validation['invalid_dams_info'],
                                    'total_dams': st.session_state['Full_positive'].size(),
                                    'valid_count': distance_validation['valid_count'],
                                    'invalid_count': distance_validation['invalid_count']
                                }
                                
                                # Store validation results in session state
                                st.session_state['validation_results'] = validation_results
                                st.session_state['validation_step'] = 'show_options'
                                
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
                                
                        except Exception as e:
                            st.error(f"Error during validation: {str(e)}")
                            st.write("Debug information:")
                            st.write(f"Data size: {st.session_state['Full_positive'].size().getInfo()}")
                            st.write(f"Data type: {type(st.session_state['Full_positive'])}")
                            st.write(f"Waterway data size: {st.session_state['Waterway'].size().getInfo()}")
                            st.write(f"Waterway data type: {type(st.session_state['Waterway'])}")
            
            # Show options after validation is complete
            if st.session_state['validation_step'] == 'show_options':
                if 'validation_results' in st.session_state:
                    validation_results = st.session_state['validation_results']
                    valid_count = validation_results['valid_count'].getInfo()
                    invalid_count = validation_results['invalid_count'].getInfo()
                    
                    if valid_count == 0:
                        st.error("No valid dam locations found. All dams failed validation.")
                        st.write("Please check your dam locations and waterway data.")
                        st.stop()
                    elif invalid_count > 0:
                        st.warning("Some dam locations have been identified as potentially invalid. Please review the validation report and map above. You can continue with all dams or only use the valid dams.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Continue with all dams", key="use_all_dams_btn"):
                                st.session_state['validation_complete'] = True
                                st.session_state['use_all_dams'] = True
                                st.session_state['Dam_data'] = st.session_state['Full_positive']
                                st.session_state['show_non_dam_section'] = True
                                st.success("Selected to continue with all dams. Click 'Next' to proceed.")
                                st.session_state['current_step'] += 1
                        with col2:
                            if st.button("Only use valid dams"):
                                valid_dams = validation_results['valid_dams']
                                valid_count = validation_results['valid_count'].getInfo()
                                
                                if valid_count > 0:
                                    st.session_state['Full_positive'] = valid_dams
                                    st.session_state['validation_step'] = 'completed'
                                    st.session_state['validation_complete'] = True
                                    st.session_state['use_all_dams'] = False
                                    st.session_state['Dam_data'] = valid_dams
                                    st.session_state['show_non_dam_section'] = True
                                    st.success(f"Successfully filtered to {valid_count} valid dams. Click 'Next' to proceed.")
                                else:
                                    st.warning("No valid dams found. Please adjust the validation criteria.")
                    else:
                        st.success("All dam locations are valid!")
                        st.session_state['validation_complete'] = True
                        st.session_state['use_all_dams'] = True
                        st.session_state['Dam_data'] = st.session_state['Full_positive']
                        st.session_state['show_non_dam_section'] = True
                        st.success("All dams are valid. Click 'Next' to proceed.")
                else:
                    st.error("No validation results found. Please run validation first.")

    elif st.session_state['current_step'] == 4:
        st.header("Step 4: Upload or Generate Non-Dam Locations")
        
        # Check required states
        if not st.session_state.get('validation_complete', False):
            st.error("Please complete the validation step first.")
            st.session_state['current_step'] = 3
            st.rerun()
        
        if not st.session_state.get('show_non_dam_section', False):
            st.error("Please complete the validation step first.")
            st.session_state['current_step'] = 3
            st.rerun()
        
        if st.session_state['use_all_dams']:
            st.info("Using all dam locations for analysis")
        else:
            st.info("Using only valid dam locations for analysis")
        
        upload_negatives_checkbox = st.checkbox("Upload Non-Dam Dataset (must be on a waterbody)")
        generate_negatives_checkbox = st.checkbox("Generate Non-Dam Locations")

        if upload_negatives_checkbox:
            uploaded_negatives_file = st.file_uploader("Choose a CSV or GeoJSON file", type=["csv", "geojson"], key="Non_Dam_file_uploader")
            if uploaded_negatives_file:
                with st.spinner("Processing uploaded file..."):
                    try:
                        negative_feature_collection = upload_points_to_ee(uploaded_negatives_file, widget_prefix="NonDam")
                        if negative_feature_collection:
                            st.session_state.Negative_upload_collection = negative_feature_collection
                            st.session_state['Full_negative'] = st.session_state.Negative_upload_collection
                            st.success("Non-dam locations uploaded successfully!")
                            # Display data preview
                            st.subheader("Data Preview")
                            preview_map = geemap.Map()
                            preview_map.add_basemap("SATELLITE")
                            preview_map.addLayer(negative_feature_collection, {'color': 'red'}, 'Non-Dam Locations')
                            preview_map.centerObject(negative_feature_collection)
                            preview_map.to_streamlit(width=800, height=600)
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")

        if generate_negatives_checkbox:
            st.subheader("Specify the parameters for negative point generation:")

            innerRadius = st.number_input("Inner Radius (meters)", value=200, min_value=0, step=50, key="inner_radius_input")
            outerRadius = st.number_input("Outer Radius (meters)", value=350, min_value=0, step=100, key="outer_radius_input")
            # samplingScale = st.number_input("Sampling Scale (meters)", value=10, min_value=1, step=1, key="sampling_scale_input")
            samplingScale = 10
            
            if st.button("Generate Negative Points"):
                with st.spinner("Generating negative points..."):
                    try:
                        # Get the first positive dam for date
                        if not st.session_state['use_all_dams']:
                            positive_dams_fc = st.session_state['Dam_data']
                        else:
                            positive_dams_fc = st.session_state['Positive_collection']
                        
                        # Check if we have valid data
                        if positive_dams_fc.size().getInfo() == 0:
                            st.error("No valid dam data found. Please check your data and try again.")
                            st.stop()
                            
                        # Get the bounds of positive dams
                        positive_bounds = positive_dams_fc.geometry().bounds()
                        
                        # Check if bounds are valid with error margin
                        bounds_area = positive_bounds.area(1).getInfo()
                        
                        if bounds_area == 0:
                            st.error("No valid dam locations found. Please check your data.")
                            st.stop()
                            
                        # Clip waterway to positive dams bounds
                        waterway_fc = st.session_state.selected_waterway.filterBounds(positive_bounds)
                        
                        # Check if waterway data is available
                        if waterway_fc.size().getInfo() == 0:
                            st.error("No waterway data found within the dam locations area. Please check your waterway selection.")
                            st.stop()
                            
                        hydroRaster = prepareHydro(waterway_fc)
                    
                        # Generate negative points
                        negativePoints = sampleNegativePoints(
                            positive_dams_fc, 
                            hydroRaster, 
                            innerRadius, 
                            outerRadius, 
                            samplingScale
                        )
                        
                        # Check if negative points were generated
                        if negativePoints.size().getInfo() == 0:
                            st.error("No negative points were generated. Please try adjusting the parameters.")
                            st.stop()
                            
                        # Get date from first positive dam
                        first_pos = positive_dams_fc.first()
                        date = ee.Date(first_pos.get('date'))
                        year_string = date.format('YYYY')
                        full_date = ee.String(year_string).cat('-07-01')
                        
                        negativePoints = negativePoints.map(lambda feature: feature.set('Dam', 'negative').set("date", full_date))
                        
                        # Process negative points
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
                        
                        # Use filtered positive dams if only valid dams are selected
                        if not st.session_state['use_all_dams']:
                            Pos_collection = st.session_state['Dam_data']
                        else:
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

                        # Create and display the map
                        Negative_points = geemap.Map()
                        Negative_points.add_basemap("SATELLITE")
                        Negative_points.addLayer(negativePoints,{'color': 'red', 'width': 2},'Negative')
                        Negative_points.addLayer(Positive_dam_id,{'color': 'blue'},'Positive')
                        Negative_points.centerObject(Merged_collection)
                        Negative_points.to_streamlit(width=1200, height=700)
                
                        # Set completion status
                        st.session_state['step4_complete'] = True
                        st.success("Negative points generated successfully! Click 'Next' to proceed to buffering.")
                    except Exception as e:
                        st.error(f"Error generating negative points: {str(e)}")

    elif st.session_state['current_step'] == 5:
        st.header("Step 5: Create Buffers")
        
        # Check if step 4 is completed
        if not st.session_state.get('step4_complete', False):
            st.error("Please complete Step 4 first.")
            st.session_state['current_step'] = 4
            st.rerun()
        
        # Check if merged collection exists
        if 'Merged_collection' not in st.session_state:
            st.error("No merged data found. Please complete Step 4 first.")
            st.session_state['current_step'] = 4
            st.rerun()
        
        # Display buffer settings
        st.subheader("Buffer Settings")
        buffer_radius = st.number_input(
            "Enter buffer radius (meters):", 
            min_value=1, 
            step=1, 
            value=st.session_state.buffer_radius,
            key="buffer_radius_input"
        )

        if st.button("Create Buffers"):
            with st.spinner("Creating buffers..."):
                try:
                    # Get date from first positive dam
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
                            "Survey_Date": standardized_date,
                            "Damdate": ee.String("DamDate_").cat(formatted_date),
                            "Point_geo": feature.geometry(),
                            "id_property": feature.get("id_property")
                        })

                    # Create buffers
                    Buffered_collection = st.session_state['Merged_collection'].map(add_dam_buffer_and_standardize_date)
                    
                    # Select relevant properties
                    Dam_data = Buffered_collection.select(['id_property', 'Dam', 'Survey_Date', 'Damdate', 'Point_geo'])

                    # Save to session state
                    st.session_state['Dam_data'] = Dam_data
                    st.session_state['buffers_created'] = True
                    
                    # Split into positive and negative points
                    Negative = Dam_data.filter(ee.Filter.eq('Dam', 'negative'))
                    Positive = Dam_data.filter(ee.Filter.eq('Dam', 'positive'))

                    # Display buffer preview
                    st.subheader("Buffer Preview")
                    Buffer_map = geemap.Map()
                    Buffer_map.add_basemap("SATELLITE")
                    Buffer_map.addLayer(Negative, {'color': 'red'}, 'Negative')
                    Buffer_map.addLayer(Positive, {'color': 'blue'}, 'Positive')
                    Buffer_map.centerObject(Dam_data)
                    Buffer_map.to_streamlit(width=800, height=600)
                    
                    # Set completion status
                    st.session_state['step5_complete'] = True
                    st.success(f"Buffers created successfully with radius {buffer_radius} meters! Click 'Next' to proceed to visualization.")
                    
                except Exception as e:
                    st.error(f"Error creating buffers: {str(e)}")

    elif st.session_state['current_step'] == 6:
        st.header("Step 6: Visualize Trends")
        
        # Check if step 5 is completed
        if not st.session_state.get('step5_complete', False):
            st.error("Please complete Step 5 first.")
            st.session_state['current_step'] = 5
            st.rerun()
        
        if not st.session_state.visualization_complete:
            if st.button("Generate Visualization"):
                with st.spinner("Processing visualization... This may take some time."):
                    try:
                        # Filter Imagery
                        Dam_data = st.session_state['Dam_data']

                        S2_cloud_mask_export = ee.ImageCollection(S2_Export_for_visual(Dam_data))
                        S2_ImageCollection = ee.ImageCollection(S2_cloud_mask_export)

                        S2_with_LST = S2_ImageCollection.map(add_landsat_lst_et)
                        results_fc_lst = S2_with_LST.map(compute_all_metrics_LST_ET)       

                        results_fcc_lst = ee.FeatureCollection(results_fc_lst)       

                        # Create DataFrame
                        df_lst = geemap.ee_to_df(results_fcc_lst)
                        st.success("Dataframe with NDVI, NDWI, LST, and ET generated!")

                        # Convert columns to numeric
                        df_lst['Image_month'] = pd.to_numeric(df_lst['Image_month'])
                        df_lst['Image_year'] = pd.to_numeric(df_lst['Image_year'])
                        df_lst['Dam_status'] = df_lst['Dam_status'].replace({'positive': 'Dam', 'negative': 'Non-dam'})
                        df_lst['LST'] = pd.to_numeric(df_lst['LST'])
                        df_lst['ET'] = pd.to_numeric(df_lst['ET'])

                        # Set up plotting style
                        sns.set(style="whitegrid", palette="muted")
                        fig = plt.figure(figsize=(12, 18), facecolor='white', edgecolor='white')

                        # Sort DataFrame
                        df_lst = df_lst.sort_values(by=['Image_year', 'Image_month'])

                        # Plot NDVI
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

                        # Plot NDWI_Green
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

                        # Plot LST
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

                        # Plot ET
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

                        # Final layout adjustments
                        fig.tight_layout()
                        
                        # Save to session state
                        st.session_state.df_lst = df_lst
                        st.session_state.fig = fig
                        st.session_state.visualization_complete = True
                        
                        # Display the figure
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error during visualization: {str(e)}")
        
        if st.session_state.visualization_complete:
            # Create download section
            st.subheader("Download Results")
            
            # Create two columns for download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download figure button
                buf = io.BytesIO()
                st.session_state.fig.savefig(buf, format="png", facecolor=st.session_state.fig.get_facecolor())
                buf.seek(0)
                st.download_button(
                    label="Download Visualization",
                    data=buf,
                    file_name="trends_figure.png",
                    mime="image/png"
                )
            
            with col2:
                # Data format selection
                data_format = st.selectbox(
                    "Select data format:",
                    ["CSV", "JSON", "TXT", "Excel (xlsx)", "Markdown (md)"]
                )
                
                if data_format == "CSV":
                    csv_buf = io.StringIO()
                    st.session_state.df_lst.to_csv(csv_buf, index=False)
                    csv_buf.seek(0)
                    st.download_button(
                        label="Download Data (CSV)",
                        data=csv_buf.getvalue(),
                        file_name="trends_data.csv",
                        mime="text/csv"
                    )
                
                elif data_format == "JSON":
                    json_buf = io.StringIO()
                    st.session_state.df_lst.to_json(json_buf, orient='records', indent=2)
                    json_buf.seek(0)
                    st.download_button(
                        label="Download Data (JSON)",
                        data=json_buf.getvalue(),
                        file_name="trends_data.json",
                        mime="application/json"
                    )
                
                elif data_format == "TXT":
                    txt_buf = io.StringIO()
                    st.session_state.df_lst.to_string(txt_buf, index=False)
                    txt_buf.seek(0)
                    st.download_button(
                        label="Download Data (TXT)",
                        data=txt_buf.getvalue(),
                        file_name="trends_data.txt",
                        mime="text/plain"
                    )
                
                elif data_format == "Excel (xlsx)":
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
                        st.session_state.df_lst.to_excel(writer, index=False, sheet_name='Trends Data')
                    excel_buf.seek(0)
                    st.download_button(
                        label="Download Data (Excel)",
                        data=excel_buf.getvalue(),
                        file_name="trends_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif data_format == "Markdown (md)":
                    md_buf = io.StringIO()
                    md_buf.write("# Trends Data\n\n")
                    md_buf.write(st.session_state.df_lst.to_markdown(index=False))
                    md_buf.seek(0)
                    st.download_button(
                        label="Download Data (Markdown)",
                        data=md_buf.getvalue(),
                        file_name="trends_data.md",
                        mime="text/markdown"
                    )
            
            st.success("Visualization completed! You can download the visualization or data in your preferred format.")

    # Add navigation buttons at the bottom of each step
    st.markdown("---")  # Add a horizontal line to separate content from navigation

    # Create a centered container for navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.session_state['current_step'] > 1:
            if st.button("Previous"):
                st.session_state['current_step'] -= 1
                st.rerun()

    with col2:
        st.markdown(f"**Step {st.session_state['current_step']}/{st.session_state['total_steps']}**")

    with col3:
        if st.session_state['current_step'] < st.session_state['total_steps']:
            if st.button("Next"):
                st.session_state['current_step'] += 1
                st.rerun()