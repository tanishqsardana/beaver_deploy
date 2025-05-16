# Beaver Impacts Tool: Developer Guide (Tentative)

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Code Structure](#code-structure)
3. [Key Components](#key-components)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Earth Engine Integration](#earth-engine-integration)
6. [Visualization Components](#visualization-components)
7. [Adding New Features](#adding-new-features)
8. [Common Issues and Debugging](#common-issues-and-debugging)

## Architecture Overview

The Beaver Impacts Tool is built on:
- **Streamlit**: For the web interface
- **Google Earth Engine (GEE)**: For satellite imagery processing
- **Pandas/NumPy**: For data manipulation
- **Seaborn/Matplotlib**: For visualization

The application follows a step-by-step workflow where users:
1. Upload dam locations
2. Select waterway datasets
3. Validate dam locations
4. Generate or upload non-dam locations
5. Create buffered analysis zones
6. Analyze and visualize environmental metrics

Each step involves interactions between the frontend (Streamlit) and backend processing using Earth Engine's Python API.

## Code Structure

```
beaver-impacts-tool/
├── pages/                      # Streamlit pages
│   ├── Exports_page.py         # Main analysis workflow
│   └── [other pages...]        # Additional functionality
├── service/                    # Core business logic
│   ├── Sentinel2_functions.py  # Sentinel-2 image processing
│   ├── Export_dam_imagery.py   # Image export functionality
│   ├── Visualize_trends.py     # Visualization and metrics computation
│   ├── Negative_sample_functions.py # Non-dam point generation
│   ├── Parser.py               # Data parsing and input handling
│   ├── Data_management.py      # Data management utilities
│   └── Validation_service.py   # Validation logic
├── assets/                     # Static assets
├── app.py                      # Main application entry point
├── README.md                   # This documentation
└── requirements.txt            # Dependencies
```

## Key Components

### Earth Engine Authentication
```python
credentials_info = {
    "type": st.secrets["gcp_service_account"]["type"],
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    # Other credentials
}
credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/earthengine"]
)
ee.Initialize(credentials, project="ee-beaver-lab")
```
This establishes the connection to Earth Engine using service account credentials.

### Session State Management
```python
# Initialize session state variables
if "Positive_collection" not in st.session_state:
    st.session_state.Positive_collection = None
# More state variables...
```
The application uses Streamlit's session state to maintain state between user interactions.

### Multi-step Workflow
Each analysis step is implemented as an expandable section:
```python
with st.expander("Step 1: Upload Dam Locations", expanded=not st.session_state.step1_complete):
    # Step 1 implementation
```

## Data Processing Pipeline

The application implements a complex data processing pipeline that transforms user inputs into actionable insights. The following six steps accurately reflect the actual code implementation:

### 1. Point Data Processing and Standardization

**Input**: CSV/GeoJSON files containing dam/non-dam locations
**Processing**: 
```python
# Upload and standardize dam points
feature_collection = upload_points_to_ee(uploaded_file, widget_prefix="Dam")
feature_collection = feature_collection.map(set_id_year_property)

# For non-dam points
negative_points = sampleNegativePoints(positive_dams_fc, hydroRaster, innerRadius, outerRadius, samplingScale)
negative_points = negative_points.map(set_id_negatives)
```
This function:
1. Validates spatial data (coordinates)
2. Standardizes date formats
3. Assigns unique identifiers (P1, P2... for dams; N1, N2... for non-dams)
4. Sets properties like dam status (positive/negative)

**Output**: Earth Engine FeatureCollection with standardized points

### 2. Buffer Creation and Elevation Masking

**Input**: Standardized FeatureCollection of points
**Processing**:
```python
Buffered_collection = Merged_collection.map(add_dam_buffer_and_standardize_date)
```
This function:
1. Creates circular buffers (default: 150m radius)
2. Applies elevation masking (±3m from point elevation)
3. Preserves original point geometry as a property
4. Sets date-related properties for time series analysis

**Output**: FeatureCollection with polygon geometries (buffers) constrained by elevation

### 3. Satellite Image Acquisition and Filtering

**Input**: Buffered FeatureCollection
**Processing**:
```python
# For combined analysis
S2_cloud_mask_batch = ee.ImageCollection(S2_Export_for_visual(dam_batch_fc))

# For upstream/downstream analysis
S2_IC_batch = S2_Export_for_visual_flowdir(dam_batch_fc, waterway_fc)
```
This function:
1. Determines time window (±6 months from survey date)
2. Applies spatial filter (using buffer geometries)
3. Applies cloud masking using QA bands
4. Selects least cloudy image for each month (cloud coverage < 20%)
5. Standardizes band names and properties

**Output**: Earth Engine ImageCollection with filtered monthly Sentinel-2 imagery

### 4. Advanced Metric Computation (LST and ET)

**Input**: Sentinel-2 ImageCollection
**Processing**:
```python
S2_with_LST_batch = S2_ImageCollection_batch.map(add_landsat_lst_et)
```
This function:
1. Acquires synchronous Landsat 8 thermal data for each Sentinel-2 image
2. Applies radiometric calibration to thermal bands
3. Calculates Land Surface Temperature (LST) using NDVI-based emissivity
4. Retrieves monthly evapotranspiration (ET) data from OpenET
5. Handles edge cases using median values when multiple images exist
6. Provides fallback values (99) when data is unavailable

**Output**: Enhanced ImageCollection with LST and ET bands added

### 5. Environmental Metrics Calculation

**Input**: Enhanced ImageCollection with LST and ET
**Processing**:
```python
# For combined analysis
results_fc_lst_batch = S2_with_LST_batch.map(compute_all_metrics_LST_ET)

# For upstream/downstream analysis
results_batch = S2_with_LST_ET.map(compute_all_metrics_up_downstream)
```
This function calculates:
1. NDVI (Normalized Difference Vegetation Index): (NIR-Red)/(NIR+Red)
2. NDWI (Normalized Difference Water Index): (Green-NIR)/(Green+NIR)
3. LST statistics (mean temperature in buffer area)
4. ET statistics (mean evapotranspiration in buffer area)
5. For upstream/downstream: calculates separate metrics for areas above and below dam points

**Output**: FeatureCollection with calculated environmental metrics

### 6. Data Processing and Visualization

**Input**: FeatureCollection with calculated metrics
**Processing**:
```python
# Convert to DataFrame
df_batch = geemap.ee_to_df(results_fcc_lst_batch)
df_list.append(df_batch)
df_lst = pd.concat(df_list, ignore_index=True)

# Data preparation
df_lst['Image_month'] = pd.to_numeric(df_lst['Image_month'])
df_lst['Image_year'] = pd.to_numeric(df_lst['Image_year'])
df_lst['Dam_status'] = df_lst['Dam_status'].replace({'positive': 'Dam', 'negative': 'Non-dam'})

# Visualization
fig, axes = plt.subplots(4, 1, figsize=(12, 18))
for ax, metric, title in zip(axes, metrics, titles):
    sns.lineplot(data=df_lst, x="Image_month", y=metric, hue="Dam_status", style="Dam_status",
                markers=True, dashes=False, ax=ax)
```
This function:
1. Converts Earth Engine data to DataFrame format
2. Standardizes data types (numeric months, years)
3. Applies proper labeling for visualization
4. Creates time series plots with confidence intervals (95% by default)
5. Computes statistical significance between dam and non-dam areas
6. Generates exportable visualizations and data tables

**Output**: Interactive visualizations and downloadable CSV data

## Earth Engine Integration

The application extensively uses Google Earth Engine for geospatial analysis. Key integration points include:

### Batch Processing

One of the most critical patterns is batch processing to manage memory:

```python
total_count = Dam_data.size().getInfo()
batch_size = 10
num_batches = (total_count + batch_size - 1) // batch_size

for i in range(num_batches):
    # Get current batch
    dam_batch = Dam_data.toList(batch_size, i * batch_size)
    dam_batch_fc = ee.FeatureCollection(dam_batch)
    
    # Process batch
    # ...
```

This pattern:
1. Divides large collections into manageable batches
2. Processes each batch independently
3. Combines results after processing

### LST Calculation

The Land Surface Temperature calculation demonstrates complex Earth Engine operations:

```python
def robust_compute_lst(filtered_col, boxArea):
    # Compute NDVI
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    
    # Calculate vegetation fraction
    fv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(2).rename('FV')
    
    # Calculate emissivity
    em = fv.multiply(0.004).add(0.986).rename('EM')
    
    # Apply radiative transfer equation
    lst = thermal.expression(
        '(TB / (1 + (0.00115 * (TB / 1.438)) * log(em))) - 273.15',
        {'TB': thermal, 'em': em}
    ).rename('LST')
    
    return lst
```

### Cloud Masking

Cloud masking is essential for reliable analysis:

```python
def cloud_mask(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(
           qa.bitwiseAnd(1 << 5).eq(0))
    return image.updateMask(mask)
```

## Visualization Components

The application creates several visualization types:

### Time Series Plots

```python
fig, axes = plt.subplots(4, 1, figsize=(12, 18))
metrics = ['NDVI', 'NDWI_Green', 'LST', 'ET']
titles = ['NDVI', 'NDWI Green', 'LST (°C)', 'ET']

for ax, metric, title in zip(axes, metrics, titles):
    sns.lineplot(data=df_lst, x="Image_month", y=metric, hue="Dam_status", 
                style="Dam_status", markers=True, dashes=False, ax=ax)
    ax.set_title(f'{title} by Month', fontsize=14)
    ax.set_xticks(range(1, 13))
```

### Upstream vs. Downstream Analysis

```python
def melt_and_plot(df, metric, ax):
    melted = df.melt(['Image_year','Image_month','Dam_status'], 
                  [f"{metric}_up", f"{metric}_down"], 
                  'Flow', metric)
    melted['Flow'].replace({f"{metric}_up":'Upstream', 
                         f"{metric}_down":'Downstream'}, 
                        inplace=True)
    sns.lineplot(data=melted, x='Image_month', y=metric, 
              hue='Dam_status', style='Flow', 
              markers=True, ax=ax)
```

## Adding New Features

To add new features to the application:

1. **Add new Earth Engine functions**:
   - Create functions in the appropriate service module
   - Ensure proper error handling
   - Test processing on small datasets first

2. **Add new UI components**:
   - Add new sections to the appropriate Streamlit page
   - Use `st.session_state` to maintain state
   - Follow the step pattern of existing code

3. **Add new metrics**:
   - Modify the `compute_all_metrics_LST_ET` function
   - Add processing code for the new metric
   - Update visualization code to include the new metric

## Common Issues and Debugging

### Memory Management

The most common issue is memory limits in Earth Engine:

```python
# Use batch processing
total_count = Dam_data.size().getInfo()
batch_size = 10  # Adjust this value based on data complexity
num_batches = (total_count + batch_size - 1) // batch_size

for i in range(num_batches):
    # Process in batches
    dam_batch = Dam_data.toList(batch_size, i * batch_size)
    # ...
```

### Error Handling

Always implement proper error handling:

```python
try:
    # Process data
    # ...
except Exception as e:
    st.warning(f"Error processing batch {i+1}: {e}")
    # Continue with next batch
    continue
```

### Dealing with Cloud Coverage

Use cloud masking and select least cloudy images:

```python
def get_monthly_least_cloudy_images(Collection):
    months = ee.List.sequence(1, 12)
    def get_month_image(month):
        monthly_images = Collection.filter(
            ee.Filter.calendarRange(month, month, 'month'))
        return ee.Image(monthly_images.sort('Cloud_coverage').first())
    
    monthly_images_list = months.map(get_month_image)
    return ee.ImageCollection.fromImages(monthly_images_list)
```


Happy coding!
