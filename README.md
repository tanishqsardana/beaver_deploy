# Beaver Impacts Tool: A Remote Sensing Framework for Quantifying Ecosystem Effects of Beaver Dam Activity

## Overview

The Beaver Impacts Tool is a sophisticated geospatial analysis framework developed to quantify the ecological and hydrological impacts of beaver dams and beaver-based restoration interventions. This web-based tool integrates multiple remote sensing data sources with advanced spatial analysis algorithms to provide land managers, researchers, and conservation practitioners with robust metrics for assessing beaver-induced landscape transformations across spatial and temporal scales.

## Scientific Background

Beaver (*Castor* spp.) are recognized as ecosystem engineers that significantly modify riparian habitats through dam building activities. These modifications create heterogeneous wetland complexes that influence:

1. **Hydrology**: Altered flow regimes, increased water residence time, and expanded riparian zones
2. **Vegetation**: Modified plant community composition and increased productivity
3. **Microclimates**: Changed temperature regimes and evapotranspiration patterns
4. **Ecosystem services**: Enhanced water quality, flood attenuation, and habitat provision

Despite growing recognition of beavers as restoration agents, quantitative assessment of their impacts has been limited by methodological challenges in measuring landscape-scale effects across heterogeneous environments. The Beaver Impacts Tool addresses this gap by leveraging Earth observation data with consistent spatio-temporal coverage to quantify beaver-induced landscape changes.

## Methodological Framework

### Data Sources and Processing Pipeline

The tool integrates multiple remote sensing datasets:

1. **Sentinel-2 Multispectral Imagery** (10m resolution)
   - Provides spectral bands for calculating vegetation and water indices
   - Cloud masking algorithms ensure data quality
   - Temporal compositing yields monthly cloud-free observations

2. **Landsat 8 Thermal Data**
   - Land Surface Temperature (LST) estimation at 30m resolution
   - Thermal regime characterization of beaver-influenced landscapes

3. **OpenET Evapotranspiration Data**
   - Ensemble evapotranspiration estimates
   - Quantification of beaver-influenced water cycling

4. **USGS 3DEP 10m Digital Elevation Model**
   - Topographic context for dam impact assessment
   - Elevation-based masking for hydrologically connected areas

5. **NHD and WWF Free-Flowing Rivers Datasets**
   - Waterway network connectivity analysis
   - Flow direction determination for upstream/downstream impact differentiation

### Analytical Approach

The tool employs a paired sampling design comparing:
- **Treatment sites**: User-identified beaver dams or beaver dam analogs (BDAs)
- **Control sites**: Algorithmically or user-defined non-dam locations along the same waterway networks

For each location, the tool:
1. Validates dam locations against waterway networks
2. Applies elevation-based buffers (±3m from dam elevation)
3. Differentiates upstream and downstream effects
4. Calculates monthly values of key indicators:
   - Normalized Difference Vegetation Index (NDVI)
   - Green Normalized Difference Water Index (NDWI)
   - Land Surface Temperature (LST)
   - Evapotranspiration (ET)
5. Produces statistical comparisons between dam and non-dam sites

## Technical Implementation

The Beaver Impacts Tool is implemented as a Streamlit web application with Google Earth Engine backend processing. Key components include:

1. **User Interface Layer**: Streamlit-based interface for data upload, parameter configuration, and visualization
2. **Geospatial Processing Engine**: Earth Engine API for server-side computation and data access
3. **Statistical Analysis Module**: Python-based statistical computation for trend analysis
4. **Visualization Framework**: Matplotlib/Seaborn for scientifically accurate data representation

## Data Processing Workflow

### 1. Initial Data Input Processing

#### 1.1 Data Upload and Validation
**Input Format**:
- CSV or GeoJSON files containing:
  - Required columns: latitude, longitude
  - Optional columns: date, id, additional metadata

**Processing Steps**:
1. **Data Validation**:
   - Verification of required columns
   - Coordinate range validation
   - Date format standardization

2. **Feature Collection Creation**:
   - Conversion of tabular data to Earth Engine FeatureCollection
   - Point geometry creation from coordinates
   - Property standardization

**Output Format**:
```python
{
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [longitude, latitude]
            },
            'properties': {
                'latitude': float,
                'longitude': float,
                'date': 'YYYY-MM-DD',
                'id_property': 'P{index}'
            }
        }
    ]
}
```

### 2. Buffer Zone Generation

#### 2.1 Buffer Creation
**Input**: FeatureCollection from Step 1
**Parameters**:
- Buffer radius (default: 150m)
- Elevation threshold (±3m)

**Processing Steps**:
1. **Buffer Generation**:
   - Circular buffer creation around each point
   - Preservation of original point geometry
   - Property inheritance and augmentation

2. **Elevation Masking**:
   - DEM data integration (USGS 3DEP 10m)
   - Elevation-based masking
   - Hydrologically connected area identification

**Output Format**:
```python
{
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [...]
            },
            'properties': {
                'Dam': 'positive/negative',
                'Survey_Date': 'YYYY-MM-DD',
                'Damdate': 'DamDate_YYYYMMDD',
                'Point_geo': original_point_geometry,
                'id_property': 'P{index}',
                'elevation_mask': elevation_image
            }
        }
    ]
}
```

### 3. Satellite Image Processing

#### 3.1 Time Series Image Collection
**Input**: Buffered FeatureCollection
**Parameters**:
- Time window: ±6 months from survey date
- Cloud coverage threshold: 20%

**Processing Steps**:
1. **Image Collection**:
   - Sentinel-2 image filtering
   - Temporal and spatial subsetting
   - Cloud masking implementation

2. **Quality Control**:
   - Cloud coverage calculation
   - Monthly least cloudy image selection
   - Band normalization

**Output Format**:
```python
{
    'type': 'ImageCollection',
    'images': [
        {
            'type': 'Image',
            'bands': [
                {
                    'name': 'B2',  # Blue
                    'type': 'float',
                    'values': [...]
                },
                {
                    'name': 'B3',  # Green
                    'type': 'float',
                    'values': [...]
                },
                {
                    'name': 'B4',  # Red
                    'type': 'float',
                    'values': [...]
                },
                {
                    'name': 'B8',  # NIR
                    'type': 'float',
                    'values': [...]
                },
                {
                    'name': 'cloudMask',
                    'type': 'int',
                    'values': [...]
                }
            ],
            'properties': {
                'Cloud_coverage': float,
                'Image_month': int,
                'Image_year': int,
                'Dam_status': str,
                'id_property': str
            }
        }
    ]
}
```

### 4. Metric Computation

#### 4.1 Vegetation and Water Indices
**Processing Steps**:
1. **NDVI Calculation**:
   ```python
   NDVI = (NIR - Red) / (NIR + Red)
   ```
   - Range: [-1, 1]
   - No-data handling: masked values excluded
   - Statistical aggregation: mean values per buffer

2. **NDWI Calculation**:
   ```python
   NDWI = (Green - NIR) / (Green + NIR)
   ```
   - Range: [-1, 1]
   - Water detection threshold: > 0.2
   - Statistical aggregation: mean values per buffer

#### 4.2 Thermal and Evapotranspiration Data
**Processing Steps**:
1. **LST Computation**:
   - Landsat 8 thermal band integration
   - Emissivity correction
   - Temperature conversion to Celsius
   - Statistical aggregation: mean values per buffer

2. **ET Data Integration**:
   - OpenET data assimilation
   - Monthly aggregation
   - Statistical aggregation: mean values per buffer

**Output Format**:
```python
{
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {
                'NDVI': float,
                'NDWI_Green': float,
                'LST': float,
                'ET': float,
                'Image_month': int,
                'Image_year': int,
                'Dam_status': str,
                'id_property': str
            }
        }
    ]
}
```

### 5. Data Quality Control and Validation

#### 5.1 Quality Control Measures
1. **Cloud Coverage**:
   - Threshold: < 20% cloud coverage
   - Monthly selection: least cloudy image
   - No-data handling: masked values excluded

2. **Elevation Masking**:
   - Threshold: ±3m from dam elevation
   - Hydrological connectivity: ensured through DEM analysis
   - Edge effects: excluded from analysis

3. **Statistical Validation**:
   - Outlier detection: 3-sigma rule
   - Missing data: interpolated using temporal neighbors
   - Confidence intervals: 95% level

### 6. Visualization and Data Export

#### 6.1 Data Transformation
**Processing Steps**:
1. **Data Conversion**:
   - Earth Engine to Pandas DataFrame conversion
   - Temporal aggregation: monthly means
   - Spatial aggregation: buffer means

2. **Data Cleaning**:
   - Missing value handling
   - Outlier removal
   - Data type standardization

**Output Format**:
```python
DataFrame columns:
- Image_month: int (1-12)
- Image_year: int
- Dam_status: str ('Dam'/'Non-dam')
- NDVI: float
- NDWI_Green: float
- LST: float
- ET: float
- longitude: float
- latitude: float
```

#### 6.2 Visualization Generation
**Processing Steps**:
1. **Time Series Plotting**:
   - Monthly aggregation
   - Confidence interval calculation
   - Trend line fitting

2. **Statistical Analysis**:
   - Mean difference calculation
   - Significance testing
   - Seasonal pattern identification

3. **Export Options**:
   - PNG format: high-resolution plots
   - CSV format: processed data
   - Metadata: processing parameters

**Special Considerations**:
1. **Zero Value Handling**:
   - Cloud-covered pixels: masked
   - Invalid measurements: excluded
   - Edge effects: removed

2. **Statistical Aggregation**:
   - Mean values: weighted by valid pixels
   - Confidence intervals: 95% level
   - Seasonal patterns: monthly aggregation

3. **Data Quality Indicators**:
   - Cloud coverage percentage
   - Valid pixel count
   - Statistical significance

## Workflow Architecture

The analytical workflow consists of six sequential stages:

1. **Dam Location Upload**: Import of spatially-explicit dam locations via CSV or GeoJSON
2. **Waterway Network Selection**: Integration with hydrological network data
3. **Location Validation**: Distance-based validation against waterway networks
4. **Control Site Selection**: Generation of hydrologically connected non-dam reference points
5. **Analysis Buffer Generation**: Creation of elevation-constrained analysis zones
6. **Metric Computation and Visualization**: Statistical analysis and trend visualization

## Validation and Limitations

The tool implements several quality control mechanisms:
- Cloud coverage filtering for optimal image selection
- Elevation-based masking to constrain analysis to hydrologically relevant areas
- Distance-based validation of dam locations against waterway networks

Current limitations include:
- OpenET data availability constraints in eastern United States and for years prior to 2018
- Coarse approximation of up/downstream delineation in complex waterways
- Spatial resolution constraints based on underlying satellite data (10-30m)

## Applications

The Beaver Impacts Tool enables:
1. **Before-After Impact Assessment**: Quantification of beaver restoration effects
2. **Spatial Comparison Studies**: Evaluation of dam effectiveness across environmental gradients
3. **Temporal Trend Analysis**: Monitoring of seasonal and annual variation in beaver-influenced ecosystems
4. **Restoration Planning**: Identification of potential restoration sites based on predicted impacts

## Technical Requirements

The application is built upon Python 3.7+ with dependencies including:
- Earth Engine API for Google Earth Engine access
- Streamlit for web interface
- Geospatial libraries (Folium, GeoPandas)
- Scientific computing packages (NumPy, Pandas, SciPy)
- Visualization tools (Matplotlib, Seaborn)
