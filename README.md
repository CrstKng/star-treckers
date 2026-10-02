# Star Treckers

## Members
Anastasia, Dariy, Emanuela, Michael, Rumen

## Goal
Create a lost-in-space star tracker that is fast, reliable and efficient.

## Overview
Star Treckers is a star identification and attitude determination system designed for spacecraft navigation. The system processes star field images to identify stars and determine the spacecraft's orientation in space using advanced algorithms and a comprehensive star catalog.

## Architecture
The system consists of two main pipelines:

### Image Recognition Pipeline
- **Purpose**: Detects stars in captured images and provides unit vectors for each detected star
- **Key Components**:
  - `capture_star_vectors.py`: Main star detection algorithm using OpenCV
  - `star_frame.py`: Star frame processing utilities
  - Image preprocessing with Gaussian blur and thresholding
  - Connected component analysis for star identification
  - Unit vector calculation for detected stars

### Catalog Mapping Pipeline
- **Purpose**: Applies algorithms for catalog matching, star identification, and attitude determination
- **Key Components**:
  - `identify_stars.py`: Star identification using pairwise angle matching
  - `quest.py`: QUEST algorithm for attitude determination
  - `db_operations.py`: Database operations for star catalog
  - `radec_to_vec.py`: Coordinate conversion utilities
  - `real_image_valuation.py`: Main evaluation script for real images

## Features
- **Star Detection**: Robust detection of stars in star field images
- **Catalog Matching**: Efficient matching against HIP star catalog
- **Attitude Determination**: QUEST algorithm implementation for 3D orientation
- **Database Integration**: SQLite-based star catalog with precomputed pairs
- **Error Analysis**: Comprehensive error metrics and validation

## Dependencies
- **Core Libraries**: numpy, opencv-python, matplotlib, scikit-image
- **Astronomy**: astropy for astronomical calculations
- **Optimization**: scipy for scientific computing
- **Database**: sqlite3 for star catalog storage

## Installation
1. Navigate to the `src/` directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For OpenCV installation issues, use:
   ```bash
   pip install --only-binary=:all: opencv-python
   pip install "numpy<2.0"
   pip install scikit-image
   ```

## Usage
### Basic Usage
Run the main evaluation script for test sequences:
```bash
cd src/catalog_pipeline
python real_image_valuation.py
```

### Testing Mode (Simulated Quaternion)
To run the system in testing mode without connecting to the Raspberry Pi, use the `--test` flag. This will generate and print a simulated quaternion (in the same format as the real system) and its corresponding yaw, pitch, and roll angles:
```bash
cd src
python main.py --test
```
This is useful for development and frontend-backend integration without hardware.

### Image Processing
Process star field images:
```bash
cd src/image_pipeline
python capture_star_vectors.py
```

### Database Operations
Initialize and populate the star catalog:
```bash
cd src/catalog_pipeline
python fill_db.py
python index_data.py
```
### MAC IP Operation
```bash
ipconfig getifaddr en0
```

## Project Structure
```
star-treckers/
├── src/
│   ├── image_pipeline/          # Star detection and image processing
│   │   ├── capture_star_vectors.py
│   │   ├── star_frame.py
│   │   └── Taken Test Images/   # Test image collection
│   ├── catalog_pipeline/        # Star identification and attitude determination
│   │   ├── real_image_valuation.py
│   │   ├── identify_stars.py
│   │   ├── quest.py
│   │   ├── db_operations.py
│   │   ├── star_catalog.db      # Star catalog database
│   │   └── testing/            # Test scripts
│   └── requirements.txt
└── docs/                       # Documentation
```

## Documentation
Additional documentation is available at: [Google Drive Documentation](https://drive.google.com/drive/folders/11sBpqvF0sbLGrn1AJVCqiCKHf7mg8zti?usp=sharing)


## Motivation

## Quick Start

## Contributing
