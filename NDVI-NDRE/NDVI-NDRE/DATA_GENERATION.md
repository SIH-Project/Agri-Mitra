# Data Generation Guide

## Overview

The synthetic data generation script creates realistic training data for the DSIG 1 solution, based on ICAR 2023-24 standards for potato farming.

## Generated Data Format

The `src/generate_data.py` script creates `data/synthetic_soil_data.csv` with the following columns:

- **latitude**: GPS latitude (11.25 to 11.35)
- **longitude**: GPS longitude (78.15 to 78.25)
- **growth_stage**: One of ['Vegetative', 'Tuber Initiation', 'Bulking']
- **pH**: Soil pH (5.5 to 7.5)
- **N**: Nitrogen (kg/ha) - Stage-specific:
  - Vegetative: 60-100 kg/ha
  - Tuber Initiation: 100-140 kg/ha
  - Bulking: 140-200 kg/ha
- **P**: Phosphorus (kg/ha) - Stage-specific:
  - Vegetative: 30-50 kg/ha
  - Tuber Initiation: 40-60 kg/ha
  - Bulking: 50-80 kg/ha
- **K**: Potassium (kg/ha) - Stage-specific:
  - Vegetative: 50-80 kg/ha
  - Tuber Initiation: 70-100 kg/ha
  - Bulking: 90-140 kg/ha
- **NDVI**: Normalized Difference Vegetation Index (0.2 to 0.8)
  - Correlated with Nitrogen: `NDVI = 0.2 + (N/200) * 0.6 + noise`
- **NDRE**: Normalized Difference Red Edge (0.2 to 0.7)
  - Correlated with NDVI: `NDRE = NDVI * 0.8 + noise`

## Usage

### Basic Usage

```bash
python src/generate_data.py --samples 200
```

### Options

- `--samples N`: Number of samples to generate (default: 200)
- `--patches`: Also generate image patches for CNN training
- `--patch-size SIZE`: Size of patches (default: 64)
- `--output PATH`: Output CSV path (default: `data/synthetic_soil_data.csv`)

### Example

```bash
# Generate 500 samples
python src/generate_data.py --samples 500

# Generate data with image patches
python src/generate_data.py --samples 200 --patches --patch-size 64
```

## Output

The script generates:

1. **`data/synthetic_soil_data.csv`**: Main training data with all features
2. **`data/patches.npy`** (if `--patches`): Image patches for CNN
3. **`data/labels.npy`** (if `--patches`): Growth stage labels
4. **`data/nitrogen_values.npy`** (if `--patches`): Nitrogen values

## Integration with Models

The generated CSV is automatically used by:

1. **`src/models.py`**: 
   - Loads CSV to create realistic image patches
   - Uses NDVI/NDRE values for feature extraction
   - Uses Nitrogen values for Random Forest training

2. **Training Command**:
   ```bash
   python src/models.py --train --samples 200 --data-dir data
   ```

## Data Statistics

For 200 samples, typical distribution:
- Growth stages: ~84 Vegetative, ~57 Tuber Initiation, ~59 Bulking
- Nitrogen range: 60-200 kg/ha (mean: ~118 kg/ha)
- NDVI range: 0.30-0.89 (mean: ~0.56)
- NDRE range: 0.20-0.75 (mean: ~0.45)

## Notes

- Data is based on ICAR 2023-24 standards for Tamil Nadu potato farming
- NPK values are stage-specific to reflect real crop nutrient requirements
- NDVI/NDRE correlations simulate realistic satellite imagery relationships
- Random seed (42) ensures reproducibility

## Next Steps

After generating data:

1. **Train models**: `python src/models.py --train`
2. **Process images**: `python src/image_processor.py`
3. **Run pipeline**: `python main.py`

