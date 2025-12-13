"""
Generate Synthetic Training Data for DSIG 1 Solution
Creates synthetic soil data with growth stages, NPK values, and NDVI/NDRE correlations
Based on ICAR 2023-24 standards for potato farming
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_synthetic_soil_data(n_samples=200, output_path="data/synthetic_soil_data.csv"):
    """
    Generate synthetic soil and crop data for training
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the CSV file
    
    Returns:
        DataFrame with synthetic data
    """
    print(f"Generating {n_samples} synthetic data samples...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base data
    data = {
        'latitude': np.round(11.25 + np.random.uniform(0, 0.1, n_samples), 4),
        'longitude': np.round(78.15 + np.random.uniform(0, 0.1, n_samples), 4),
        'growth_stage': np.random.choice(['Vegetative', 'Tuber Initiation', 'Bulking'], n_samples),
        'pH': np.round(np.random.uniform(5.5, 7.5, n_samples), 1)
    }
    
    df = pd.DataFrame(data)
    
    # Stage-specific NPK (based on ICAR 2023-24)
    def assign_npk(stage):
        """Assign NPK values based on growth stage"""
        if stage == 'Vegetative':
            return np.random.randint(60, 100), np.random.randint(30, 50), np.random.randint(50, 80)
        if stage == 'Tuber Initiation':
            return np.random.randint(100, 140), np.random.randint(40, 60), np.random.randint(70, 100)
        # Bulking
        return np.random.randint(140, 200), np.random.randint(50, 80), np.random.randint(90, 140)
    
    # Assign NPK values
    df[['N', 'P', 'K']] = df['growth_stage'].apply(lambda x: pd.Series(assign_npk(x)))
    
    # NDVI/NDRE correlation with Nitrogen
    df['NDVI'] = np.round(0.2 + (df['N'] / 200) * 0.6 + np.random.uniform(-0.1, 0.1, n_samples), 2)
    df['NDRE'] = np.round(df['NDVI'] * 0.8 + np.random.uniform(-0.1, 0.1, n_samples), 2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"[OK] Synthetic data saved to: {output_path}")
    print(f"\nData Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Growth stages: {df['growth_stage'].value_counts().to_dict()}")
    print(f"\nNPK Statistics:")
    print(f"  Nitrogen (N): {df['N'].min()}-{df['N'].max()} kg/ha (mean: {df['N'].mean():.1f})")
    print(f"  Phosphorus (P): {df['P'].min()}-{df['P'].max()} kg/ha (mean: {df['P'].mean():.1f})")
    print(f"  Potassium (K): {df['K'].min()}-{df['K'].max()} kg/ha (mean: {df['K'].mean():.1f})")
    print(f"\nNDVI Statistics:")
    print(f"  NDVI: {df['NDVI'].min():.2f}-{df['NDVI'].max():.2f} (mean: {df['NDVI'].mean():.2f})")
    print(f"  NDRE: {df['NDRE'].min():.2f}-{df['NDRE'].max():.2f} (mean: {df['NDRE'].mean():.2f})")
    
    return df


def generate_patch_dataset(n_samples=200, patch_size=64, output_dir="data"):
    """
    Generate synthetic image patches for CNN training
    
    Args:
        n_samples: Number of patches to generate
        patch_size: Size of each patch (patch_size x patch_size)
        output_dir: Directory to save patches
    
    Returns:
        Tuple of (patches, labels) arrays
    """
    print(f"\nGenerating {n_samples} synthetic image patches ({patch_size}x{patch_size})...")
    
    np.random.seed(42)
    
    # Load growth stage mapping
    stage_map = {
        'Vegetative': 0,
        'Tuber Initiation': 1,
        'Bulking': 2
    }
    
    # Load soil data to get realistic correlations
    soil_data_path = os.path.join(output_dir, "synthetic_soil_data.csv")
    if os.path.exists(soil_data_path):
        soil_df = pd.read_csv(soil_data_path)
    else:
        print("Soil data not found. Generating patches with random values...")
        soil_df = None
    
    patches = []
    labels = []
    nitrogen_values = []
    
    for i in range(n_samples):
        # Generate patch (13 bands for Sentinel-2)
        patch = np.random.rand(patch_size, patch_size, 13) * 10000
        
        # If we have soil data, use it to create more realistic patches
        if soil_df is not None and i < len(soil_df):
            row = soil_df.iloc[i]
            # Adjust patch based on NDVI and growth stage
            ndvi_factor = row['NDVI']
            stage = row['growth_stage']
            nitrogen = row['N']
            
            # Simulate realistic band values based on NDVI
            # NIR (B08) - higher for vegetation
            patch[:, :, 3] = patch[:, :, 3] * (1 + ndvi_factor * 2)
            # Red (B04) - lower for vegetation
            patch[:, :, 2] = patch[:, :, 2] * (1 - ndvi_factor * 0.5)
            
            # Add stage-specific characteristics
            if stage == 'Vegetative':
                # More green in early stages
                patch[:, :, 1] = patch[:, :, 1] * 1.2
            elif stage == 'Bulking':
                # More red edge in bulking
                patch[:, :, 4] = patch[:, :, 4] * 1.3
            
            label = stage_map[stage]
            nitrogen_values.append(nitrogen)
        else:
            # Random assignment
            stage = np.random.choice(['Vegetative', 'Tuber Initiation', 'Bulking'])
            label = stage_map[stage]
            nitrogen_values.append(np.random.randint(60, 200))
        
        patches.append(patch)
        labels.append(label)
    
    patches = np.array(patches)
    labels = np.array(labels)
    nitrogen_values = np.array(nitrogen_values)
    
    # Save patches
    np.save(os.path.join(output_dir, "patches.npy"), patches)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "nitrogen_values.npy"), nitrogen_values)
    
    print(f"[OK] Patches saved to: {output_dir}/")
    print(f"  - patches.npy: {patches.shape}")
    print(f"  - labels.npy: {labels.shape}")
    print(f"  - nitrogen_values.npy: {nitrogen_values.shape}")
    
    return patches, labels, nitrogen_values


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--patches", action="store_true", help="Also generate image patches")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size for CNN")
    parser.add_argument("--output", type=str, default="data/synthetic_soil_data.csv", 
                       help="Output CSV path")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Synthetic Data Generator for DSIG 1 Solution")
    print("="*60)
    
    # Generate soil data
    df = generate_synthetic_soil_data(n_samples=args.samples, output_path=args.output)
    
    # Generate patches if requested
    if args.patches:
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "data"
        generate_patch_dataset(n_samples=args.samples, 
                             patch_size=args.patch_size,
                             output_dir=output_dir)
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60)

