"""
Image Processing Module for Sentinel-2 JP2 Files
Calculates NDVI, NDRE, and processes bands for ML pipeline
"""

import rasterio
import numpy as np
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2

class ImageProcessor:
    """Process Sentinel-2 JP2 images and calculate vegetation indices"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.r10_dir = self.config['paths']['r10_dir']
        self.r20_dir = self.config['paths']['r20_dir']
        self.output_dir = self.config['paths']['output_dir']
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_band(self, band_name, resolution="10m"):
        """
        Load a JP2 band file
        
        Args:
            band_name: Band name (e.g., 'B04', 'B08')
            resolution: '10m' or '20m'
        
        Returns:
            numpy array of band data
        """
        dir_path = self.r10_dir if resolution == "10m" else self.r20_dir
        
        # Find matching file
        pattern = f"*_{band_name}_*{resolution}.jp2"
        files = list(Path(dir_path).glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"Band {band_name} at {resolution} not found")
        
        with rasterio.open(str(files[0])) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
        
        return data, transform, crs
    
    def calculate_ndvi(self):
        """
        Calculate NDVI: (NIR - Red) / (NIR + Red)
        Uses B08 (NIR) and B04 (Red) at 10m resolution
        """
        print("Calculating NDVI...")
        
        # Load bands
        red, transform, crs = self.load_band("B04", "10m")
        nir, _, _ = self.load_band("B08", "10m")
        
        # Convert to float to avoid overflow
        red = red.astype(np.float32)
        nir = nir.astype(np.float32)
        
        # Calculate NDVI
        denominator = nir + red
        denominator[denominator == 0] = np.nan  # Avoid division by zero
        
        ndvi = (nir - red) / denominator
        
        # Clip values to [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi, transform, crs
    
    def calculate_ndre(self):
        """
        Calculate NDRE: (NIR - RedEdge) / (NIR + RedEdge)
        Uses B08 (NIR) at 10m and B05 (RedEdge) at 20m
        """
        print("Calculating NDRE...")
        
        # Load bands
        nir, transform, crs = self.load_band("B08", "10m")
        rededge, _, _ = self.load_band("B05", "20m")
        
        # Resample 20m to 10m (simple upsampling)
        if rededge.shape != nir.shape:
            rededge = cv2.resize(rededge.astype(np.float32), 
                                (nir.shape[1], nir.shape[0]), 
                                interpolation=cv2.INTER_CUBIC)
        
        # Convert to float
        nir = nir.astype(np.float32)
        rededge = rededge.astype(np.float32)
        
        # Calculate NDRE
        denominator = nir + rededge
        denominator[denominator == 0] = np.nan
        
        ndre = (nir - rededge) / denominator
        ndre = np.clip(ndre, -1, 1)
        
        return ndre, transform, crs
    
    def create_nutrient_map(self, ndvi, output_name="Nutrient_Map_Enhanced.png"):
        """
        Create color-coded nutrient map based on NDVI thresholds
        
        Args:
            ndvi: NDVI array
            output_name: Output filename
        """
        print("Creating nutrient map...")
        
        thresholds = self.config['indices']['ndvi']['thresholds']
        low_threshold = thresholds['low_fertility']
        high_threshold = thresholds['high_fertility']
        
        # Create classified map
        nutrient_map = np.zeros_like(ndvi)
        nutrient_map[ndvi < low_threshold] = 0  # Low fertility (red)
        nutrient_map[(ndvi >= low_threshold) & (ndvi < high_threshold)] = 1  # Medium (yellow)
        nutrient_map[ndvi >= high_threshold] = 2  # High fertility (green)
        
        # Create custom colormap: Red -> Yellow -> Green
        colors = ['red', 'yellow', 'green']
        n_bins = 3
        cmap = LinearSegmentedColormap.from_list('nutrient', colors, N=n_bins)
        
        # Plot
        plt.figure(figsize=(12, 10))
        plt.imshow(nutrient_map, cmap=cmap, vmin=0, vmax=2)
        plt.colorbar(label='Nutrient Level', ticks=[0, 1, 2], 
                    format=['Low', 'Medium', 'High'])
        plt.title('Nutrient Health Map (NDVI-based)', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Save
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Nutrient map saved to {output_path}")
        return nutrient_map
    
    def stack_all_bands(self):
        """
        Stack all available bands into a single array for ML processing
        Returns: (H, W, C) array where C is number of bands
        """
        print("Stacking all bands...")
        
        bands_data = []
        band_names = []
        
        # 10m bands
        for band in ['B02', 'B03', 'B04', 'B08']:
            try:
                data, transform, crs = self.load_band(band, "10m")
                bands_data.append(data)
                band_names.append(f"{band}_10m")
            except FileNotFoundError:
                print(f"Warning: {band} at 10m not found")
        
        # 20m bands (need to resample to 10m)
        base_shape = bands_data[0].shape if bands_data else None
        
        for band in ['B05', 'B11', 'B12']:
            try:
                data, _, _ = self.load_band(band, "20m")
                if base_shape and data.shape != base_shape:
                    data = cv2.resize(data.astype(np.float32), 
                                    (base_shape[1], base_shape[0]), 
                                    interpolation=cv2.INTER_CUBIC)
                bands_data.append(data)
                band_names.append(f"{band}_20m")
            except FileNotFoundError:
                print(f"Warning: {band} at 20m not found")
        
        if not bands_data:
            raise ValueError("No bands found!")
        
        # Stack bands: (H, W, C)
        stacked = np.stack(bands_data, axis=-1)
        
        print(f"Stacked {len(bands_data)} bands: {band_names}")
        print(f"Shape: {stacked.shape}")
        
        return stacked, transform, crs, band_names
    
    def process_images(self):
        """Main processing function"""
        print("="*50)
        print("Processing Sentinel-2 Images")
        print("="*50)
        
        # Calculate indices
        ndvi, transform, crs = self.calculate_ndvi()
        ndre, _, _ = self.calculate_ndre()
        
        # Create nutrient map
        nutrient_map = self.create_nutrient_map(ndvi)
        
        # Stack bands for ML
        stacked_bands, _, _, band_names = self.stack_all_bands()
        
        # Save NDVI and NDRE as numpy arrays
        np.save(os.path.join(self.output_dir, "ndvi.npy"), ndvi)
        np.save(os.path.join(self.output_dir, "ndre.npy"), ndre)
        np.save(os.path.join(self.output_dir, "stacked_bands.npy"), stacked_bands)
        
        print("\nProcessing complete!")
        print(f"Outputs saved to: {self.output_dir}")
        
        return {
            'ndvi': ndvi,
            'ndre': ndre,
            'nutrient_map': nutrient_map,
            'stacked_bands': stacked_bands,
            'band_names': band_names,
            'transform': transform,
            'crs': crs
        }


if __name__ == "__main__":
    processor = ImageProcessor()
    results = processor.process_images()

