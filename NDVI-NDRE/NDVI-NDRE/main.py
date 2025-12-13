"""
Main Pipeline Script for Weekly Automated Processing
Downloads Sentinel-2 images, processes them, runs ML models, and generates reports
"""

import os
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import time

from src.image_processor import ImageProcessor
from src.models import GrowthStageClassifier, NitrogenPredictor
import numpy as np

class WeeklyPipeline:
    """Automated weekly processing pipeline"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = self.config['paths']['output_dir']
        self.data_dir = self.config['paths']['data_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.processor = ImageProcessor(config_path)
        self.stage_classifier = GrowthStageClassifier(config_path)
        self.nitrogen_predictor = NitrogenPredictor(config_path)
        
        # Load models if available
        self.stage_classifier.load_model()
        self.nitrogen_predictor.load_model()
    
    def download_images(self, date=None):
        """
        Download Sentinel-2 images for a specific date
        In production, integrate with sentinelsat or Google Earth Engine
        """
        print(f"Downloading images for {date or 'latest'}...")
        # Placeholder: In production, use sentinelsat API or GEE
        # For now, assume images are already in r10/ and r20/ folders
        print("Using existing images in r10/ and r20/ folders")
        return True
    
    def process_images(self):
        """Process images and calculate indices"""
        print("Processing images...")
        results = self.processor.process_images()
        return results
    
    def classify_growth_stages(self, stacked_bands, patch_size=64):
        """
        Classify growth stages using CNN
        Divides image into patches and classifies each
        """
        print("Classifying growth stages...")
        
        if self.stage_classifier.model is None:
            print("Warning: CNN model not loaded. Using random predictions.")
            # Create dummy patches for demonstration
            h, w, c = stacked_bands.shape
            n_patches_h = h // patch_size
            n_patches_w = w // patch_size
            
            stages = np.random.randint(0, 4, size=(n_patches_h, n_patches_w))
            stage_probs = np.random.rand(n_patches_h, n_patches_w, 4)
            stage_probs = stage_probs / stage_probs.sum(axis=2, keepdims=True)
            
            return stages, stage_probs
        
        # Extract patches
        h, w, c = stacked_bands.shape
        patches = []
        patch_coords = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = stacked_bands[i:i+patch_size, j:j+patch_size, :]
                patches.append(patch)
                patch_coords.append((i, j))
        
        patches = np.array(patches)
        
        # Predict
        stage_labels, stage_probs = self.stage_classifier.predict(patches)
        
        # Reshape to grid
        n_patches_h = len(range(0, h - patch_size, patch_size))
        n_patches_w = len(range(0, w - patch_size, patch_size))
        stages = stage_labels.reshape(n_patches_h, n_patches_w)
        
        return stages, stage_probs
    
    def predict_nitrogen(self, ndvi, ndre):
        """
        Predict Nitrogen levels using Random Forest
        """
        print("Predicting Nitrogen levels...")
        
        if self.nitrogen_predictor.model is None:
            print("Warning: RF model not loaded. Using NDVI-based estimation.")
            # Simple NDVI-based estimation
            nitrogen = 100 + ndvi * 100  # kg/ha
            return nitrogen
        
        # Extract features (in production, use more sophisticated feature extraction)
        h, w = ndvi.shape
        features = []
        coords = []
        
        for i in range(0, h, 100):  # Sample every 100 pixels
            for j in range(0, w, 100):
                features.append([ndvi[i, j], ndre[i, j]])
                coords.append((i, j))
        
        features = np.array(features)
        nitrogen_pred = self.nitrogen_predictor.predict(features)
        
        # Create full map
        nitrogen_map = np.zeros_like(ndvi)
        for idx, (i, j) in enumerate(coords):
            nitrogen_map[i, j] = nitrogen_pred[idx]
        
        # Interpolate for full coverage (simplified: fill with mean)
        nitrogen_map = np.where(nitrogen_map == 0, 
                               np.nanmean(nitrogen_map) if np.any(nitrogen_map > 0) else 150.0, 
                               nitrogen_map)
        
        return nitrogen_map
    
    def generate_report(self, results, stages, nitrogen_map):
        """
        Generate weekly report JSON
        """
        print("Generating weekly report...")
        
        # Calculate statistics
        stage_names = self.config['growth_stages']
        stage_counts = {}
        for stage_id in range(len(stage_names)):
            count = np.sum(stages == stage_id)
            percentage = (count / stages.size) * 100
            stage_counts[stage_names[stage_id]] = {
                "count": int(count),
                "percentage": round(percentage, 2)
            }
        
        dominant_stage = stage_names[np.argmax([stage_counts[s]["count"] for s in stage_names])]
        
        # Nitrogen statistics
        nitrogen_mean = np.nanmean(nitrogen_map)
        nitrogen_std = np.nanstd(nitrogen_map)
        
        # NDVI statistics
        ndvi_mean = np.nanmean(results['ndvi'])
        ndvi_std = np.nanstd(results['ndvi'])
        
        report = {
            "date": datetime.now().strftime("%Y%m%d"),
            "timestamp": datetime.now().isoformat(),
            "growth_stages": stage_counts,
            "dominant_stage": dominant_stage,
            "nitrogen": {
                "mean_kg_per_ha": round(float(nitrogen_mean), 2),
                "std_kg_per_ha": round(float(nitrogen_std), 2),
                "low_zones": int(np.sum(nitrogen_map < 140)),
                "optimal_zones": int(np.sum((nitrogen_map >= 140) & (nitrogen_map <= 180))),
                "high_zones": int(np.sum(nitrogen_map > 180))
            },
            "ndvi": {
                "mean": round(float(ndvi_mean), 3),
                "std": round(float(ndvi_std), 3),
                "min": round(float(np.nanmin(results['ndvi'])), 3),
                "max": round(float(np.nanmax(results['ndvi'])), 3)
            },
            "processing_status": "completed"
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, f"weekly_report_{report['date']}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {report_path}")
        return report
    
    def run_weekly_processing(self):
        """Execute full weekly processing pipeline"""
        print("="*60)
        print(f"Weekly Processing Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        try:
            # 1. Download images (if needed)
            self.download_images()
            
            # 2. Process images
            results = self.process_images()
            
            # 3. Classify growth stages
            stages, stage_probs = self.classify_growth_stages(results['stacked_bands'])
            
            # 4. Predict Nitrogen
            nitrogen_map = self.predict_nitrogen(results['ndvi'], results['ndre'])
            
            # 5. Generate report
            report = self.generate_report(results, stages, nitrogen_map)
            
            print("\n" + "="*60)
            print("Pipeline completed successfully!")
            print("="*60)
            
            return report
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def run_scheduled():
    """Run pipeline on schedule (every Monday at 6 AM)"""
    pipeline = WeeklyPipeline()
    
    # Schedule weekly run
    schedule.every().monday.at("06:00").do(pipeline.run_weekly_processing)
    
    print("Pipeline scheduler started. Runs every Monday at 6:00 AM")
    print("Press Ctrl+C to stop")
    
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        # Run with scheduler
        run_scheduled()
    else:
        # Run once immediately
        pipeline = WeeklyPipeline()
        pipeline.run_weekly_processing()

