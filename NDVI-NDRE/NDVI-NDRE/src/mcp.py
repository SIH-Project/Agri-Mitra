"""
MCP (Model Control Protocol) Trigger System
Monitors NDVI changes and triggers classification/nutrient prediction updates
"""

import os
import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

class MCPTrigger:
    """MCP system to trigger model updates based on NDVI changes"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = self.config['paths']['output_dir']
        self.threshold = self.config['pipeline']['ndvi_change_threshold']
        self.dashboard_path = os.path.join(self.output_dir, "dashboard_data.json")
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_previous_ndvi(self):
        """Load previous week's NDVI"""
        prev_ndvi_path = os.path.join(self.output_dir, "ndvi_previous.npy")
        if os.path.exists(prev_ndvi_path):
            return np.load(prev_ndvi_path)
        return None
    
    def save_current_ndvi(self, ndvi):
        """Save current NDVI as previous for next week"""
        prev_ndvi_path = os.path.join(self.output_dir, "ndvi_previous.npy")
        np.save(prev_ndvi_path, ndvi)
    
    def calculate_ndvi_change(self, current_ndvi, previous_ndvi):
        """
        Calculate NDVI change between current and previous week
        
        Returns:
            mean_change: Mean absolute change
            change_map: Pixel-wise change map
        """
        if previous_ndvi is None:
            return None, None
        
        # Resize if shapes don't match
        if current_ndvi.shape != previous_ndvi.shape:
            from scipy.ndimage import zoom
            zoom_factors = (current_ndvi.shape[0] / previous_ndvi.shape[0],
                          current_ndvi.shape[1] / previous_ndvi.shape[1])
            previous_ndvi = zoom(previous_ndvi, zoom_factors, order=1)
        
        # Calculate change
        change_map = current_ndvi - previous_ndvi
        mean_change = np.nanmean(np.abs(change_map))
        
        return mean_change, change_map
    
    def should_trigger(self, mean_change):
        """
        Check if change exceeds threshold to trigger model rerun
        
        Args:
            mean_change: Mean absolute NDVI change
        
        Returns:
            bool: True if should trigger
        """
        if mean_change is None:
            return True  # First run, always trigger
        
        return mean_change > self.threshold
    
    def update_dashboard(self, report_data):
        """
        Update dashboard data JSON
        
        Args:
            report_data: Dictionary with metrics, stages, nitrogen, etc.
        """
        dashboard_data = {
            "last_updated": datetime.now().isoformat(),
            "growth_stages": report_data.get("growth_stages", {}),
            "dominant_stage": report_data.get("dominant_stage", "Unknown"),
            "nitrogen": report_data.get("nitrogen", {}),
            "ndvi": report_data.get("ndvi", {}),
            "status": "active"
        }
        
        with open(self.dashboard_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(f"Dashboard updated: {self.dashboard_path}")
        return dashboard_data
    
    def trigger_classification(self, current_ndvi, report_data):
        """
        MCP trigger logic: Check NDVI change and trigger updates if needed
        
        Args:
            current_ndvi: Current week's NDVI array
            report_data: Report data from pipeline
        
        Returns:
            triggered: True if models were triggered
            change_info: Change statistics
        """
        print("="*50)
        print("MCP Trigger System")
        print("="*50)
        
        # Load previous NDVI
        previous_ndvi = self.load_previous_ndvi()
        
        # Calculate change
        mean_change, change_map = self.calculate_ndvi_change(current_ndvi, previous_ndvi)
        
        if mean_change is not None:
            print(f"Mean NDVI change: {mean_change:.4f}")
            print(f"Threshold: {self.threshold}")
        
        # Check if should trigger
        should_trigger = self.should_trigger(mean_change)
        
        if should_trigger:
            print("✓ NDVI change exceeds threshold - Triggering model updates")
            
            # Update dashboard
            dashboard_data = self.update_dashboard(report_data)
            
            # Save current NDVI as previous
            self.save_current_ndvi(current_ndvi)
            
            change_info = {
                "triggered": True,
                "mean_change": float(mean_change) if mean_change is not None else None,
                "threshold": self.threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            return True, change_info
        else:
            print(f"✗ NDVI change ({mean_change:.4f}) below threshold - No trigger needed")
            
            change_info = {
                "triggered": False,
                "mean_change": float(mean_change) if mean_change is not None else None,
                "threshold": self.threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            return False, change_info


if __name__ == "__main__":
    # Example usage
    mcp = MCPTrigger()
    
    # Simulate current NDVI
    current_ndvi = np.random.rand(100, 100) * 0.5 + 0.3
    
    # Simulate report data
    report_data = {
        "growth_stages": {
            "Vegetative": {"count": 100, "percentage": 25.0},
            "Tuber_Initiation": {"count": 150, "percentage": 37.5},
            "Bulking": {"count": 100, "percentage": 25.0},
            "Maturation": {"count": 50, "percentage": 12.5}
        },
        "dominant_stage": "Tuber_Initiation",
        "nitrogen": {
            "mean_kg_per_ha": 150.5,
            "std_kg_per_ha": 25.3
        },
        "ndvi": {
            "mean": 0.65,
            "std": 0.15
        }
    }
    
    triggered, change_info = mcp.trigger_classification(current_ndvi, report_data)
    print(f"\nTriggered: {triggered}")
    print(f"Change Info: {change_info}")

