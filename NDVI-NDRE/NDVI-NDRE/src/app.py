"""
Flask API for Mobile/Web App Integration
Endpoints for predictions, recommendations, and dashboard data
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import yaml
import numpy as np
from datetime import datetime

from src.image_processor import ImageProcessor
from src.models import GrowthStageClassifier, NitrogenPredictor
from src.agentic_ai import RecommendationAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app

# Initialize components
config_path = "config.yaml"
processor = ImageProcessor(config_path)
stage_classifier = GrowthStageClassifier(config_path)
nitrogen_predictor = NitrogenPredictor(config_path)
recommendation_agent = RecommendationAgent(config_path)

# Load models if available
stage_classifier.load_model()
nitrogen_predictor.load_model()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "growth_stage": stage_classifier.model is not None,
            "nitrogen": nitrogen_predictor.model is not None
        }
    })


@app.route('/predict-gee', methods=['POST'])
def predict_from_gee():
    """
    Predict from Google Earth Engine coordinates
    Accepts GPS coordinates and returns NDVI, growth stage, nitrogen
    
    Request body:
    {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "date": "2025-11-03"  # optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({"error": "Missing latitude/longitude"}), 400
        
        lat = data['latitude']
        lon = data['longitude']
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        # In production, fetch from GEE
        # For now, use local processing
        results = processor.process_images()
        
        # Get NDVI at coordinates (simplified - in production, use actual geo-referencing)
        ndvi_value = np.nanmean(results['ndvi'])
        ndre_value = np.nanmean(results['ndre'])
        
        # Predict nitrogen
        features = np.array([[ndvi_value, ndre_value]])
        nitrogen_pred = nitrogen_predictor.predict(features)[0] if nitrogen_predictor.model else 150.0
        
        # Classify growth stage (simplified)
        stage_labels = ["Vegetative", "Tuber_Initiation", "Bulking", "Maturation"]
        if ndvi_value < 0.4:
            stage = "Vegetative"
        elif ndvi_value < 0.6:
            stage = "Tuber_Initiation"
        elif ndvi_value < 0.75:
            stage = "Bulking"
        else:
            stage = "Maturation"
        
        response = {
            "coordinates": {"latitude": lat, "longitude": lon},
            "date": date,
            "ndvi": round(float(ndvi_value), 3),
            "ndre": round(float(ndre_value), 3),
            "nitrogen_kg_per_ha": round(float(nitrogen_pred), 2),
            "growth_stage": stage,
            "nutrient_status": "low" if nitrogen_pred < 140 else ("optimal" if nitrogen_pred <= 180 else "high")
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Get zone-wise recommendations
    
    Request body:
    {
        "zones": [
            {
                "zone_id": 1,
                "stage": "Bulking",
                "nitrogen": 140,
                "ndvi": 0.58,
                "area_ha": 2.0
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'zones' not in data:
            return jsonify({"error": "Missing zones data"}), 400
        
        zones = data['zones']
        recommendations = recommendation_agent.generate_zone_recommendations(zones)
        output = recommendation_agent.save_recommendations(recommendations)
        
        return jsonify(output), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Get dashboard data"""
    try:
        dashboard_path = os.path.join("outputs", "dashboard_data.json")
        
        if os.path.exists(dashboard_path):
            with open(dashboard_path, 'r') as f:
                dashboard_data = json.load(f)
            return jsonify(dashboard_data), 200
        else:
            return jsonify({
                "status": "no_data",
                "message": "Dashboard data not available. Run pipeline first."
            }), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process', methods=['POST'])
def process_images():
    """Process images and generate all outputs"""
    try:
        # Process images
        results = processor.process_images()
        
        # Generate nutrient map
        nutrient_map = processor.create_nutrient_map(results['ndvi'])
        
        # Predictions would go here
        # (simplified for API endpoint)
        
        response = {
            "status": "success",
            "message": "Images processed successfully",
            "outputs": {
                "ndvi_mean": float(np.nanmean(results['ndvi'])),
                "ndre_mean": float(np.nanmean(results['ndre'])),
                "nutrient_map": "Nutrient_Map_Enhanced.png",
                "output_directory": processor.output_dir
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/zones', methods=['GET'])
def get_zones():
    """Get zone analysis from current processing"""
    try:
        # Load latest report
        outputs_dir = "outputs"
        report_files = [f for f in os.listdir(outputs_dir) if f.startswith("weekly_report_")]
        
        if not report_files:
            return jsonify({"error": "No reports available"}), 404
        
        latest_report = sorted(report_files)[-1]
        report_path = os.path.join(outputs_dir, latest_report)
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return jsonify(report), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    host = config['api']['host']
    port = config['api']['port']
    debug = config['api']['debug']
    
    print(f"Starting Flask API server on {host}:{port}")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict-gee - Predict from GPS coordinates")
    print("  POST /recommend - Get recommendations")
    print("  GET  /dashboard - Get dashboard data")
    print("  POST /process - Process images")
    print("  GET  /zones - Get zone analysis")
    
    app.run(host=host, port=port, debug=debug)

