"""
Agentic AI Module for Zone-wise Recommendations
Uses LangChain to generate stage-specific irrigation and fertilizer recommendations
"""

import os
import json
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

try:
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Using rule-based recommendations.")


class RecommendationAgent:
    """Agentic AI for generating zone-wise recommendations"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = self.config['paths']['output_dir']
        self.recommendations_path = os.path.join(self.output_dir, "recommendations.json")
        
        # Initialize LLM if available
        self.llm = None
        if LANGCHAIN_AVAILABLE:
            try:
                # Use OpenAI or other provider
                # For production, set OPENAI_API_KEY in environment
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
            except:
                print("LLM initialization failed, using rule-based approach")
    
    def analyze_zone(self, zone_id: int, stage: str, nitrogen: float, 
                    ndvi: float, area_ha: float = 1.0) -> Dict:
        """
        Analyze a single zone and generate recommendations
        
        Args:
            zone_id: Zone identifier
            stage: Growth stage name
            nitrogen: Current Nitrogen level (kg/ha)
            ndvi: NDVI value
            area_ha: Zone area in hectares
        
        Returns:
            Dictionary with recommendations
        """
        # Determine nitrogen status
        if nitrogen < 140:
            n_status = "low"
            n_recommendation = 180 - nitrogen  # Recommended addition
        elif nitrogen <= 180:
            n_status = "optimal"
            n_recommendation = 180 - nitrogen
        else:
            n_status = "high"
            n_recommendation = 0
        
        # Get irrigation recommendation based on stage
        irrigation_map = self.config['recommendations']['irrigation']
        irrigation = irrigation_map.get(stage.lower(), "30-35 mm/week")
        
        # Calculate costs (simplified)
        fertilizer_cost_per_kg = 50  # INR per kg
        irrigation_cost_per_mm = 100  # INR per mm per hectare
        
        total_fertilizer_cost = n_recommendation * fertilizer_cost_per_kg * area_ha
        irrigation_volume = float(irrigation.split('-')[0])  # Extract first value
        total_irrigation_cost = irrigation_volume * irrigation_cost_per_mm * area_ha
        
        total_cost = total_fertilizer_cost + total_irrigation_cost
        
        # Expected yield improvement (simplified estimation)
        if n_status == "low":
            yield_improvement = 0.15  # 15% improvement
        elif n_status == "optimal":
            yield_improvement = 0.05  # 5% improvement
        else:
            yield_improvement = 0.0
        
        recommendation = {
            "zone_id": zone_id,
            "growth_stage": stage,
            "current_nitrogen_kg_per_ha": round(nitrogen, 2),
            "ndvi": round(ndvi, 3),
            "nitrogen_status": n_status,
            "recommendations": {
                "nitrogen": {
                    "recommended_addition_kg_per_ha": round(max(0, n_recommendation), 2),
                    "target_kg_per_ha": 180,
                    "fertilizer_type": "Urea (46-0-0)",
                    "application_method": "Top-dressing"
                },
                "irrigation": {
                    "volume_mm_per_week": irrigation,
                    "frequency": "Weekly",
                    "method": "Drip irrigation recommended"
                },
                "potassium": {
                    "recommended_kg_per_ha": 60 if stage == "Bulking" else 40,
                    "fertilizer_type": "Muriate of Potash (0-0-60)"
                }
            },
            "cost_analysis": {
                "fertilizer_cost_inr": round(total_fertilizer_cost, 2),
                "irrigation_cost_inr": round(total_irrigation_cost, 2),
                "total_cost_inr": round(total_cost, 2),
                "cost_per_hectare": round(total_cost / area_ha, 2)
            },
            "expected_benefits": {
                "yield_improvement_percent": round(yield_improvement * 100, 1),
                "resource_optimization": "High" if n_status == "low" else "Medium",
                "soil_health_impact": "Positive"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Use LLM to enhance recommendation if available
        if self.llm:
            recommendation = self._enhance_with_llm(recommendation)
        
        return recommendation
    
    def _enhance_with_llm(self, recommendation: Dict) -> Dict:
        """Enhance recommendation using LLM"""
        if not self.llm:
            return recommendation
        
        prompt = PromptTemplate(
            input_variables=["stage", "nitrogen", "ndvi", "n_status"],
            template="""
            As an agricultural expert, provide a brief, actionable recommendation for potato farming.
            
            Growth Stage: {stage}
            Current Nitrogen: {nitrogen} kg/ha
            NDVI: {ndvi}
            Nitrogen Status: {n_status}
            
            Provide a 2-3 sentence recommendation focusing on:
            1. Optimal timing for intervention
            2. Expected impact on yield
            3. Any specific considerations for this growth stage
            
            Recommendation:
            """
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt)
            enhanced_text = chain.run(
                stage=recommendation["growth_stage"],
                nitrogen=recommendation["current_nitrogen_kg_per_ha"],
                ndvi=recommendation["ndvi"],
                n_status=recommendation["nitrogen_status"]
            )
            recommendation["ai_insight"] = enhanced_text.strip()
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
        
        return recommendation
    
    def generate_zone_recommendations(self, zones_data: List[Dict]) -> List[Dict]:
        """
        Generate recommendations for multiple zones
        
        Args:
            zones_data: List of zone dictionaries with stage, nitrogen, ndvi, area
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        for zone_data in zones_data:
            rec = self.analyze_zone(
                zone_id=zone_data.get("zone_id", 0),
                stage=zone_data.get("stage", "Vegetative"),
                nitrogen=zone_data.get("nitrogen", 150.0),
                ndvi=zone_data.get("ndvi", 0.5),
                area_ha=zone_data.get("area_ha", 1.0)
            )
            recommendations.append(rec)
        
        return recommendations
    
    def save_recommendations(self, recommendations: List[Dict]):
        """Save recommendations to JSON file"""
        output = {
            "generated_at": datetime.now().isoformat(),
            "total_zones": len(recommendations),
            "recommendations": recommendations,
            "summary": {
                "total_cost_inr": sum(r["cost_analysis"]["total_cost_inr"] for r in recommendations),
                "avg_yield_improvement": np.mean([r["expected_benefits"]["yield_improvement_percent"] 
                                                 for r in recommendations]),
                "zones_requiring_intervention": sum(1 for r in recommendations 
                                                   if r["nitrogen_status"] == "low")
            }
        }
        
        with open(self.recommendations_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Recommendations saved to {self.recommendations_path}")
        return output


if __name__ == "__main__":
    # Example usage
    agent = RecommendationAgent()
    
    # Sample zones
    zones = [
        {"zone_id": 1, "stage": "Vegetative", "nitrogen": 120, "ndvi": 0.45, "area_ha": 2.5},
        {"zone_id": 2, "stage": "Tuber_Initiation", "nitrogen": 150, "ndvi": 0.62, "area_ha": 3.0},
        {"zone_id": 3, "stage": "Bulking", "nitrogen": 140, "ndvi": 0.58, "area_ha": 2.0},
        {"zone_id": 4, "stage": "Maturation", "nitrogen": 175, "ndvi": 0.70, "area_ha": 1.5}
    ]
    
    recommendations = agent.generate_zone_recommendations(zones)
    output = agent.save_recommendations(recommendations)
    
    print("\nRecommendations Generated:")
    print(f"Total Zones: {output['total_zones']}")
    print(f"Total Cost: ₹{output['summary']['total_cost_inr']:.2f}")
    print(f"Avg Yield Improvement: {output['summary']['avg_yield_improvement']:.1f}%")

