# -*- coding: utf-8 -*-
"""
CropSpectra - Indian Crop Recommendation Engine
Rule-based agronomic scoring engine based on ICAR standards and regional suitability.
Evaluates Season, Soil Type, Climate (Temperature, Rainfall, Humidity), Soil pH,
NPK fertility, State/Region, and Irrigation availability.
"""

import json
import os
from datetime import datetime

CROPS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'crops_india.json')

def load_crops_database():
    """Loads all crops from data/crops_india.json."""
    if os.path.exists(CROPS_FILE):
        with open(CROPS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def get_current_season_suggestion():
    """Auto-suggest Indian agricultural season based on current calendar month."""
    month = datetime.now().month
    # June to October: Kharif (Monsoon)
    # November to March: Rabi (Winter)
    # April to May: Zaid (Summer)
    if month in [6, 7, 8, 9, 10]:
        return 'Kharif'
    elif month in [11, 12, 1, 2, 3]:
        return 'Rabi'
    else:
        return 'Zaid'

def score_range(val, r_min, r_max, ideal_min, ideal_max, weight):
    """
    Scores a continuous numerical variable (Temperature, Rainfall, pH, Humidity)
    Returns: (score, status: 'ideal' | 'tolerable' | 'borderline' | 'unfavorable')
    """
    if val is None:
        return (weight * 0.75, 'unspecified')
    
    val = float(val)
    if ideal_min <= val <= ideal_max:
        return (weight * 1.0, 'ideal')
    elif r_min <= val <= r_max:
        if val < ideal_min:
            dist = (val - r_min) / max(ideal_min - r_min, 1e-5)
        else:
            dist = (r_max - val) / max(r_max - ideal_max, 1e-5)
        factor = 0.5 + 0.45 * max(0.0, min(1.0, dist))
        return (weight * factor, 'tolerable')
    else:
        diff = min(abs(val - r_min), abs(val - r_max))
        tolerance_buffer = (r_max - r_min) * 0.25
        if diff <= tolerance_buffer:
            penalty = 0.25 * (1.0 - diff / max(tolerance_buffer, 1e-5))
            return (weight * max(0.05, penalty), 'borderline')
        return (0.0, 'unfavorable')

def recommend_crops(inputs):
    """
    Computes suitability score (0-100%) for all crops and returns the Top 5 matches.
    
    Inputs expected:
    - state: str
    - district: str (optional)
    - season: 'Kharif' | 'Rabi' | 'Zaid'
    - soil_type: str ('Alluvial', 'Black (Regur)', 'Red', 'Laterite', 'Sandy', 'Loamy', 'Clay')
    - temperature: float (°C)
    - humidity: float (%)
    - rainfall: float (mm)
    - ph: float
    - nitrogen: float | None (kg/ha)
    - phosphorus: float | None (kg/ha)
    - potassium: float | None (kg/ha)
    - irrigation: 'Yes' | 'No' | bool
    """
    crops_db = load_crops_database()
    if not crops_db:
        return []

    user_state = (inputs.get('state') or '').strip().title()
    user_season = (inputs.get('season') or '').strip().title()
    user_soil = (inputs.get('soil_type') or '').strip()
    
    temp = float(inputs['temperature']) if inputs.get('temperature') not in [None, ''] else None
    hum = float(inputs['humidity']) if inputs.get('humidity') not in [None, ''] else None
    rain = float(inputs['rainfall']) if inputs.get('rainfall') not in [None, ''] else None
    ph = float(inputs['ph']) if inputs.get('ph') not in [None, ''] else None
    
    n_val = float(inputs['nitrogen']) if inputs.get('nitrogen') not in [None, ''] else None
    p_val = float(inputs['phosphorus']) if inputs.get('phosphorus') not in [None, ''] else None
    k_val = float(inputs['potassium']) if inputs.get('potassium') not in [None, ''] else None
    
    has_irrigation = inputs.get('irrigation') in ['Yes', 'yes', True, 'true', 1, '1']

    results = []

    for crop in crops_db:
        score = 0.0
        reasons_matched = []
        warnings = []

        # 1. Season Match (Weight: 20%)
        crop_seasons = [s.strip().title() for s in crop.get('seasons', [])]
        if user_season in crop_seasons:
            score += 20.0
            reasons_matched.append(f"Optimal for {user_season} season")
        else:
            score += 2.0
            seasons_str = ", ".join(crop_seasons)
            warnings.append(f"Not standard for {user_season} season (usually grown in {seasons_str})")

        # 2. Soil Type Match (Weight: 20%)
        crop_soils = [s.lower() for s in crop.get('soil_types', [])]
        user_soil_clean = user_soil.lower()
        if any(user_soil_clean in s or s in user_soil_clean for s in crop_soils):
            score += 20.0
            reasons_matched.append(f"Well-suited to {user_soil} soil")
        else:
            score += 5.0
            soils_str = ", ".join(crop.get('soil_types', []))
            warnings.append(f"{user_soil} soil is suboptimal (prefers {soils_str})")

        # 3. Temperature Match (Weight: 15%)
        t_score, t_status = score_range(
            temp,
            crop.get('temp_min', 10),
            crop.get('temp_max', 40),
            crop.get('temp_ideal_min', 18),
            crop.get('temp_ideal_max', 30),
            15.0
        )
        score += t_score
        if t_status == 'ideal':
            reasons_matched.append(f"Temperature ({temp}°C) is in ideal range ({crop['temp_ideal_min']}-{crop['temp_ideal_max']}°C)")
        elif t_status == 'tolerable':
            reasons_matched.append(f"Temperature ({temp}°C) is tolerable ({crop['temp_min']}-{crop['temp_max']}°C)")
        elif t_status in ['borderline', 'unfavorable'] and temp is not None:
            warnings.append(f"Temperature ({temp}°C) is outside optimum range ({crop['temp_ideal_min']}-{crop['temp_ideal_max']}°C)")

        # 4. Rainfall / Water Match (Weight: 15%)
        r_score, r_status = score_range(
            rain,
            crop.get('rainfall_min', 100),
            crop.get('rainfall_max', 2000),
            crop.get('rainfall_min', 100) * 1.1,
            crop.get('rainfall_max', 2000) * 0.9,
            15.0
        )
        if rain is not None:
            if rain < crop.get('rainfall_min', 100):
                if has_irrigation:
                    r_score = max(r_score, 12.0)
                    reasons_matched.append("Rainfall is low, but available irrigation compensates")
                else:
                    r_score = max(0.0, r_score - 8.0)
                    warnings.append(f"Water deficit: requires {crop['rainfall_min']}mm+ rainfall or assured irrigation")
            else:
                score += r_score
                if r_status in ['ideal', 'tolerable']:
                    reasons_matched.append(f"Rainfall ({rain}mm) satisfies crop water needs")
        else:
            score += 10.0

        # 5. Soil pH Match (Weight: 10%)
        ph_score, ph_status = score_range(
            ph,
            crop.get('ph_min', 5.5),
            crop.get('ph_max', 8.0),
            crop.get('ph_ideal_min', 6.0),
            crop.get('ph_ideal_max', 7.5),
            10.0
        )
        score += ph_score
        if ph_status == 'ideal':
            reasons_matched.append(f"Soil pH ({ph}) is ideal ({crop['ph_ideal_min']}-{crop['ph_ideal_max']})")
        elif ph_status == 'tolerable':
            reasons_matched.append(f"Soil pH ({ph}) is acceptable")
        elif ph_status in ['borderline', 'unfavorable'] and ph is not None:
            warnings.append(f"Soil pH ({ph}) requires correction (ideal: {crop['ph_ideal_min']}-{crop['ph_ideal_max']})")

        # 6. Humidity / Moisture (Weight: 10%)
        h_score, h_status = score_range(
            hum,
            crop.get('humidity_min', 30),
            crop.get('humidity_max', 95),
            crop.get('humidity_min', 30) * 1.15,
            crop.get('humidity_max', 95) * 0.90,
            10.0
        )
        score += h_score
        if h_status in ['ideal', 'tolerable'] and hum is not None:
            reasons_matched.append(f"Relative humidity ({hum}%) is favorable")

        # 7. NPK & State Compatibility Bonus (Weight: 10%)
        npk_score = 0.0
        crop_npk = crop.get('npk_ideal', {'N': 80, 'P': 40, 'K': 40})
        if n_val is not None and p_val is not None and k_val is not None:
            n_diff = abs(n_val - crop_npk['N']) / max(crop_npk['N'], 1)
            p_diff = abs(p_val - crop_npk['P']) / max(crop_npk['P'], 1)
            k_diff = abs(k_val - crop_npk['K']) / max(crop_npk['K'], 1)
            avg_diff = (n_diff + p_diff + k_diff) / 3.0
            npk_score = max(0.0, 6.0 * (1.0 - min(1.0, avg_diff)))
            if avg_diff < 0.35:
                reasons_matched.append("Existing soil N-P-K levels align well with crop nutrient demands")
        else:
            npk_score = 4.5
        score += npk_score

        # Regional State Bonus (4%)
        crop_states = [s.lower() for s in crop.get('states', [])]
        if user_state and any(user_state.lower() in s or s in user_state.lower() for s in crop_states):
            score += 4.0
            reasons_matched.append(f"Widely cultivated and highly productive across {user_state}")

        final_percentage = round(min(100.0, max(5.0, score)), 1)

        results.append({
            'crop_id': crop['id'],
            'name': crop['name'],
            'hindi_name': crop['hindi_name'],
            'category': crop.get('category', 'Field Crop'),
            'suitability': final_percentage,
            'duration_days': crop.get('duration_days', 'N/A'),
            'water_requirement': crop.get('water_requirement', 'Moderate'),
            'npk_ideal': crop.get('npk_ideal', {}),
            'description': crop.get('description', ''),
            'tips': crop.get('tips', ''),
            'reasons': reasons_matched[:4],
            'warnings': warnings[:2],
            'source': crop.get('agronomic_source', 'ICAR Agronomic Guidelines')
        })

    results.sort(key=lambda x: x['suitability'], reverse=True)
    return results[:5]
