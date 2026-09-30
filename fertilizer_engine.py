# -*- coding: utf-8 -*-
"""
CropSpectra - Indian Fertilizer Recommendation Engine
Calculates precise commercial fertilizer quantities, split schedules, organic alternatives,
and soil condition corrections based on ICAR standards, soil fertility, and land area.
"""

import json
import os
import math

FERTILIZER_FILE = os.path.join(os.path.dirname(__file__), 'data', 'fertilizer_india.json')
CROPS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'crops_india.json')

def load_fertilizer_database():
    if os.path.exists(FERTILIZER_FILE):
        with open(FERTILIZER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_crop_info(crop_id):
    if os.path.exists(CROPS_FILE):
        with open(CROPS_FILE, 'r', encoding='utf-8') as f:
            crops = json.load(f)
            for c in crops:
                if c['id'].lower() == crop_id.lower() or c['name'].lower() == crop_id.lower():
                    return c
    return None

def convert_area_to_hectares(area_value, unit):
    """Converts input area to hectares for agronomic calculation."""
    unit = (unit or 'acre').lower().strip()
    area = float(area_value) if area_value else 1.0
    
    if unit in ['hectare', 'ha', 'hectares']:
        return area, 'Hectare(s)'
    elif unit in ['bigha', 'bighas']:
        # Standard benchmark: 1 Hectare ≈ 4 Bighas (varies regionally, 0.25 ha/bigha standard)
        return area * 0.25, 'Bigha(s)'
    else:
        # Default: Acre (1 Hectare = 2.47105 Acres => 1 Acre = 0.404686 Hectares)
        return area * 0.404686, 'Acre(s)'

def parse_fertility_level(val, nutrient_type):
    """
    Parses numerical value or text rating (Low/Medium/High) into an estimated soil nutrient level.
    Standard Indian Soil Health Card Rating (kg/ha available):
    Nitrogen: Low < 280, Medium 280-560, High > 560
    Phosphorus (P2O5): Low < 23, Medium 23-56, High > 56
    Potassium (K2O): Low < 140, Medium 140-330, High > 330
    """
    if val is None or val == '':
        return None
    
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip().lower()
    if val_str.replace('.', '', 1).isdigit():
        return float(val_str)
    
    if nutrient_type == 'N':
        if 'low' in val_str: return 200.0
        elif 'high' in val_str: return 600.0
        else: return 350.0 # medium
    elif nutrient_type == 'P':
        if 'low' in val_str: return 15.0
        elif 'high' in val_str: return 65.0
        else: return 35.0
    elif nutrient_type == 'K':
        if 'low' in val_str: return 100.0
        elif 'high' in val_str: return 380.0
        else: return 220.0
    return None

def calculate_fertilizer_recommendation(inputs):
    """
    Calculates precise commercial fertilizer doses for given farm area and soil condition.
    
    Inputs expected:
    - crop_id: str (e.g. 'wheat', 'rice', 'cotton', etc.)
    - soil_n: float | str | None (kg/ha or 'low'/'medium'/'high')
    - soil_p: float | str | None
    - soil_k: float | str | None
    - soil_ph: float | None
    - soil_type: str | None
    - area: float (e.g. 2.0)
    - area_unit: 'acre' | 'hectare' | 'bigha'
    - growth_stage: str (optional)
    """
    fert_db = load_fertilizer_database()
    crop_id = (inputs.get('crop_id') or 'wheat').lower().strip()
    crop_info = load_crop_info(crop_id)
    
    crop_doses = fert_db.get('crop_doses', {})
    crop_dose_info = crop_doses.get(crop_id, crop_doses.get('default', {}))
    rdf_ha = crop_dose_info.get('rdf_ha', {'N': 100, 'P': 50, 'K': 50})
    
    # Area conversion
    raw_area = float(inputs.get('area', 1.0)) if inputs.get('area') else 1.0
    unit_str = inputs.get('area_unit', 'acre')
    area_ha, display_unit = convert_area_to_hectares(raw_area, unit_str)
    
    # Soil nutrient adjustments (Soil Health Card adjustment factor)
    # If soil is low, increase dose by 25%; if high, decrease by 25%
    s_n = parse_fertility_level(inputs.get('soil_n'), 'N')
    s_p = parse_fertility_level(inputs.get('soil_p'), 'P')
    s_k = parse_fertility_level(inputs.get('soil_k'), 'K')
    
    adj_factor_n = 1.0
    adj_factor_p = 1.0
    adj_factor_k = 1.0
    
    if s_n is not None:
        if s_n < 280: adj_factor_n = 1.25
        elif s_n > 560: adj_factor_n = 0.75
    if s_p is not None:
        if s_p < 23: adj_factor_p = 1.25
        elif s_p > 56: adj_factor_p = 0.75
    if s_k is not None:
        if s_k < 140: adj_factor_k = 1.25
        elif s_k > 330: adj_factor_k = 0.75
        
    net_n_ha = rdf_ha['N'] * adj_factor_n
    net_p_ha = rdf_ha['P'] * adj_factor_p
    net_k_ha = rdf_ha['K'] * adj_factor_k
    
    # Total nutrient requirement for the farm area (in kg)
    total_n_req = round(net_n_ha * area_ha, 1)
    total_p_req = round(net_p_ha * area_ha, 1)
    total_k_req = round(net_k_ha * area_ha, 1)
    
    # -------------------------------------------------------------
    # PLAN A: Standard Indian Combination (DAP + Urea + MOP)
    # DAP (18% N, 46% P2O5)
    # Urea (46% N)
    # MOP (60% K2O)
    # -------------------------------------------------------------
    dap_kg = (total_p_req / 0.46) if total_p_req > 0 else 0.0
    n_from_dap = dap_kg * 0.18
    remaining_n_for_urea = max(0.0, total_n_req - n_from_dap)
    urea_kg = (remaining_n_for_urea / 0.46) if remaining_n_for_urea > 0 else 0.0
    mop_kg = (total_k_req / 0.60) if total_k_req > 0 else 0.0
    
    plan_a = {
        'title': 'Plan A: DAP + Urea + MOP (Most Popular)',
        'dap_kg': round(dap_kg, 1),
        'dap_bags_50kg': round(dap_kg / 50.0, 1),
        'urea_kg': round(urea_kg, 1),
        'urea_bags_45kg': round(urea_kg / 45.0, 1),
        'mop_kg': round(mop_kg, 1),
        'mop_bags_50kg': round(mop_kg / 50.0, 1),
    }
    
    # -------------------------------------------------------------
    # PLAN B: Alternative Combination (SSP + Urea + MOP)
    # Excellent for oilseeds/pulses due to 11% Sulfur in SSP
    # SSP (16% P2O5, 11% S, 19% Ca)
    # -------------------------------------------------------------
    ssp_kg = (total_p_req / 0.16) if total_p_req > 0 else 0.0
    urea_plan_b_kg = (total_n_req / 0.46) if total_n_req > 0 else 0.0
    
    plan_b = {
        'title': 'Plan B: SSP + Urea + MOP (Best for Oilseeds & Sulfur Needs)',
        'ssp_kg': round(ssp_kg, 1),
        'ssp_bags_50kg': round(ssp_kg / 50.0, 1),
        'urea_kg': round(urea_plan_b_kg, 1),
        'urea_bags_45kg': round(urea_plan_b_kg / 45.0, 1),
        'mop_kg': round(mop_kg, 1),
        'mop_bags_50kg': round(mop_kg / 50.0, 1),
    }
    
    # -------------------------------------------------------------
    # Split Application Schedule
    # -------------------------------------------------------------
    raw_splits = crop_dose_info.get('splits', [])
    split_schedule = []
    
    for split in raw_splits:
        stage_name = split['stage']
        timing = split['timing']
        n_pct = split.get('n_pct', 0) / 100.0
        p_pct = split.get('p_pct', 0) / 100.0
        k_pct = split.get('k_pct', 0) / 100.0
        
        # Calculate for Plan A
        stage_dap = round(dap_kg * p_pct, 1)
        # Remaining N is distributed through urea
        stage_urea = round(urea_kg * n_pct, 1)
        stage_mop = round(mop_kg * k_pct, 1)
        
        applied_items = []
        if stage_dap > 0: applied_items.append(f"DAP: {stage_dap} kg")
        if stage_urea > 0: applied_items.append(f"Urea: {stage_urea} kg")
        if stage_mop > 0: applied_items.append(f"MOP: {stage_mop} kg")
        
        split_schedule.append({
            'stage': stage_name,
            'timing': timing,
            'dap_kg': stage_dap,
            'urea_kg': stage_urea,
            'mop_kg': stage_mop,
            'summary': ", ".join(applied_items) if applied_items else "No chemical fertilizer in this stage"
        })
        
    # -------------------------------------------------------------
    # Soil Amendments & Micronutrients (pH & Zinc)
    # -------------------------------------------------------------
    amendments = []
    ph_val = float(inputs.get('soil_ph')) if inputs.get('soil_ph') not in [None, ''] else None
    
    if ph_val is not None:
        if ph_val < 6.0:
            lime_amount = round(2.0 * area_ha * 1000, 0) # 2 tonnes/ha
            amendments.append({
                'type': 'Acidic Soil Correction (Lime)',
                'name': 'Agricultural Lime (CaCO3)',
                'quantity': f"{lime_amount} kg ({round(lime_amount/50.0, 0):.0f} bags of 50kg)",
                'advice': f"Your soil pH ({ph_val}) is acidic. Broadcast agricultural lime 2-3 weeks before sowing to neutralize acidity and unlock soil phosphorus."
            })
        elif ph_val > 8.0:
            gypsum_amount = round(2.5 * area_ha * 1000, 0) # 2.5 tonnes/ha
            amendments.append({
                'type': 'Alkaline / Sodic Soil Correction (Gypsum)',
                'name': 'Agricultural Gypsum',
                'quantity': f"{gypsum_amount} kg ({round(gypsum_amount/50.0, 0):.0f} bags of 50kg)",
                'advice': f"Your soil pH ({ph_val}) is alkaline/sodic. Apply gypsum followed by light flooding and leaching to replace excess sodium with calcium."
            })
            
    # Zinc requirement
    zn_amount = round(25.0 * area_ha, 1) # 25 kg/ha zinc sulphate
    amendments.append({
        'type': 'Micronutrient (Zinc Deficiency Prevention)',
        'name': 'Zinc Sulphate (21% Zn)',
        'quantity': f"{zn_amount} kg ({round(zn_amount/25.0, 1)} bags of 25kg)",
        'advice': "Apply Zinc Sulphate at final land preparation. IMPORTANT: Never mix Zinc Sulphate directly with DAP or SSP in the same application to prevent insoluble zinc phosphate precipitation."
    })
    
    # Organic Manure recommendation scaled to area
    organic_text = crop_dose_info.get('organic', 'Apply 8-10 tonnes/ha Farm Yard Manure (FYM) or 2.5 tonnes/ha Vermicompost.')
    fym_scaled = round(8.0 * area_ha, 1)
    vermi_scaled = round(2.5 * area_ha, 1)
    
    return {
        'crop_id': crop_id,
        'crop_name': crop_info['name'] if crop_info else crop_id.title(),
        'crop_hindi_name': crop_info.get('hindi_name', '') if crop_info else '',
        'area_entered': raw_area,
        'area_unit': display_unit,
        'area_ha': round(area_ha, 3),
        'rdf_ha': rdf_ha,
        'net_nutrients_req': {
            'N': total_n_req,
            'P': total_p_req,
            'K': total_k_req
        },
        'plan_a': plan_a,
        'plan_b': plan_b,
        'split_schedule': split_schedule,
        'amendments': amendments,
        'organic_advice': f"Apply approximately {fym_scaled} tonnes of well-rotted FYM (cow dung manure) or {vermi_scaled} tonnes of Vermicompost during summer ploughing.",
        'special_advisory': crop_dose_info.get('special_advisory', ''),
        'safety_tips': [
            "Never mix Zinc Sulphate directly with DAP / SSP (keep a 3-5 day gap between their applications).",
            "Always apply phosphatic (DAP/SSP) and potassic (MOP) fertilizers as basal placement 4-5 cm below seed depth.",
            "Avoid broadcasting urea immediately before heavy rainfall or on dry cracked soil to prevent nitrogen volatilization and runoff.",
            "Always maintain optimal soil moisture before applying urea top dressing."
        ],
        'disclaimer': fert_db.get('disclaimer', 'This is a general guideline based on ICAR agronomic standards. Please get a soil test from your nearest Krishi Vigyan Kendra (KVK) / Soil Health Card laboratory and follow your State Agriculture Department recommendations.')
    }
