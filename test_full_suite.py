# -*- coding: utf-8 -*-
import sys
import os
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app import app, init_db, DB_NAME
import crop_engine
import fertilizer_engine

def run_suite():
    print("==================================================")
    print("CROPSPECTRA 2026 - FULL TEST SUITE VERIFICATION")
    print("==================================================")

    # 1. Agronomic Crop Engine Tests
    print("\n[1] Testing Crop Recommendation Engine...")
    test_punjab = {
        'state': 'Punjab',
        'district': 'Ludhiana',
        'season': 'Rabi',
        'soil_type': 'Alluvial',
        'temperature': 18,
        'humidity': 60,
        'rainfall': 80,
        'ph': 7.0,
        'nitrogen': 120,
        'phosphorus': 60,
        'potassium': 40,
        'irrigation': 'Yes'
    }
    res_punjab = crop_engine.recommend_crops(test_punjab)
    assert len(res_punjab) == 5, "Expected Top 5 crops"
    assert res_punjab[0]['name'] == 'Wheat', f"Expected Wheat top for Punjab Rabi, got {res_punjab[0]['name']}"
    print(f"  ✓ Punjab Rabi Case: Top match is {res_punjab[0]['name']} ({res_punjab[0]['suitability']}%)")

    test_mh = {
        'state': 'Maharashtra',
        'district': 'Nagpur',
        'season': 'Kharif',
        'soil_type': 'Black (Regur)',
        'temperature': 28,
        'humidity': 75,
        'rainfall': 800,
        'ph': 7.5,
        'nitrogen': 100,
        'phosphorus': 50,
        'potassium': 50,
        'irrigation': 'Yes'
    }
    res_mh = crop_engine.recommend_crops(test_mh)
    assert len(res_mh) == 5
    top_names = [c['name'] for c in res_mh[:2]]
    assert 'Cotton' in top_names or 'Soybean' in top_names or 'Maize / Corn' in top_names
    print(f"  ✓ Maharashtra Kharif Case: Top matches are {top_names}")

    test_wb = {
        'state': 'West Bengal',
        'district': 'Burdwan',
        'season': 'Kharif',
        'soil_type': 'Clay',
        'temperature': 30,
        'humidity': 85,
        'rainfall': 1400,
        'ph': 6.2,
        'nitrogen': 100,
        'phosphorus': 50,
        'potassium': 50,
        'irrigation': 'Yes'
    }
    res_wb = crop_engine.recommend_crops(test_wb)
    assert len(res_wb) == 5
    assert 'Rice' in res_wb[0]['name'] or 'Jute' in res_wb[0]['name']
    print(f"  ✓ West Bengal Kharif Case: Top match is {res_wb[0]['name']} ({res_wb[0]['suitability']}%)")

    # 2. Fertilizer Engine Tests
    print("\n[2] Testing Fertilizer Recommendation Engine...")
    test_fert = {
        'crop_id': 'wheat',
        'soil_n': 'medium',
        'soil_p': 'medium',
        'soil_k': 'medium',
        'soil_ph': 7.2,
        'soil_type': 'Alluvial',
        'area': 2.0,
        'area_unit': 'acre'
    }
    res_fert = fertilizer_engine.calculate_fertilizer_recommendation(test_fert)
    assert res_fert['area_ha'] > 0
    assert 'plan_a' in res_fert and res_fert['plan_a']['dap_kg'] > 0
    assert len(res_fert['split_schedule']) >= 2
    print(f"  ✓ 2 Acres Wheat Fertilizer: DAP {res_fert['plan_a']['dap_kg']} kg, Urea {res_fert['plan_a']['urea_kg']} kg, MOP {res_fert['plan_a']['mop_kg']} kg")

    # 3. Flask Client Route Tests
    print("\n[3] Testing Flask Routes with Test Client...")
    client = app.test_client()

    # Unauthorized access redirects to login
    r_unauth = client.get('/crop-recommendation')
    assert r_unauth.status_code == 302 and '/login' in r_unauth.headers['Location']
    print("  ✓ Unauthenticated access correctly redirects to /login")

    # Authenticated Session
    with client.session_transaction() as sess:
        sess['user'] = 'testfarmer'

    # GET /crop-recommendation
    r_crop_get = client.get('/crop-recommendation')
    assert r_crop_get.status_code == 200
    assert b'Crop Recommendation Advisor' in r_crop_get.data
    print("  ✓ GET /crop-recommendation returned 200 OK")

    # POST /crop-recommendation
    r_crop_post = client.post('/crop-recommendation', data={
        'state': 'Punjab',
        'district': 'Ludhiana',
        'season': 'Rabi',
        'soil_type': 'Alluvial',
        'temperature': '18.5',
        'humidity': '60',
        'rainfall': '90',
        'ph': '6.8',
        'nitrogen': '120',
        'phosphorus': '60',
        'potassium': '40',
        'irrigation': 'Yes'
    })
    assert r_crop_post.status_code == 200
    assert b'Top Recommended Crops' in r_crop_post.data
    print("  ✓ POST /crop-recommendation computed recommendations and rendered 200 OK")

    # GET /fertilizer-recommendation
    r_fert_get = client.get('/fertilizer-recommendation?crop=wheat')
    assert r_fert_get.status_code == 200
    assert b'Fertilizer Dosage' in r_fert_get.data
    print("  ✓ GET /fertilizer-recommendation returned 200 OK")

    # POST /fertilizer-recommendation
    r_fert_post = client.post('/fertilizer-recommendation', data={
        'crop_id': 'wheat',
        'area': '2.5',
        'area_unit': 'acre',
        'soil_n': 'medium',
        'soil_p': 'medium',
        'soil_k': 'medium',
        'soil_ph': '6.8',
        'soil_type': 'Alluvial',
        'growth_stage': 'Sowing / Planting (Basal)'
    })
    assert r_fert_post.status_code == 200
    assert b'Fertilizer Schedule for Wheat' in r_fert_post.data
    print("  ✓ POST /fertilizer-recommendation computed dosage and rendered 200 OK")

    # GET /history
    r_hist = client.get('/history')
    assert r_hist.status_code == 200
    assert b'Recommendation Consultation History' in r_hist.data
    assert b'Wheat' in r_hist.data
    print("  ✓ GET /history returned 200 OK with logged consultation records")

    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (100% HEALTHY) ✨")
    print("==================================================")

if __name__ == '__main__':
    run_suite()
