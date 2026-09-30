# -*- coding: utf-8 -*-
import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import crop_engine
import fertilizer_engine

def run_tests():
    # Test 1: Punjab + Rabi + Alluvial -> Wheat expected top
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
    res1 = crop_engine.recommend_crops(test_punjab)
    print("=== TEST 1 (Punjab Rabi) ===")
    for r in res1:
        print(f"  {r['name']} ({r['hindi_name']}): {r['suitability']}%")

    # Test 2: Maharashtra + Kharif + Black soil -> Cotton / Soybean expected top
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
    res2 = crop_engine.recommend_crops(test_mh)
    print("\n=== TEST 2 (Maharashtra Kharif) ===")
    for r in res2:
        print(f"  {r['name']} ({r['hindi_name']}): {r['suitability']}%")

    # Test 3: West Bengal + Kharif + Clay/Alluvial -> Rice / Jute expected top
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
    res3 = crop_engine.recommend_crops(test_wb)
    print("\n=== TEST 3 (West Bengal Kharif) ===")
    for r in res3:
        print(f"  {r['name']} ({r['hindi_name']}): {r['suitability']}%")

    # Test 4: Fertilizer calculation for 2 Acres of Wheat
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
    res4 = fertilizer_engine.calculate_fertilizer_recommendation(test_fert)
    print("\n=== TEST 4 (Fertilizer Plan for 2 Acres Wheat) ===")
    print("Crop:", res4['crop_name'])
    print("Area in Ha:", res4['area_ha'])
    print("Plan A:", res4['plan_a'])
    print("Plan B:", res4['plan_b'])
    print("Split Schedule Stages:", len(res4['split_schedule']))

if __name__ == '__main__':
    run_tests()
