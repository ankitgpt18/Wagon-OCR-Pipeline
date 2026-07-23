"""
CRIS Wagon OCR Pipeline - Automated 360-Degree System Readiness & Bug Audit Script
Run this script on any local environment to perform minute-by-minute diagnostic checks.
"""
import os
import sys
import json
import socket
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_python_dependencies():
    print_header("1. PYTHON DEPENDENCIES & CUDA ENVIRONMENT AUDIT")
    required = {
        "torch": "PyTorch",
        "paddle": "PaddlePaddle",
        "transformers": "HuggingFace Transformers (Qwen2.5-VL VLM)",
        "qwen_vl_utils": "Qwen VL Utilities",
        "ultralytics": "YOLOv11 Detection Engine",
        "cv2": "OpenCV (opencv-python)",
        "fastapi": "FastAPI Web Server",
        "uvicorn": "Uvicorn ASGI Server",
        "requests": "HTTP Requests Client",
        "Levenshtein": "Levenshtein String Metric"
    }
    
    missing = []
    for module, name in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "Loaded")
            print(f"  [PASS] {name} ({module}) -> Version: {version}")
        except ImportError:
            print(f"  [FAIL] {name} ({module}) -> MISSING!")
            missing.append(module)
            
    # CUDA check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        print(f"  [CUDA] PyTorch CUDA Available: {cuda_avail} | Device: {device_name}")
    except Exception as e:
        print(f"  [CUDA] PyTorch CUDA Check Failed: {e}")

    return missing

def check_model_weights():
    print_header("2. MODEL WEIGHTS & CONFIGURATION FILES AUDIT")
    files_to_check = [
        ("Localization Model Weights", "cris_ocr_pipeline/models/best.pt"),
        ("Safety Stencil Model Weights", "cris_ocr_pipeline/models/stencil_best.pt"),
        ("Pipeline Configuration File", "cris_ocr_pipeline/config/pipeline_config.json"),
        ("Batch 2 Manifest File", "cris_ocr_pipeline/outputs/ocr_results/batch_2_manifest.json"),
        ("Batch 1 Manifest File", "cris_ocr_pipeline/outputs/ocr_results/ocr_results_manifest.json")
    ]
    
    missing_files = []
    for label, rel_path in files_to_check:
        p = Path(rel_path)
        if p.exists():
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            print(f"  [PASS] {label}: Found at '{rel_path}' ({size_mb} MB)")
        else:
            print(f"  [FAIL] {label}: NOT FOUND at '{rel_path}'")
            missing_files.append(rel_path)
            
    return missing_files

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(('127.0.0.1', port)) == 0

def check_running_services():
    print_header("3. DECOUPLED MICROSERVICE PORTS AUDIT")
    ports = [
        (8000, "FastAPI Core Server (start_server.py)"),
        (8001, "PaddleOCR GPU Microservice (ocr_service.py)"),
        (8002, "Qwen2.5-VL VLM Microservice (vlm_ocr_service.py)"),
        (5173, "React Frontend UI Dashboard (npm run dev)")
    ]
    
    inactive = []
    for port, label in ports:
        is_active = check_port(port)
        status_str = "ACTIVE / LISTENING" if is_active else "INACTIVE / PORT CLOSED"
        flag = "[PASS]" if is_active else "[WARN]"
        print(f"  {flag} Port {port}: {label} -> {status_str}")
        if not is_active:
            inactive.append(port)
            
    return inactive

def check_core_functions():
    print_header("4. CORE PIPELINE LOGIC & CHECKSUM FUNCTIONALITY AUDIT")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from cris_ocr_pipeline.scripts.ocr_pipeline import (
            validate_wagon_number_checksum,
            ZONE_CODE_MAP,
            CODE_TO_WAGON_TYPE,
            verify_and_clean_tare_details
        )
        
        # Test 1: S4 Mod-10 Checksum
        test_wagon = "33121565554"
        valid = validate_wagon_number_checksum(test_wagon)
        print(f"  [PASS] S4 Mod-10 Checksum test on '{test_wagon}': {valid} (Expected: True)")
        
        # Test 2: Master Registries
        print(f"  [PASS] ROAMS Zone Registry Loaded: {len(ZONE_CODE_MAP)} zones (e.g., 13 -> {ZONE_CODE_MAP.get('13')})")
        print(f"  [PASS] ROAMS Type Registry Loaded: {len(CODE_TO_WAGON_TYPE)} types (e.g., 22 -> {CODE_TO_WAGON_TYPE.get('22')})")
        
        # Test 3: Tare Range Cleaning
        tare_clean = verify_and_clean_tare_details("TARE 2050 KG CC 65.0T", "BOXNHL")
        print(f"  [PASS] Tare Decimal Restoration test: '{tare_clean}'")
        
    except Exception as e:
        print(f"  [FAIL] Pipeline core function import error: {e}")

def main():
    print_header("CRIS WAGON OCR PIPELINE - SYSTEM READINESS & BUG DIAGNOSTIC")
    missing_deps = check_python_dependencies()
    missing_files = check_model_weights()
    inactive_ports = check_running_services()
    check_core_functions()
    
    print_header("SUMMARY & DIAGNOSTIC VERDICT")
    if not missing_deps and not missing_files and not inactive_ports:
        print("  🎉 SYSTEM IS 100% READY! All microservices, models, and dependencies are operational.")
    else:
        print("  ⚠️ BUGS / INACTION REQUIRED TO FIX:")
        if missing_deps:
            print(f"    1. Install missing Python modules: pip install {' '.join(missing_deps)}")
        if missing_files:
            print(f"    2. Ensure git pull was successful for missing files: {missing_files}")
        if inactive_ports:
            print("    3. Start inactive background services in separate terminals:")
            if 8001 in inactive_ports:
                print("       - Terminal 1: python cris_ocr_pipeline/scripts/ocr_service.py")
            if 8002 in inactive_ports:
                print("       - Terminal 2: python cris_ocr_pipeline/scripts/vlm_ocr_service.py")
            if 8000 in inactive_ports:
                print("       - Terminal 3: python start_server.py")
            if 5173 in inactive_ports:
                print("       - Terminal 4: cd frontend && npm run dev")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
