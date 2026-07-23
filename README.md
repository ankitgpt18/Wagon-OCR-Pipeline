# Wagon-OCR-Pipeline

> **Multimodal VLM-Guided Parameter Extraction and Safety Compliance Auditing System for Indian Railways Freight Wagons**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?logo=ultralytics)](https://ultralytics.com)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-GPU-117D23)](https://github.com/PaddlePaddle/PaddleOCR)
[![Qwen2.5-VL](https://img.shields.io/badge/Qwen2.5--VL-Vision--Language-6B46C1)](https://github.com/QwenLM/Qwen2.5-VL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

Wagon-OCR-Pipeline is a production-grade Wagon OCR and Stencil audit pipeline designed for Indian Railways (CRIS). It localizes and extracts wagon details (11-digit wagon number, type, owner, zone, tare weight) from high-resolution raw images, performs cross-validation via S4 Mod-10 checksum validation and ROAMS master registries, and classifies maintenance stencil presence.

---

## ⚙️ System Architecture

1. **YOLOv11 Localization**: Localizes parameters across 5 classes (`wagon_number`, `wagon_type`, `owner_name`, `zone_markings`, `tare_load_details`).
2. **Multi-Tier OCR Ensemble**:
   - **Tier 1**: Qwen2.5-VL Vision-Language Model OCR on crops.
   - **Tier 2**: PaddleOCR on preprocessed crops (deskewed + CLAHE contrast boosted).
   - **Tier 3**: PaddleOCR on raw BGR bounding box crops.
   - **Tier 4**: Bounding Box Margin Iteration (+10% adaptive expansion).
   - **Tier 5**: Full Frame Scan fallback.
3. **Checksum & ROAMS Master Validation**:
   - Algorithmic S4 Mod-10 Checksum verification for 11-digit wagon numbers.
   - Combinatorial Levenshtein distance correction against 209 ROAMS Master Wagon Types and 186 Zonal/Owner Registries.
   - Physical boundary range checks ($10\text{--}45\text{T}$ tare, $40\text{--}100\text{T}$ CC) and missing decimal dot restoration.
4. **Maintenance Safety Stencil Classification**: Custom-trained YOLOv11 stencil presence model.

---

## 🚀 Getting Started

### 1. Pre-warm GPU OCR Microservice
```bash
python cris_ocr_pipeline/scripts/ocr_service.py
```

### 2. Start FastAPI Server
```bash
python start_server.py
```

### 3. Launch React Dashboard UI
```bash
cd frontend
npm install
npm run dev
```

Dashboard runs live at `http://localhost:5173`.
