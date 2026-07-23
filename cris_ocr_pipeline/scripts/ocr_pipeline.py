#!/usr/bin/env python3
"""
Wagon OCR Inference Pipeline (Production-Grade)
Combines:
1. Local YOLOv8 model (GPU/CUDA) for text region detection
2. OpenCV-based image pre-processing (contrast normalization, adaptive resize)
3. PaddleOCR for high-accuracy text recognition
4. Combinatorial OCR error correction & 11-digit checksum validation
5. Owning Railway (Zone) cross-verification
"""
import os
from pathlib import Path
# CRITICAL: Pre-load PyTorch C++ DLLs on Windows using ctypes to prevent WinError 127 shm.dll loader error
if os.name == "nt":
    import ctypes
    import site
    t_lib = r"C:\Users\ankit\AppData\Roaming\Python\Python311\site-packages\torch\lib"
    if os.path.exists(t_lib):
        try:
            os.add_dll_directory(t_lib)
            for dll_name in ["c10.dll", "torch_cpu.dll", "shm.dll"]:
                dll_p = os.path.join(t_lib, dll_name)
                if os.path.exists(dll_p):
                    try:
                        ctypes.CDLL(dll_p)
                    except Exception:
                        pass
        except Exception:
            pass
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        torch_lib = Path(sp) / "torch" / "lib"
        if torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass
        nvidia_path = Path(sp) / "nvidia"
        if nvidia_path.exists():
            for p in nvidia_path.rglob("bin"):
                try:
                    os.add_dll_directory(str(p))
                except Exception:
                    pass

import argparse
import csv
import glob
import json
import re
import time

# Global cache for YOLO models to avoid reloading from disk on each image prediction
YOLO_MODEL_CACHE = {}
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from paddleocr import PaddleOCR
from PIL import Image

# ── Standards & Lookups ──────────────────────────────────────────────
ZONE_CODE_MAP = {
    "01": "CR",
    "02": "ER",
    "03": "NR",
    "04": "NER",
    "05": "NFR",
    "06": "SR",
    "07": "SER",
    "08": "WR",
    "09": "SCR",
    "10": "ECR",
    "11": "NWR",
    "12": "ECoR",
    "13": "NCR",
    "14": "SECR",
    "15": "SWR",
    "16": "WCR",
    "24": "CONCOR",
    "25": "PRCL",
    "26": "RSPL",
    "35": "SCAL",
    "37": "ARIL",
    "40": "GIPL",
}

CODE_TO_WAGON_TYPE = {
    "10": "BOXN", "11": "BOXNHA", "12": "BOXNHS", "13": "BOXNCR", "14": "BOXNLW", "15": "BOXNB", "16": "BOXNF", "17": "BOXNG", "18": "BOY", "19": "BOST", "20": "BOXNAL", "21": "BOSTHS", "22": "BOXNHL",
    "30": "BCNA", "31": "BCNAHS", "32": "BCCNR", "33": "BCNHL",
    "40": "BTPN", "41": "BTPNHS", "42": "BTPGLN", "43": "BTALN", "44": "BTCS", "45": "BTPH", "46": "BTAP", "47": "BTFLN",
    "55": "BRNA", "56": "BRNAHS", "57": "BFNS", "58": "BOMN", "59": "BRSTH", "60": "BFAT", "61": "BLCA", "62": "BLCB", "63": "BLLA", "64": "BLLB", "65": "BRS", "66": "BFU", "67": "BRHNEHS", "68": "BCL", "69": "BCLA",
    "70": "BOBYN", "71": "BOBYNHS", "72": "BOBRN", "73": "BOBRNHS", "74": "BOBRAL",
    "80": "BWTB", "81": "MBWT", "82": "DBKM", "83": "MBWC",
    "85": "BVZC", "86": "BVZI"
}


WAGON_TYPES_LIST = []
OWNER_MAP = {}
VALID_OWNER_NAMES = set()
VALID_WAGON_TYPES = set()

try:
    root_dir = Path(__file__).resolve().parent.parent
    owner_files = glob.glob(str(root_dir / "config" / "roams_owner_master_*.csv")) or glob.glob("cris_ocr_pipeline/config/roams_owner_master_*.csv")
    type_files = glob.glob(str(root_dir / "config" / "roams_wagon_type_master_*.csv")) or glob.glob("cris_ocr_pipeline/config/roams_wagon_type_master_*.csv")
    
    if owner_files:
        with open(owner_files[0], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
                oid = cleaned_row.get("owner_id", "").upper()
                oname = cleaned_row.get("owner_name", "")
                if oid:
                    OWNER_MAP[oid] = oname
                    VALID_OWNER_NAMES.add(oid)
                    
    if type_files:
        with open(type_files[0], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
                wtype = cleaned_row.get("wagon_type", "").upper()
                if wtype:
                    WAGON_TYPES_LIST.append(wtype)
                    VALID_WAGON_TYPES.add(wtype)
                    
    # Always include standard suffix variants (2D, 25T, etc.)
    VALID_WAGON_TYPES.update({"BOXNHL2D", "BOXNHL25T", "BOXNHL25T2D", "BOXN25T2D", "BOXN25T", "BOXNCR", "BOXNHSM1", "BFNSM1", "BCNM", "BCNM1", "BCNM2"})
except Exception as e:
    print(f"Error loading master CSV files dynamically: {e}")

if not VALID_WAGON_TYPES:
    VALID_WAGON_TYPES = {
        "BOXN", "BOXNHS", "BOXNHL", "BOXNHL2D", "BOXNHL25T", "BOXN25T", "BOXNCR", "BOXNHSM1", "BOXNAL", "BOXNC", "BOXNF", "BOXNG", 
        "BOXNEL", "BOXNLW", "BOXNM1", "BOXNUTR", "BOY", "BOST", "BOSTHS",
        "BCN", "BCNA", "BCNAHS", "BCCN", "BCCNR", "BCNHL", "BCBFG",
        "BTPN", "BTPNHS", "BTPGLN", "BTFLN", "BTALAN", "BTCS", "BTPH", "BTAP",
        "BOBRN", "BOBRNHS", "BOBYN", "BOBYNHS", "BOBRAL", "BRN", "BRNA", "BRNAHS", 
        "BFNS", "BFNSM1", "BFAT", "BRSTH", "BLCA", "BLCB", "BLLA", "BLLB", "BLC", "BLSS", 
        "BFU", "BWTB", "MBWT", "MBWZ", "DBKM", "BFR", "BOMN", "BVZC", "BVZI", "WAGON"
    }
    WAGON_TYPES_LIST = list(VALID_WAGON_TYPES)

if not VALID_OWNER_NAMES:
    VALID_OWNER_NAMES = {
        "CONCOR", "IOCL", "NTPC", "IPL", "BALCO", "NALCO", "SAIL", "TATA", 
        "IFFCO", "KRIBHCO", "PRCL", "CR", "ER", "NR", "NER", "NFR", "NF", "SR", 
        "SER", "WR", "SCR", "ECR", "NWR", "ECoR", "ECOR", "NCR", "SECR", "SWR", "WCR",
        "SCAL", "RSPL", "ARIL", "GIPL", "MBS"
    }
    for item in VALID_OWNER_NAMES:
        OWNER_MAP[item] = item
        OWNER_MAP[item.upper()] = item


# OCR Confusion candidates mapping
CONFUSION_MAP = {
    "O": ["0", "8"],
    "o": ["0"],
    "D": ["0"],
    "I": ["1"],
    "l": ["1"],
    "i": ["1"],
    "|": ["1"],
    "T": ["1"],
    "Z": ["2"],
    "z": ["2"],
    "S": ["5"],
    "s": ["5"],
    "B": ["8", "0"],
    "G": ["6", "0"],
    "A": ["4"],
}


# ── Algorithms & Helpers ────────────────────────────────────────────
def validate_wagon_number_checksum(num_str):
    """
    Validate 11-digit Indian Railway wagon number using standard checksum.
    S4 = (EvenPosSum * 3) + OddPosSum
    CheckDigit = (10 - (S4 % 10)) % 10
    """
    if not num_str or len(num_str) != 11 or not num_str.isdigit():
        return False

    digits = [int(d) for d in num_str]

    # Even positions (2nd, 4th, 6th, 8th, 10th - 1-based index)
    # Python 0-based indices: 1, 3, 5, 7, 9
    s_even = digits[1] + digits[3] + digits[5] + digits[7] + digits[9]

    # Odd positions (1st, 3rd, 5th, 7th, 9th)
    # Python 0-based indices: 0, 2, 4, 6, 8
    s_odd = digits[0] + digits[2] + digits[4] + digits[6] + digits[8]

    s4 = (s_even * 3) + s_odd
    expected_check_digit = (10 - (s4 % 10)) % 10

    return expected_check_digit == digits[10]


def correct_wagon_number_combinatorial(raw_text):
    """
    Tries all combinations of OCR-confused digits and characters to find a valid wagon number.
    Supports 10-digit and 9-digit reconstruction if digits were missed.
    Returns: (corrected_str, method) where method is 'checksum', 'reconstructed', or 'best_guess'
    """
    if not raw_text:
        return "", "fallback"

    # Fast Path 1: Check if raw_text contains an 11-digit sequence that is ALREADY valid
    raw_digits_only = re.sub(r"\D", "", raw_text)
    for i in range(len(raw_digits_only) - 10):
        candidate = raw_digits_only[i : i + 11]
        if validate_wagon_number_checksum(candidate):
            return candidate, "exact"

    # Filter string to only keep digits and characters in CONFUSION_MAP
    allowed_chars = set("0123456789") | set(CONFUSION_MAP.keys())
    cleaned = "".join([c for c in raw_text if c in allowed_chars])

    # Find sequences of length around 11
    candidates_substrings = []
    for i in range(len(cleaned) - 10):
        candidates_substrings.append(cleaned[i : i + 11])

    # Common digit-to-digit OCR confusions
    DIGIT_CONFUSION = {
        "0": ["0", "8", "9", "6"],
        "1": ["1", "7"],
        "2": ["2", "7"],
        "3": ["3", "8", "9"],
        "5": ["5", "6", "0"],
        "6": ["6", "8", "5", "0"],
        "7": ["7", "1", "2"],
        "8": ["8", "0", "3", "6"],
        "9": ["9", "0", "8"],
    }

    # 1. Try 11-digit window combinations
    if candidates_substrings:
        for sub in candidates_substrings:
            choices = []
            for char in sub:
                pos_choices = []
                if char in CONFUSION_MAP:
                    pos_choices.extend(CONFUSION_MAP[char])
                elif char in DIGIT_CONFUSION:
                    pos_choices.extend(DIGIT_CONFUSION[char])
                else:
                    pos_choices.append(char)
                choices.append(list(dict.fromkeys(pos_choices)))

            valid_candidates = []
            for combo in product(*choices):
                candidate = "".join(combo)
                if len(candidate) == 11 and validate_wagon_number_checksum(candidate):
                    dist = sum(
                        1
                        for c1, c2 in zip(sub, candidate)
                        if c1 != c2 and not (c1 in CONFUSION_MAP and c2 in CONFUSION_MAP[c1])
                    )
                    valid_candidates.append((candidate, dist))

            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[1])
                return valid_candidates[0][0], "checksum"

    # 2. If no 11-digit match, but we have a 10-digit number, try 10-digit mathematical reconstruction
    digits_only = re.sub(r"\D", "", cleaned)
    
    candidates_10 = []
    for i in range(len(digits_only) - 9):
        candidates_10.append(digits_only[i : i + 10])
        
    if candidates_10:
        for sub_10 in candidates_10:
            reconstructed_candidates = []
            for pos in range(11):
                for digit in "0123456789":
                    candidate = sub_10[:pos] + digit + sub_10[pos:]
                    if len(candidate) == 11:
                        if candidate[0:2] in CODE_TO_WAGON_TYPE and candidate[2:4] in ZONE_CODE_MAP:
                            if validate_wagon_number_checksum(candidate):
                                reconstructed_candidates.append(candidate)
                        
            if reconstructed_candidates:
                return reconstructed_candidates[0], "reconstructed"

    # 3. If still no match, try 9-digit mathematical reconstruction (2 missing digits)
    candidates_9 = []
    for i in range(len(digits_only) - 8):
        candidates_9.append(digits_only[i : i + 9])
        
    if candidates_9:
        for sub_9 in candidates_9:
            reconstructed_candidates = []
            for pos1 in range(10):
                for pos2 in range(pos1 + 1, 11):
                    for d1 in "0123456789":
                        for d2 in "0123456789":
                            temp = list(sub_9)
                            temp.insert(pos1, d1)
                            temp.insert(pos2, d2)
                            candidate = "".join(temp)
                            if len(candidate) == 11:
                                if candidate[0:2] in CODE_TO_WAGON_TYPE and candidate[2:4] in ZONE_CODE_MAP:
                                    if validate_wagon_number_checksum(candidate):
                                        reconstructed_candidates.append(candidate)
            if reconstructed_candidates:
                return reconstructed_candidates[0], "reconstructed"

    # Default fallback: replace OCR confusions with their first candidate
    if candidates_substrings:
        best_guess = []
        for char in candidates_substrings[0]:
            if char.isdigit():
                best_guess.append(char)
            elif char in CONFUSION_MAP:
                best_guess.append(CONFUSION_MAP[char][0])
        return "".join(best_guess), "best_guess"

    # Fallback to standard digit-only cleanup
    digits_only_raw = re.sub(r"\D", "", raw_text)
    return digits_only_raw[:11], "fallback"


def preprocess_crop(crop_pil):
    """
    Enhances crop region to boost PaddleOCR read accuracy.
    - Rotation/De-skewing: Detects angle and rotates the crop to make it horizontal.
    - Contrast: Applies CLAHE on L channel of LAB color space.
    - Adaptive Thresholding: Binarizes low-contrast or faded text.
    """
    # Convert PIL Image to cv2 BGR format
    open_cv_image = np.array(crop_pil)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    h, w = open_cv_image.shape[:2]
    if w == 0 or h == 0:
        return open_cv_image

    # 1. De-skewing / Rotation Correction
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 10:
        # Get minAreaRect parameters
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        # Normalize skew angle to range (-45, 45]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        if 0.5 < abs(angle) < 15.0:
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            open_cv_image = cv2.warpAffine(
                open_cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            h, w = open_cv_image.shape[:2]

    # 2. LAB space CLAHE for contrast enhancement
    lab = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b_ch))
    open_cv_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. Scale up if text region is small (aim for height of ~80px)
    target_h = 80
    if h < target_h:
        scale = target_h / h
        open_cv_image = cv2.resize(
            open_cv_image, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC
        )

    return open_cv_image


def levenshtein_distance(s1, s2):
    """Calculates Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def clean_text_class(text, label_class):
    """Clean text specifically based on its target category"""
    if not text:
        return ""
    text = text.strip()

    if label_class == "wagon_number":
        # Extract corrected numeric string
        corrected, method = correct_wagon_number_combinatorial(text)
        return corrected

    elif label_class == "wagon_type":
        # 1. Devanagari Hindi Translation
        for hi_type, en_type in sorted(DEVANAGARI_WAGON_TYPES.items(), key=lambda x: len(x[0]), reverse=True):
            if hi_type in text:
                return en_type

        # Remove symbols and lowercase, keep alphanumeric upper
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        # Direct exact lookup check
        if cleaned in VALID_WAGON_TYPES:
            return cleaned
        # Fuzzy match to closest valid type using Levenshtein distance
        best_match, min_dist = cleaned, 99
        for vt in VALID_WAGON_TYPES:
            dist = levenshtein_distance(cleaned, vt)
            if dist < min_dist:
                min_dist = dist
                best_match = vt
        return best_match if min_dist <= 2 else cleaned

    elif label_class == "zone_markings":
        cleaned = re.sub(r"[^A-Za-z]", "", text).upper()
        # Common corrections & aliases
        if cleaned in ["NF", "NFR"]:
            return "NFR"
        elif cleaned in ["ECOR", "ECR", "ECOR"]:
            return "ECoR"
        
        valid_zones = list(ZONE_CODE_MAP.values()) + ["ECR", "NFR", "ECoR", "ECOR", "NF"]
        if cleaned in valid_zones:
            return cleaned
            
        best_match, min_dist = cleaned, 99
        for vz in valid_zones:
            dist = levenshtein_distance(cleaned, vz)
            if dist < min_dist:
                min_dist = dist
                best_match = vz
        return best_match if min_dist <= 1 else cleaned

    elif label_class == "owner_name":
        # 1. Devanagari Hindi Translation
        for hi_owner, en_owner in sorted(DEVANAGARI_OWNER_TYPES.items(), key=lambda x: len(x[0]), reverse=True):
            if hi_owner in text or text in hi_owner:
                return en_owner

        cleaned = re.sub(r"[^A-Za-z]", "", text).upper()
        if cleaned in ["NF", "NFR"]:
            return "NFR"
        elif cleaned in ["ECOR", "ECR"]:
            return "ECoR"

        if cleaned in VALID_OWNER_NAMES:
            return cleaned
            
        best_match, min_dist = cleaned, 99
        for vo in VALID_OWNER_NAMES:
            dist = levenshtein_distance(cleaned, vo)
            if dist < min_dist:
                min_dist = dist
                best_match = vo
        return best_match if min_dist <= 1 else cleaned

    elif label_class == "tare_load_details":
        # Keep letters, numbers, spaces, decimals, and slashes/dashes
        cleaned = re.sub(r"[^A-Za-z0-9\s\.\-\/]", "", text).upper()
        # Clean common OCR spacing typos
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    return text


# Devanagari to English Mappings for Indian Railways Wagons & Owners
HINDI_DIGITS_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

DEVANAGARI_WAGON_TYPES = {
    "बीटीपीएन": "BTPN",
    "बी.टी.पी.एन.": "BTPN",
    "बॉक्सएन": "BOXN",
    "बॉक्सएनएचएस": "BOXNHS",
    "बॉक्सएनएचएल": "BOXNHL",
    "बॉक्सएनसीआर": "BOXNCR",
    "बॉक्सएनएचएसएम1": "BOXNHSM1",
    "बॉक्सएनएचएसएम१": "BOXNHSM1",
    "बॉक्सएनएम1": "BOXNM1",
    "बॉक्सएनसी": "BOXNC",
    "बॉक्सएनआर": "BOXNR",
    "बीसीएम": "BCN",
    "बीसीएनए": "BCNA",
    "बीसीएनएएचएस": "BCNAHS",
    "बीसीएनएचएल": "BCNHL",
    "बीसीएनएम": "BCNM",
    "बी सी एन एम": "BCNM",
    "बीसीएनएम1": "BCNM1",
    "बीसीएनएम१": "BCNM1",
    "बी सी एन एम १": "BCNM1",
    "बी सी एन एम 1": "BCNM1",
    "बीआरएन": "BRN",
    "बीआरएनए": "BRNA",
    "बीएफएनएस": "BFNS",
    "बीएफएनएसएम1": "BFNSM1",
    "बीओबीआरएन": "BOBRN",
    "बीओबीवाइएन": "BOBYN",
    "बीएलसी": "BLC",
    "बीएलसीए": "BLCA",
    "बीएलसीबी": "BLCB",
    "बीटीपीएनएचएस": "BTPNHS",
    "बॉस्ट": "BOST",
}

DEVANAGARI_OWNER_TYPES = {
    "जीआईपीएल": "GIPL",
    "जीआई पीएल": "GIPL",
    "जी आई पी एल": "GIPL",
    "जी.आई.पी.एल.": "GIPL",
    "पू त रे": "ECoR",
    "पूतरे": "ECoR",
    "पू.त.रे.": "ECoR",
    "पू म रे": "ECR",
    "पूमरे": "ECR",
    "पू.म.रे.": "ECR",
    "उ रे": "NR",
    "उरे": "NR",
    "उ.रे.": "NR",
    "उ म रे": "NCR",
    "उमरे": "NCR",
    "उ पू रे": "NER",
    "उपूरे": "NER",
    "उ सीमा रे": "NFR",
    "द रे": "SR",
    "दरे": "SR",
    "द म रे": "SCR",
    "दमरे": "SCR",
    "द पू रे": "SER",
    "दपूरे": "SER",
    "द पू म रे": "SECR",
    "दपूमरे": "SECR",
    "द प रे": "SWR",
    "दपरे": "SWR",
    "प रे": "WR",
    "परे": "WR",
    "प म रे": "WCR",
    "पमरे": "WCR",
    "म रे": "CR",
    "मरे": "CR",
}

def format_num_2dec(val_str, param_type="TARE"):
    try:
        val_str = val_str.translate(HINDI_DIGITS_MAP)
        val_str = val_str.replace("-", ".")
        val = float(val_str)
        
        # Auto-insert decimal if OCR missed dot e.g. 2050 -> 20.50, 7080 -> 70.80, 3032 -> 30.32
        if param_type == "TARE" and val > 100:
            val = val / 100.0
        elif param_type == "CC" and val > 200:
            val = val / 100.0
        elif param_type == "AREA" and val > 100:
            val = val / 10.0 if val < 1000 else val / 100.0
            
        # Physical boundary checks for heavy haul wagons
        if param_type == "TARE" and not (10.0 <= val <= 45.0):
            return None
        if param_type == "CC" and not (40.0 <= val <= 100.0):
            return None
        if param_type == "AREA" and not (10.0 <= val <= 70.0):
            return None

        return f"{val:.2f}"
    except Exception:
        return None


def verify_and_clean_tare_details(ocr_text, wagon_type):
    """
    Cleans up OCR-read tare details text in both English and Devanagari/Hindi.
    Enforces physical range validation (Tare: 10-45T, CC: 40-100T, Area: 10-70 sq.m).
    Formats numbers to 2 fixed decimal places.
    """
    if not ocr_text:
        return ""

    ocr_text = ocr_text.translate(HINDI_DIGITS_MAP)

    cleaned = re.sub(r"\b(\d{2})[\s\-]+(\d{1,2})\b", r"\1.\2", ocr_text.strip())
    cleaned = " ".join(cleaned.split())

    parts = []

    # 1. TARE
    tare_match = re.search(r"(?:TARE|टेयर|टैयर|टेर|टैर)\s*:?\s*-?\s*(\d+[\.\-]?\d*)\s*(?:T|टन)?", cleaned, re.IGNORECASE)
    if tare_match:
        val = format_num_2dec(tare_match.group(1), "TARE")
        if val:
            parts.append(f"TARE: {val}T")

    # 2. CC
    cc_match = re.search(r"(?:\bCC|CARRYING\s*CAPACITY|CAPACITY|\bCAP|भार\s*क्षमता|वहन\s*क्षमता|क्षमता|\bभार\b|वा[\.\s]*क्ष[\.\s]*)\s*:?\s*-?\s*(\d+[\.\-]?\d*)\s*(?:T|टन)?", cleaned, re.IGNORECASE)
    if cc_match:
        val = format_num_2dec(cc_match.group(1), "CC")
        if val:
            parts.append(f"CC: {val}T")

    # 3. AREA
    area_match = re.search(
        r"(?:AREA|AR|SQM|SQ\.?M|M2|वर्ग\s*मी|वर्गमी|क्षेत्रफल)\s*:?\s*-?\s*(\d+[\.\-]?\d*)|(\d+[\.\-]?\d*)\s*(?:SQM|SQ\.?M|M2|वर्ग\s*मी|वर्गमी|क्षेत्रफल)",
        cleaned, re.IGNORECASE
    )
    if area_match:
        raw_val = area_match.group(1) or area_match.group(2)
        val = format_num_2dec(raw_val, "AREA")
        if val:
            parts.append(f"AREA: {val} sq.m")

    if parts:
        return " / ".join(parts)

    return ""


def cross_validate_results(results):
    """
    Cleans and verifies OCR-read values against ROAMS master lists.
    Enforces STRICT master registry matching — non-master noise strings are 100% prohibited.
    """
    wnum = results.get("wagon_number", "").strip()
    is_wnum_valid = len(wnum) == 11 and validate_wagon_number_checksum(wnum)
    results["wagon_number_checksum_valid"] = is_wnum_valid
    results["wagon_number_verified"] = is_wnum_valid

    # 1. Clean & Validate Wagon Type against 209 ROAMS Master Types
    wtype = results.get("wagon_type", "").strip().upper()
    if wtype:
        sorted_hi = sorted(DEVANAGARI_WAGON_TYPES.items(), key=lambda x: len(x[0]), reverse=True)
        for hi_type, en_type in sorted_hi:
            if hi_type in wtype:
                wtype = en_type
                break

        best_wtype = None
        min_dist = 999
        if wtype in VALID_WAGON_TYPES:
            best_wtype = wtype
        else:
            for vt in VALID_WAGON_TYPES:
                dist = levenshtein_distance(wtype, vt)
                if dist < min_dist:
                    min_dist = dist
                    best_wtype = vt
            # STRICT RULE: Reject raw noise if Levenshtein distance > 2!
            if min_dist > 2:
                best_wtype = ""

        # Fallback to 11-digit wagon number prefix if raw OCR failed or was rejected
        if not best_wtype and is_wnum_valid:
            best_wtype = CODE_TO_WAGON_TYPE.get(wnum[:2], "")

        results["wagon_type"] = best_wtype
        results["wagon_type_verified"] = best_wtype in VALID_WAGON_TYPES
    else:
        if is_wnum_valid:
            results["wagon_type"] = CODE_TO_WAGON_TYPE.get(wnum[:2], "")
            results["wagon_type_verified"] = results["wagon_type"] in VALID_WAGON_TYPES
        else:
            results["wagon_type_verified"] = False

    # 2. Clean & Validate Owner Name (Entity: CONCOR, SAIL, IOCL, etc.)
    owner = results.get("owner_name", "").strip().upper()
    best_owner = ""
    if owner:
        for hi_owner, en_owner in sorted(DEVANAGARI_OWNER_TYPES.items(), key=lambda x: len(x[0]), reverse=True):
            if hi_owner in owner:
                owner = en_owner
                break

        min_dist = 999
        if owner in VALID_OWNER_NAMES:
            best_owner = owner
        else:
            for vo in VALID_OWNER_NAMES:
                dist = levenshtein_distance(owner, vo)
                if dist < min_dist:
                    min_dist = dist
                    best_owner = vo
            if min_dist > 2:
                best_owner = ""

    # Fallback to 11-digit wagon number digits 3-4 if owner OCR missed
    if not best_owner and is_wnum_valid:
        best_owner = ZONE_CODE_MAP.get(wnum[2:4], "")

    results["owner_name"] = best_owner
    results["owner_name_verified"] = best_owner in VALID_OWNER_NAMES

    # 3. Clean & Preserve Zone Markings (Railway Zone: SECR, ECoR, NR, WR, etc.)
    zone = results.get("zone_markings", "").strip().upper()
    best_zone = ""
    if zone:
        min_dist = 999
        if zone in VALID_OWNER_NAMES:
            best_zone = zone
        else:
            for vz in VALID_OWNER_NAMES:
                dist = levenshtein_distance(zone, vz)
                if dist < min_dist:
                    min_dist = dist
                    best_zone = vz
            if min_dist > 2:
                best_zone = ""

    # Fallback zone from 11-digit wagon number if missing
    if not best_zone and is_wnum_valid:
        best_zone = ZONE_CODE_MAP.get(wnum[2:4], "")

    # PRESERVE zone_markings as its own distinct field for stats & database
    results["zone_markings"] = best_zone
    results["zone_markings_verified"] = best_zone in VALID_OWNER_NAMES

    # 4. Standardize Tare Details
    tare_details = results.get("tare_load_details", "").strip()
    cleaned_tare = verify_and_clean_tare_details(tare_details, results.get("wagon_type", ""))
    results["tare_load_details"] = cleaned_tare
    results["tare_load_details_verified"] = bool(cleaned_tare and ("TARE" in cleaned_tare.upper() or "CC" in cleaned_tare.upper() or "AREA" in cleaned_tare.upper()))

    return results


# ── Core Pipeline Logic ─────────────────────────────────────────────
def run_pipeline(image_path, rf_client=None, ocr_client=None, model_id=None):
    """Executes the full crop, preprocess, read, validate flow"""
    start_time = time.time()
    res = {
        "image_name": Path(image_path).name,
        "chunk": Path(image_path).parent.name if ("chunk" in Path(image_path).parent.name or "dataset_v3_addition" in Path(image_path).parent.name) else "upload",
        "wagon_number": "",
        "wagon_number_checksum_valid": False,
        "wagon_type": "",
        "owner_name": "",
        "zone_markings": "",
        "tare_load_details": "",
        "stencil": "No Stencil Found",
        "processing_time_sec": 0,
    }

    try:
        # Step 1: Detect bounding boxes using local YOLOv8 (GPU/CUDA) or cloud API fallback
        local_model_path = Path("cris_ocr_pipeline/models/best.pt")
        use_local_yolo = False
        predictions = []

        if local_model_path.exists():
            try:
                from ultralytics import YOLO
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_key = str(local_model_path.resolve())
                if model_key not in YOLO_MODEL_CACHE:
                    print(f"Loading local YOLOv8 model on device: {device}...")
                    YOLO_MODEL_CACHE[model_key] = YOLO(str(local_model_path))
                model = YOLO_MODEL_CACHE[model_key]
                local_results = model.predict(str(image_path), device=device, verbose=False)
                
                if local_results and len(local_results) > 0:
                    for box in local_results[0].boxes:
                        x_center, y_center, width, height = box.xywh[0].tolist()
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        
                        classes_map = ["owner_name", "tare_load_details", "wagon_number", "wagon_type", "zone_markings"]
                        cls_name = classes_map[cls_id] if cls_id < len(classes_map) else str(cls_id)
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        points = [
                            {"x": x1, "y": y1},
                            {"x": x2, "y": y1},
                            {"x": x2, "y": y2},
                            {"x": x1, "y": y2}
                        ]
                        
                        predictions.append({
                            "class": cls_name,
                            "confidence": conf,
                            "x": x_center,
                            "y": y_center,
                            "width": width,
                            "height": height,
                            "points": points
                        })
                    use_local_yolo = True
                    print(f"Local YOLOv8 GPU/CUDA inference completed with {len(predictions)} detections.")
            except Exception as e:
                print(f"Local YOLOv8 GPU/CUDA inference failed: {e}")

        if not use_local_yolo:
            print("ERROR: Local YOLOv8 model not found at cris_ocr_pipeline/models/best.pt. Cannot run detection.")

        # Store detections in result for annotated image generation
        res["detections"] = predictions

        # Step 2: Open image & process crops
        image = Image.open(image_path).convert("RGB")

        # Check if multiple wagon_type boxes exist (e.g. Box 1: BCNA, Box 2: HSM1) and merge them left-to-right
        type_preds = [p for p in predictions if p.get("class") == "wagon_type"]
        if len(type_preds) > 1:
            type_preds.sort(key=lambda p: float(p.get("x", 0)))
            x_min = min(float(p["x"]) - float(p["width"]) / 2 for p in type_preds)
            x_max = max(float(p["x"]) + float(p["width"]) / 2 for p in type_preds)
            y_min = min(float(p["y"]) - float(p["height"]) / 2 for p in type_preds)
            y_max = max(float(p["y"]) + float(p["height"]) / 2 for p in type_preds)
            
            merged_w = x_max - x_min
            merged_h = y_max - y_min
            merged_x = x_min + merged_w / 2
            merged_y = y_min + merged_h / 2
            max_conf = max(float(p.get("confidence", 0)) for p in type_preds)
            
            merged_pred = {
                "class": "wagon_type",
                "x": merged_x,
                "y": merged_y,
                "width": merged_w,
                "height": merged_h,
                "confidence": max_conf
            }
            predictions = [p for p in predictions if p.get("class") != "wagon_type"] + [merged_pred]

        # Sort predictions so that main foreground detections (highest confidence & box area) come FIRST
        predictions.sort(
            key=lambda p: float(p.get("confidence", 0)) * (float(p.get("width", 0)) * float(p.get("height", 0))),
            reverse=True
        )

        for pred in predictions:
            cls_name = pred["class"]
            
            # If we ALREADY have a verified field for this class, skip minor/background detections
            if res.get(cls_name) and (res.get(f"{cls_name}_verified") or res.get(f"{cls_name}_checksum_valid")):
                continue
            
            # 1. Perspective Warping if YOLO OBB polygon points are present
            if "points" in pred and len(pred["points"]) == 4:
                try:
                    pts = np.array([[pt["x"], pt["y"]] for pt in pred["points"]], dtype="float32")
                    # Sort corners: top-left, top-right, bottom-right, bottom-left
                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1)
                    tl = pts[np.argmin(s)]
                    br = pts[np.argmax(s)]
                    tr = pts[np.argmin(diff)]
                    bl = pts[np.argmax(diff)]
                    
                    w_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    w_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    max_w = max(int(w_a), int(w_b))
                    
                    h_a = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))
                    h_b = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[0] - bl[0]) ** 2))
                    max_h = max(int(h_a), int(h_b))
                    
                    dst = np.array([
                        [0, 0],
                        [max_w - 1, 0],
                        [max_w - 1, max_h - 1],
                        [0, max_h - 1]
                    ], dtype="float32")
                    
                    img_cv = cv2.imread(str(image_path))
                    M_warp = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
                    warp_crop_cv = cv2.warpPerspective(img_cv, M_warp, (max_w, max_h))
                    
                    crop = Image.fromarray(cv2.cvtColor(warp_crop_cv, cv2.COLOR_BGR2RGB))
                except Exception as warping_err:
                    print(f"Angle warping failed, falling back to rectangular crop: {warping_err}")
                    # Fallback to normal bounding box crop
                    pad_w = int(pred["width"] * 0.15)
                    pad_h = int(pred["height"] * 0.10)
                    x1 = max(0, int(pred["x"] - pred["width"] / 2) - pad_w)
                    y1 = max(0, int(pred["y"] - pred["height"] / 2) - pad_h)
                    x2 = min(image.width, int(pred["x"] + pred["width"] / 2) + pad_w)
                    y2 = min(image.height, int(pred["y"] + pred["height"] / 2) + pad_h)
                    crop = image.crop((x1, y1, x2, y2))
            else:
                # 2. Adaptive Bounding Box crop padding based on aspect ratio
                width, height = pred["width"], pred["height"]
                aspect_ratio = width / max(height, 1)
                
                if aspect_ratio > 2.0:
                    # Horizontal text lines (Wagon Number, Tare capacity)
                    pad_w = int(width * 0.20)
                    pad_h = int(height * 0.08)
                else:
                    # Vertical or square markings (Railway Zone, Owner Name)
                    pad_w = int(width * 0.12)
                    pad_h = int(height * 0.12)
                
                x1 = max(0, int(pred["x"] - width / 2) - pad_w)
                y1 = max(0, int(pred["y"] - height / 2) - pad_h)
                x2 = min(image.width, int(pred["x"] + width / 2) + pad_w)
                y2 = min(image.height, int(pred["y"] + height / 2) + pad_h)
                crop = image.crop((x1, y1, x2, y2))

            preprocessed_crop = preprocess_crop(crop)
            # Multi-tier OCR routing to find a verified result for this category
            processed_text = ""
            
            # Helper to check if a text candidate is validated
            def check_field_verification(val):
                if not val:
                    return False
                cleaned = clean_text_class(val, cls_name)
                if not cleaned:
                    return False
                if cls_name == "wagon_number":
                    return len(cleaned) == 11 and validate_wagon_number_checksum(cleaned)
                elif cls_name == "wagon_type":
                    if cleaned not in VALID_WAGON_TYPES:
                        return False
                    # Family sanity check: If wagon_number is verified, ensure wagon_type matches expected family
                    wnum = res.get("wagon_number", "")
                    if len(wnum) == 11 and validate_wagon_number_checksum(wnum):
                        expected_base = CODE_TO_WAGON_TYPE.get(wnum[:2], "")
                        if expected_base:
                            exp_prefix = expected_base[:2] # e.g. "BO" for BOXN/BOXNHL, "BC" for BCNHL, "BT" for BTPN
                            if not cleaned.startswith(exp_prefix):
                                return False
                    return True
                elif cls_name == "owner_name" or cls_name == "zone_markings":
                    return cleaned in VALID_OWNER_NAMES
                elif cls_name == "tare_load_details":
                    return "TARE" in cleaned.upper() and "CC" in cleaned.upper()
                return False

            # Helper to parse PaddleOCR results
            def parse_paddle_lines(ocr_res):
                if not ocr_res or not ocr_res[0]:
                    return ""
                if isinstance(ocr_res[0], dict):
                    return " ".join(ocr_res[0].get("rec_texts", []))
                
                lines_data = []
                for line in ocr_res[0]:
                    if isinstance(line, (list, tuple)) and len(line) > 1:
                        box = line[0]
                        text = line[1][0]
                        y_center = sum([pt[1] for pt in box]) / 4
                        x_center = sum([pt[0] for pt in box]) / 4
                        lines_data.append((x_center, y_center, text))
                
                # Spatial sort: Top-to-Bottom first (y_center), then Left-to-Right (x_center)
                lines_data.sort(key=lambda item: (item[1], item[0]))
                return " ".join([item[2] for item in lines_data])

            # Tier 1 (Primary): Qwen2.5-VL Vision-Language Model OCR for all categories
            try:
                if hasattr(ocr_client, "ocr_vlm"):
                    vlm_raw = ocr_client.ocr_vlm(crop, cls_name)
                    if vlm_raw:
                        vlm_clean = clean_text_class(vlm_raw, cls_name)
                        if check_field_verification(vlm_clean):
                            processed_text = vlm_clean
                            print(f"Tier 1 (VLM Qwen2.5-VL Primary) resolved field: {processed_text}")
                        elif vlm_clean:
                            clean_t1 = vlm_clean
            except Exception as e:
                pass

            # Tier 1.5 (VLM +10% Expanded Margin Padding): Qwen2.5-VL on widened crop for edge-clipped text
            if not processed_text:
                try:
                    if hasattr(ocr_client, "ocr_vlm"):
                        width, height = pred["width"], pred["height"]
                        pad_w = int(width * 0.10)
                        pad_h = int(height * 0.10)
                        
                        x1 = max(0, int(pred["x"] - width / 2) - pad_w)
                        y1 = max(0, int(pred["y"] - height / 2) - pad_h)
                        x2 = min(image.width, int(pred["x"] + width / 2) + pad_w)
                        y2 = min(image.height, int(pred["y"] + height / 2) + pad_h)
                        
                        crop_10 = image.crop((x1, y1, x2, y2))
                        vlm_10_raw = ocr_client.ocr_vlm(crop_10, cls_name)
                        if vlm_10_raw:
                            vlm_10_clean = clean_text_class(vlm_10_raw, cls_name)
                            if check_field_verification(vlm_10_clean):
                                processed_text = vlm_10_clean
                                print(f"Tier 1.5 (VLM Qwen2.5-VL +10% Expanded) resolved field: {processed_text}")
                            elif vlm_10_clean and not clean_t1:
                                clean_t1 = vlm_10_clean
                except Exception as e:
                    pass

            # Tier 2: PaddleOCR on Preprocessed Crop
            if not processed_text:
                try:
                    res_t1 = ocr_client.ocr(preprocessed_crop)
                    text_t1 = parse_paddle_lines(res_t1)
                    clean_t1 = clean_text_class(text_t1, cls_name)
                    if check_field_verification(clean_t1):
                        processed_text = clean_t1
                        print(f"Tier 2 (PaddleOCR Preprocessed) resolved field: {processed_text}")
                except Exception as e:
                    print(f"Tier 2 PaddleOCR failed: {e}")

            # Tier 3: PaddleOCR on Raw Crop (if unverified)
            if not processed_text:
                try:
                    raw_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
                    res_t2 = ocr_client.ocr(raw_bgr)
                    text_t2 = parse_paddle_lines(res_t2)
                    clean_t2 = clean_text_class(text_t2, cls_name)
                    if check_field_verification(clean_t2):
                        processed_text = clean_t2
                        print(f"Tier 3 OCR (Raw BGR) resolved field: {processed_text}")
                    elif clean_t2 and len(clean_t2) > len(clean_t1 or ""):
                        clean_t1 = clean_t2
                except Exception as e:
                    print(f"Tier 3 PaddleOCR raw failed: {e}")

            # Tier 3: EasyOCR on Raw Crop (if Tier 2 is unverified and EasyOCR is installed)
            if not processed_text:
                try:
                    import easyocr
                    import torch
                    if not hasattr(run_pipeline, "easyocr_reader"):
                        run_pipeline.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                    
                    easy_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
                    easy_res = run_pipeline.easyocr_reader.readtext(easy_bgr, detail=0)
                    if easy_res:
                        easy_raw = " ".join(easy_res)
                        easy_processed = clean_text_class(easy_raw, cls_name)
                        if check_field_verification(easy_processed):
                            processed_text = easy_processed
                            print(f"Tier 3 EasyOCR resolved field: {processed_text}")
                        elif easy_processed and not clean_t1:
                            clean_t1 = easy_processed
                except Exception:
                    pass # Skip if not installed

            # Tier 4: Tesseract on Raw Crop (if Tier 3 is unverified and Tesseract is installed)
            if not processed_text:
                try:
                    import pytesseract
                    gray = cv2.cvtColor(cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                    tess_config = ""
                    if cls_name == "wagon_number":
                        tess_config = "--psm 7 -c tessedit_char_whitelist=0123456789"
                    elif cls_name in ["wagon_type", "owner_name", "zone_markings"]:
                        tess_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    
                    tess_raw = pytesseract.image_to_string(gray, config=tess_config)
                    if tess_raw:
                        tess_processed = clean_text_class(tess_raw, cls_name)
                        if check_field_verification(tess_processed):
                            processed_text = tess_processed
                            print(f"Tier 4 Tesseract resolved field: {processed_text}")
                        elif tess_processed and not clean_t1:
                            clean_t1 = tess_processed
                except Exception:
                    pass # Skip if not installed

            # Tier 4: Bounding Box Margin Expansion Fallback (+10% Single Pass)
            if not processed_text:
                scale = 0.10
                try:
                    width, height = pred["width"], pred["height"]
                    aspect_ratio = width / max(height, 1)
                    
                    if aspect_ratio > 2.0:
                        pad_w = int(width * (0.20 + scale))
                        pad_h = int(height * (0.08 + scale))
                    else:
                        pad_w = int(width * (0.12 + scale))
                        pad_h = int(height * (0.12 + scale))
                    
                    x1 = max(0, int(pred["x"] - width / 2) - pad_w)
                    y1 = max(0, int(pred["y"] - height / 2) - pad_h)
                    x2 = min(image.width, int(pred["x"] + width / 2) + pad_w)
                    y2 = min(image.height, int(pred["y"] + height / 2) + pad_h)
                    
                    scale_crop = image.crop((x1, y1, x2, y2))
                    scale_preprocessed = preprocess_crop(scale_crop)
                    
                    scale_res = ocr_client.ocr(scale_preprocessed)
                    scale_text = parse_paddle_lines(scale_res)
                    scale_clean = clean_text_class(scale_text, cls_name)
                    
                    if check_field_verification(scale_clean):
                        processed_text = scale_clean
                        print(f"Tier 4 Margin +10% expansion resolved field: {processed_text}")
                except Exception as scale_err:
                    print(f"Tier 4 Margin +10% expansion failed: {scale_err}")

            # Fallback if no verified candidate found: use the best raw prediction and clean it
            if not processed_text:
                processed_text = clean_text_class(clean_t1, cls_name)

            res[cls_name] = processed_text

        # Step 4: Verify checksum and cross-validate
        if res["wagon_number"]:
            res["wagon_number_checksum_valid"] = validate_wagon_number_checksum(
                res["wagon_number"]
            )

        # Full Image OCR Fallback - ONLY for wagon_number recovery (checksum-validated)
        # All other categories (wagon_type, owner_name, zone_markings, tare_load_details)
        # must come exclusively from their own YOLOv8 bounding box crops.
        res = cross_validate_results(res)
        
        if not res.get("wagon_number_checksum_valid"):
            try:
                print(f"Triggering Full Image fallback scan to recover fields: ['wagon_number']...")
                raw_bgr_full = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                fallback_result = ocr_client.ocr(raw_bgr_full)
                
                if fallback_result and fallback_result[0]:
                    fallback_lines = []
                    if isinstance(fallback_result[0], dict):
                        fallback_lines = fallback_result[0].get("rec_texts", [])
                    else:
                        fallback_lines = [line[1][0] for line in fallback_result[0] if isinstance(line, (list, tuple)) and len(line) > 1]
                    
                    # Recover Wagon Number (has strong checksum validation)
                    solved_num = None
                    for line in fallback_lines:
                        digits = re.sub(r"\D", "", line)
                        if len(digits) == 11 and validate_wagon_number_checksum(digits):
                            solved_num = digits
                            break
                    if not solved_num:
                        for line in fallback_lines:
                            digits = re.sub(r"\D", "", line)
                            if len(digits) == 10:
                                cand, _ = correct_wagon_number_combinatorial(digits)
                                if cand and len(cand) == 11 and validate_wagon_number_checksum(cand):
                                    solved_num = cand
                                    break
                    if solved_num:
                        res["wagon_number"] = solved_num
                        res["wagon_number_checksum_valid"] = True
                        res["wagon_number_verified"] = True
                        print(f"Recovered wagon number from full-image scan: {solved_num}")
            except Exception as fe:
                print(f"Error running fallback full-image OCR: {fe}")

        res = cross_validate_results(res)
        res["detections"] = predictions

        # Run Stencil Detection model if available
        stencil_model_path = Path("cris_ocr_pipeline/models/stencil_best.pt")
        if stencil_model_path.exists():
            try:
                from ultralytics import YOLO
                import torch
                print("Running Stencil Detection model...")
                s_device = "cuda" if torch.cuda.is_available() else "cpu"
                s_model_key = str(stencil_model_path.resolve())
                if s_model_key not in YOLO_MODEL_CACHE:
                    print("Loading Stencil Detection model...")
                    YOLO_MODEL_CACHE[s_model_key] = YOLO(str(stencil_model_path))
                s_model = YOLO_MODEL_CACHE[s_model_key]
                s_results = s_model.predict(str(image_path), device=s_device, verbose=False, conf=0.5)
                
                if s_results and len(s_results) > 0 and len(s_results[0].boxes) > 0:
                    names_map = s_model.names
                    detected_stencils = []
                    highest_conf = 0.0

                    for b in s_results[0].boxes:
                        cls_id = int(b.cls[0].item())
                        conf = float(b.conf[0].item())
                        cls_name = names_map.get(cls_id, str(cls_id)) if names_map else str(cls_id)
                        if conf > highest_conf:
                            highest_conf = conf
                        detected_stencils.append(f"{cls_name} ({round(conf * 100)}%)")

                    stencil_str = ", ".join(detected_stencils)
                    res["stencil"] = f"Found ({round(highest_conf * 100)}% - {stencil_str})"
                    print(f"Stencil Found: {res['stencil']}")
                else:
                    res["stencil"] = "No Stencil Found"
                    print("No Stencil Found")
            except Exception as se:
                print(f"Error running stencil model: {se}")
                res["stencil"] = "No Stencil Found"

    except Exception as e:
        res["error"] = str(e)

    res["processing_time_sec"] = round(time.time() - start_time, 2)
    return res


# ── Main / CLI Runner ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Indian Railways Wagon OCR Pipeline")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to folder or single image to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cris_ocr_pipeline/outputs/ocr_results",
        help="Folder to save output reports",
    )


    args = parser.parse_args()

    # Folders setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out_path = output_dir / "ocr_results_manifest.json"
    csv_out_path = output_dir / "ocr_results_report.csv"

    # Initialize clients
    print("Initializing clients (Local YOLOv8 + PaddleOCR)...")
    import requests
    class RemoteOCRClient:
        def ocr(self, img_np):
            try:
                _, img_encoded = cv2.imencode(".jpg", img_np)
                img_bytes = img_encoded.tobytes()
                response = requests.post(
                    "http://127.0.0.1:8001/ocr",
                    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                    timeout=15
                )
                if response.status_code == 200:
                    lines = response.json().get("lines", [])
                    return [{"rec_texts": lines}]
            except Exception as e:
                print(f"Remote GPU OCR request failed: {e}")
            return [None]

    ocr_client = RemoteOCRClient()

    # Load existing progress if available (resumability)
    existing_results = []
    processed_names = set()
    if json_out_path.exists():
        try:
            with open(json_out_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                processed_names = {r["image_name"] for r in existing_results}
            print(f"Resuming progress: {len(processed_names)} images already completed.")
        except Exception:
            pass

    # Collect images to process
    img_path = Path(args.images)
    image_files = []
    if img_path.is_dir():
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(img_path.rglob(f"*{ext}"))
    elif img_path.is_file():
        image_files.append(img_path)

    # Exclude already processed images
    image_files = [f for f in image_files if f.name not in processed_names]
    total_images = len(image_files)

    if total_images == 0:
        print("No new images to process.")
        return

    print(f"\nProcessing {total_images} new images...")
    results_list = list(existing_results)

    success_count = 0
    checksum_passed = 0

    for idx, img_file in enumerate(image_files):
        print(f"[{idx+1}/{total_images}] Running OCR on: {img_file.name}...", end="")
        res = run_pipeline(img_file, ocr_client=ocr_client)

        # Save result
        results_list.append(res)
        success_count += 1
        if res.get("wagon_number_checksum_valid"):
            checksum_passed += 1

        print(
            f" [Wagon: {res.get('wagon_number', 'N/A')}] [Valid: {res.get('wagon_number_checksum_valid', False)}]"
        )

        # Incremental Save to prevent loss if interrupted
        if True:
            # Save JSON
            with open(json_out_path, "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2)

            # Save CSV
            csv_headers = [
                "image_name",
                "wagon_number",
                "wagon_number_checksum_valid",
                "wagon_type",
                "owner_name",
                "zone_markings",
                "tare_load_details",
                "processing_time_sec",
            ]
            with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results_list)

    # Print Final Summary Report
    print(f"\n{'='*50}")
    print("PIPELINE PROCESSING COMPLETE!")
    print(f"Total Processed:  {success_count}")
    print(f"Checksum Passed:  {checksum_passed} ({round(checksum_passed/max(1, success_count)*100, 1)}%)")
    print(f"Manifest JSON:    {json_out_path}")
    print(f"CSV Report:       {csv_out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
