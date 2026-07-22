import io
import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

app = FastAPI(title="CRIS Wagon VLM OCR Service (Qwen2.5-VL)")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading VLM model '{MODEL_ID}' in 4-bit precision on device: {device}...")
try:
    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("VLM model loaded successfully!")
except Exception as e:
    print(f"Failed to load VLM model: {e}")
    model = None
    processor = None

PROMPTS = {
    "wagon_number": (
        "You are an expert Indian Railways wagon number OCR architect. Read the 11-digit wagon number painted on this crop.\n"
        "RULES:\n"
        "1. Indian Railway wagon numbers consist of EXACTLY 11 digits (English digits 0-9 or Devanagari Hindi digits ०-९).\n"
        "2. SPLIT LAYOUT HANDLING:\n"
        "   - Horizontal split (left & right blocks side-by-side): Read left-to-right to concatenate into one 11-digit sequence.\n"
        "   - Vertical split (stacked top & bottom lines, e.g. top 6 digits '331215' above bottom 5 digits '65554'): Read top line FIRST, then bottom line SECOND, concatenating into one continuous 11-digit sequence '33121565554'.\n"
        "3. Ignore surrounding letters, logos, or markings.\n"
        "4. Output ONLY the raw 11-digit sequence without spaces, hyphens, punctuation, or extra words. If unreadable, return empty string."
    ),
    "wagon_type": (
        "Read the wagon type code from this Indian Railways freight wagon crop image.\n"
        "CRITICAL FORMAT RULE: Output ONLY the bare code. No markdown, no backticks, no bold, no quotes, no explanation, no sentences. Just the code.\n\n"
        "VALID CODES (output one of these or a variant with suffix M1/M2/M3/M4/2D/25T/HS/HL/AL/CR):\n"
        "BOXN, BOXNHL, BOXNHS, BOXNR, BOXNRM2, BOXNS, BOXNM1, BOXNM2, BOXNHSM1, BOXNHSM2, BOXNCR, BOXNHA, BOXNLW, BOXNB, BOXNF, BOXNG, BOXNHL2D, BOXNHL25T, BOXNAL,\n"
        "BCNA, BCNAHS, BCNAHSM1, BCNAM1, BCNHL, BCNM1, BCCNR,\n"
        "BOBYN, BOBYNHS, BOBRN, BOBRNHS, BOBRNHSM1, BOBRAL,\n"
        "BTPN, BTPNHS, BTPGLN, BTALN, BTCS, BTPH, BTAP, BTFLN,\n"
        "BRNA, BRNAHS, BRNAM, BRSM1, BRS, BFNS, BFNSM1,\n"
        "BOST, BOSTHS, BLCA, BLCB, BLCSA, BLCAM, BLCBM, BLSSB,\n"
        "BLLA, BLLB, BFU, BWTB, MBWT, DBKM, BVZC, BVZI, BOMN, BFAT, BRSTH, RORO, BV, BOY.\n\n"
        "LANGUAGE HANDLING:\n"
        "- If text is in Devanagari Hindi (e.g. बॉक्सएन, बीसीएनए, बीटीपीएन, बाकसएनएचएल), translate to its English code.\n"
        "- बॉक्सएन/बाकसएन = BOXN, बॉक्सएनएचएल/बाकसएनएचएल = BOXNHL, बीसीएनए = BCNA, बीसीएनएचएल = BCNHL, बीटीपीएन = BTPN, बीओबीवाइएन = BOBYN.\n\n"
        "SUFFIX RULES: Preserve ALL suffixes exactly as painted (M1, M2, M3, M4, 2D, 25T, HS, HL, AL, CR, HSM1, HSM2, RM2, etc.).\n\n"
        "EXAMPLES:\n"
        "- Image shows 'BOXN' → Output: BOXN\n"
        "- Image shows 'BOXNHL' → Output: BOXNHL\n"
        "- Image shows 'बीसीएनएएचएस' → Output: BCNAHS\n"
        "- Image shows 'BCNM1' → Output: BCNM1\n"
        "- Unreadable → Output: (empty)\n"
    ),
    "owner_name": (
        "Read the owner name or railway zone marking from this Indian Railways wagon crop image.\n"
        "CRITICAL FORMAT RULE: Output ONLY the single bare code in English. No markdown, no backticks, no bold, no quotes, no explanation. Just the code.\n\n"
        "STANDARD CODES:\n"
        "- ZONES: CR, ER, ECR, ECoR, NR, NCR, NER, NFR, NWR, SR, SCR, SER, SECR, SWR, WR, WCR, KR (Konkan Railway), DFCC (Dedicated Freight Corridor).\n"
        "- LOGISTICS / PSUs: CONCOR, PRCL, RSPL, SCAL, ARIL, GIPL, SAIL, NTPC, IOCL, IOC, BALCO, NALCO, TATA, IFFCO, KRIBHCO, DPW (DP World).\n\n"
        "LANGUAGE TRANSLATION (Hindi Devanagari -> English):\n"
        "- म.रे. / मरे / मध्य रेलवे = CR\n"
        "- पू.रे. / पूरे / पूर्वी रेलवे = ER\n"
        "- पू.म.रे. / पूमरे / पूर्व मध्य रेलवे = ECR\n"
        "- पू.त.रे. / पूतरे / पूर्व तट रेलवे = ECoR\n"
        "- उ.रे. / उरे / एनआर / उत्तर रेलवे = NR\n"
        "- उ.म.रे. / उमरे / एनसीआर = NCR\n"
        "- उ.सीमा.रे. / उसिमारे / एनएफआर = NFR\n"
        "- द.पू.म.रे. / दपूमरे / एसईसीआर = SECR\n"
        "- द.म.रे. / दमरे / एससीआर = SCR\n"
        "- द.पू.रे. / दपूरे / एसईआर = SER\n"
        "- प.रे. / परे / डब्ल्यूआर = WR\n"
        "- प.म.रे. / पमरे / डब्ल्यूसीआर = WCR\n"
        "- केआर / कोंकण रेलवे = KR\n"
        "- डीएफसीसी = DFCC\n"
        "- कॉनकोर / कॉन कोर = CONCOR, सेल = SAIL, जीआईपीएल = GIPL, पीआरसीएल = PRCL, डीपी वर्ल्ड = DPW.\n\n"
        "Output ONLY the extracted English owner/zone code. If unreadable, return empty string."
    ),
    "zone_markings": (
        "Read the railway zone or owner code painted on this Indian Railways crop image.\n"
        "CRITICAL FORMAT RULE: Output ONLY the bare code in English. No markdown, no backticks, no bold, no quotes, no explanation.\n\n"
        "STANDARD CODES: CR, ER, ECR, ECoR, NR, NCR, NER, NFR, NWR, SR, SCR, SER, SECR, SWR, WR, WCR, KR, DFCC, CONCOR, PRCL, RSPL, GIPL, SAIL, NTPC, IOCL, BALCO, NALCO, TATA.\n\n"
        "ALWAYS translate Devanagari Hindi markings (e.g. म.रे. -> CR, पूमरे -> ECR, पूतरे -> ECoR, उमरे -> NCR, दपूमरे -> SECR, एनएफआर -> NFR, केआर -> KR, डीएफसीसी -> DFCC, सेल -> SAIL, कॉनकोर -> CONCOR) to their English abbreviation.\n\n"
        "Output ONLY the extracted English code. If unreadable, return empty string."
    ),
    "tare_load_details": (
        "You are an expert Indian Railways technical load details OCR architect. Read the TARE, CC, and AREA parameter values painted on this crop.\n"
        "RULES:\n"
        "1. Extract ONLY these 3 parameters when present:\n"
        "   - TARE / टेयर / टैयर / टायर (Empty wagon weight in tonnes, e.g. TARE 20.5T)\n"
        "   - CC / CARRYING CAPACITY / क्षमता / वा.क्ष. / वहन क्षमता (Carrying Capacity weight in tonnes, e.g. CC 66.8T)\n"
        "   - AREA / SQM / M2 / वर्गमी / वर्ग मी / क्षेत्रफल (Floor area in sq.m, e.g. AREA 36.5SQM)\n"
        "2. Format output cleanly as: TARE: XX.XXT / CC: XX.XXT / AREA: XX.XX sq.m.\n"
        "3. Ignore unrelated stamps, maintenance dates, or serial numbers."
    )
}

import re

def _strip_vlm_wrapper(text: str) -> str:
    """Strip markdown wrappers, conversational sentences, and extract bare code from VLM response."""
    if not text:
        return ""
    
    # Strip markdown: bold (**text**), backticks (```text```), inline code (`text`)
    cleaned = text.strip()
    cleaned = re.sub(r'```+', '', cleaned)  # remove triple backticks
    cleaned = re.sub(r'\*\*', '', cleaned)   # remove bold markers
    cleaned = re.sub(r'`', '', cleaned)      # remove inline code markers
    cleaned = re.sub(r'"', '', cleaned)      # remove quotes
    cleaned = re.sub(r"'", '', cleaned)      # remove single quotes
    cleaned = cleaned.strip()
    
    # If the result is now a short clean alphanumeric token (likely the bare code), return it
    bare = re.sub(r'[^A-Za-z0-9]', '', cleaned)
    if bare and len(bare) <= 15 and bare.upper()[:3] in ('BOX', 'BCN', 'BTP', 'BOB', 'BRN', 'BFN', 'BOS', 'BLC', 'BLL', 'BWT', 'MBW', 'DBK', 'BVZ', 'BOM', 'BFA', 'BRS', 'BFU', 'ROR', 'BOY', 'BCA'):
        return bare.upper()
    
    # If it's a long conversational response, try to find a valid wagon-type-like token
    tokens = re.findall(r'[A-Za-z0-9]+', cleaned)
    wagon_prefixes = ('BOXN', 'BOX', 'BCNA', 'BCN', 'BTPN', 'BTP', 'BOBYN', 'BOBRN', 'BOB', 'BRNA', 'BRN', 'BFNS', 'BOST', 'BLCA', 'BLCB', 'BLCS', 'BLLA', 'BLLB', 'BWTB', 'MBWT', 'DBKM', 'BVZI', 'BVZC', 'BOMN', 'BFAT', 'BRS', 'BFU', 'RORO', 'BOY', 'BV')
    
    # 1. First check if combining all adjacent tokens produces a valid composite wagon type (e.g. BCNA + HSM1 -> BCNAHSM1)
    combined_token = "".join(tokens).upper()
    for pfx in sorted(wagon_prefixes, key=len, reverse=True):
        if combined_token.startswith(pfx) and len(combined_token) <= 15:
            return combined_token

    # 2. Individual token check
    for token in tokens:
        upper_t = token.upper()
        for pfx in sorted(wagon_prefixes, key=len, reverse=True):
            if upper_t.startswith(pfx) and len(upper_t) <= 15:
                return upper_t
    
    # Last resort: return the cleaned text as-is (let the pipeline's clean_text_class handle it)
    return cleaned

@app.post("/ocr_vlm")
async def ocr_vlm_endpoint(file: UploadFile = File(...), category: str = Form("wagon_number")):
    if model is None or processor is None:
        return {"error": "VLM model not loaded", "text": ""}
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        prompt_text = PROMPTS.get(category, PROMPTS["wagon_number"])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=32)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        
        # Server-side response cleaning for wagon_type and zone categories
        if category in ("wagon_type", "owner_name", "zone_markings"):
            output_text = _strip_vlm_wrapper(output_text)
            
        return {"text": output_text, "category": category}
    except Exception as e:
        print(f"VLM OCR processing error: {e}")
        return {"error": str(e), "text": ""}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
