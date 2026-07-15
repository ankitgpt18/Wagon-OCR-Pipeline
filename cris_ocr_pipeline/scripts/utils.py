#!/usr/bin/env python3
"""
Utility functions for Wagon OCR Pipeline
- Wagon number checksum validation
- Text cleaning and normalization
"""
import re


def clean_wagon_number(text):
    """Clean and normalize wagon number text"""
    # Remove non-digit characters
    cleaned = re.sub(r"\D", "", text)
    return cleaned


def validate_wagon_number(wagon_number):
    """
    Validate Indian Railways wagon number using checksum algorithm
    
    Indian Railways uses a 11-digit wagon number with a check digit
    The algorithm is similar to Luhn algorithm but with specific weights
    """
    # Clean the number first
    cleaned = clean_wagon_number(wagon_number)
    
    if len(cleaned) != 11:
        return False, "Wagon number must be exactly 11 digits"
    
    digits = [int(d) for d in cleaned]
    
    # Checksum calculation (Indian Railways standard)
    # Weights for positions 1-10 (from left to right): 1, 2, 1, 2, 1, 2, 1, 2, 1, 2
    weights = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    total = 0
    
    for i in range(10):
        product = digits[i] * weights[i]
        # Sum the digits of the product (e.g., 14 -> 1 + 4 = 5)
        total += sum(int(d) for d in str(product))
    
    # Calculate check digit
    check_digit = (10 - (total % 10)) % 10
    
    is_valid = check_digit == digits[10]
    
    if is_valid:
        return True, f"Valid wagon number (check digit: {check_digit})"
    else:
        return False, f"Invalid check digit: expected {check_digit}, got {digits[10]}"


def extract_wagon_numbers(text):
    """Extract potential wagon numbers from text using regex"""
    # Look for sequences of 8-15 digits (common variations)
    matches = re.findall(r"\d{8,15}", text)
    
    valid_numbers = []
    for match in matches:
        is_valid, message = validate_wagon_number(match)
        if is_valid:
            valid_numbers.append(match)
        else:
            # Also check if we can truncate or pad to 11 digits
            cleaned = clean_wagon_number(match)
            if len(cleaned) >= 11:
                # Try first 11 digits
                candidate = cleaned[:11]
                is_valid, _ = validate_wagon_number(candidate)
                if is_valid:
                    valid_numbers.append(candidate)
                # Try last 11 digits
                candidate = cleaned[-11:]
                is_valid, _ = validate_wagon_number(candidate)
                if is_valid and candidate not in valid_numbers:
                    valid_numbers.append(candidate)
    
    return valid_numbers


def clean_text(text):
    """General text cleaning for OCR output"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Common OCR fixes for wagon numbers
    # Replace 'O' with '0', 'I' with '1', 'l' with '1'
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    
    return text


if __name__ == "__main__":
    # Test the functions
    test_cases = [
        "12345678901",  # Example (not a real valid number)
        "10002480190",
        "BOXN 12345678901 WR",
        "OCR might read 12345678901 as I234567890I"
    ]
    
    print("Testing wagon number utilities...")
    for test in test_cases:
        print(f"\nInput: {test}")
        cleaned = clean_text(test)
        print(f"  Cleaned: {cleaned}")
        
        numbers = extract_wagon_numbers(cleaned)
        print(f"  Extracted numbers: {numbers}")
        
        for num in numbers:
            is_valid, msg = validate_wagon_number(num)
            print(f"  {num}: {msg}")
