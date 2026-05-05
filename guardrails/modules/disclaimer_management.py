"""
Medical disclaimer management utilities.

Provides standardized disclaimer detection, insertion, and formatting
for pharmaceutical safety compliance.
"""
import re
from typing import Optional

from ..constants import DISCLAIMER_DETECTION_PATTERNS
from ..constants import MEDICAL_DISCLAIMER_MARKDOWN as STANDARD_MEDICAL_DISCLAIMER

# Import unified disclaimer constants


def contains_medical_disclaimer(text: Optional[str]) -> bool:
    """
    Standardized disclaimer detection across all components.

    Args:
        text: Text to check for medical disclaimers

    Returns:
        True if text contains a medical disclaimer, False otherwise
    """
    if not text:
        return False

    for pattern in DISCLAIMER_DETECTION_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


def insert_medical_disclaimer(text: str, disclaimer_type: str = "general") -> str:
    """
    Insert standardized medical disclaimer into text.

    Args:
        text: Text to add disclaimer to
        disclaimer_type: Type of disclaimer (general, drug_information, etc.)

    Returns:
        Text with disclaimer appended
    """
    if contains_medical_disclaimer(text):
        return text  # Disclaimer already present

    # Use standard disclaimer format
    disclaimer = STANDARD_MEDICAL_DISCLAIMER

    # Add type-specific disclaimers if needed
    if disclaimer_type == "drug_information":
        disclaimer = (
            "**Medical Disclaimer:** This drug information is for educational purposes only. "
            "Always consult your healthcare provider before starting, stopping, or changing any medication."
        )
    elif disclaimer_type == "interactions":
        disclaimer = (
            "**Medical Disclaimer:** Drug interaction information is for educational purposes. "
            "Always inform your healthcare provider about all medications and supplements you are taking."
        )

    return f"{text}\n\n{disclaimer}"


def format_disclaimer_consistently(text: str) -> str:
    """
    Ensure disclaimer follows consistent formatting (bold markdown).

    Args:
        text: Text potentially containing disclaimers

    Returns:
        Text with consistently formatted disclaimers
    """
    # Replace various disclaimer formats with standard format
    replacements = [
        (r"(?i)medical disclaimer:\s*", "**Medical Disclaimer:** "),
        (r"(?i)\*medical disclaimer:\*", "**Medical Disclaimer:**"),
        (r"(?i)medical disclaimer\s*-\s*", "**Medical Disclaimer:** "),
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    return result
