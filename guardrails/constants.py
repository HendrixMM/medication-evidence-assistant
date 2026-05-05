"""
Unified medical disclaimer constants and detection patterns for RAG Template.

This module provides a single source of truth for disclaimer constants
across all components as per verification Comment 4.
"""
# Unified medical disclaimer constant
MEDICAL_DISCLAIMER = (
    "Medical Disclaimer: This information is for research and educational purposes only and is not intended as "
    "medical advice, diagnosis, or treatment. It is not a substitute for professional medical consultation. "
    "Always consult qualified healthcare professionals or licensed clinicians for any medical concerns or decisions. "
    "This system does not handle medical emergencies - seek immediate medical attention for urgent conditions. "
    "The information provided may contain inaccuracies and should be verified with authoritative medical sources."
)

# Markdown formatted version for bold formatting
MEDICAL_DISCLAIMER_MARKDOWN = (
    "**Medical Disclaimer:** This information is for research and educational purposes only and is not intended as "
    "medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions."
)

# Disclaimer detection patterns for standardized detection
DISCLAIMER_DETECTION_PATTERNS = [
    r"(?i)\*\*medical disclaimer:\*\*",  # Bold markdown format
    r"(?i)medical disclaimer:",  # Colon format
    r"(?i)medical disclaimer\b",  # Word boundary format
    r"(?i)this information is for.*educational purposes",  # Content-based detection
    r"(?i)consult.*healthcare.*professional",  # Professional consultation notice
    r"(?i)not intended as.*medical advice",  # Advice disclaimer
    r"(?i)not a substitute for.*medical consultation",  # Consultation disclaimer
]

# Specialized disclaimer types
DISCLAIMER_TYPES = {
    "general": MEDICAL_DISCLAIMER,
    "markdown": MEDICAL_DISCLAIMER_MARKDOWN,
    "drug_information": (
        "**Medical Disclaimer:** This drug information is for educational purposes only. "
        "Always consult your healthcare provider before starting, stopping, or changing any medication."
    ),
    "interactions": (
        "**Medical Disclaimer:** Drug interaction information is for educational purposes. "
        "Always inform your healthcare provider about all medications and supplements you are taking."
    ),
    "research": (
        "**Medical Disclaimer:** This research information is for educational purposes only. "
        "Results may not be applicable to individual cases. Consult healthcare professionals for personalized advice."
    ),
}
