"""Shared constants for patient query processing and LLM query generation."""
# DEPRECATED: replaced by src/agent/planner.py (LLM-driven, no hard-coded MeSH maps)
# This module is superseded by the agent layer introduced in Phase 4.
# It is retained for backward compatibility with Streamlit apps and existing tests.
# Do not add new functionality here. See src/agent/ for the active implementation.
# MeSH Subheading Mapping for Intent-Specific Retrieval
# Maps query intents to relevant MeSH subheadings for targeted article retrieval
MESH_SUBHEADING_MAP = {
    "side_effects": [
        "/adverse effects",
        "/toxicity",
        "/contraindications",
    ],
    "dosage": [
        "/administration and dosage",
        "/therapeutic use",
    ],
    "timing": [
        "/administration and dosage",
        "/pharmacokinetics",
    ],
    "drug_interaction": [
        "/pharmacology",
        "/adverse effects",
        "/pharmacokinetics",
    ],
    "mechanism": [
        "/pharmacology",
        "/pharmacokinetics",
        "/metabolism",
    ],
    "comparison": [
        "/therapeutic use",
        "/adverse effects",
    ],
}

# Fallback subheadings when intent not in map or for unknown intents
DEFAULT_SUBHEADINGS = ["/therapeutic use", "/adverse effects"]

# Brand name to generic name mapping for query normalization
# Extends the mapping from patient_entity_extractor.py for query building
ADDITIONAL_DRUG_SYNONYMS = {
    "zantac": "ranitidine",
    "pepcid": "famotidine",
    "zocor": "simvastatin",
    "plavix": "clopidogrel",
    "acetylsalicylic acid": "aspirin",
    "bayer": "aspirin",
    "excedrin": "aspirin",
    "mucinex": "guaifenesin",
    "robitussin": "guaifenesin",
    "sudafed": "pseudoephedrine",
    "claritin": "loratadine",
    "zyrtec": "cetirizine",
    "allegra": "fexofenadine",
    "benadryl": "diphenhydramine",
    "tums": "calcium carbonate",
    "maalox": "aluminum hydroxide",
    "mylanta": "aluminum hydroxide",
    "tagamet": "cimetidine",
    "prevacid": "lansoprazole",
    "protonix": "pantoprazole",
    "aciphex": "rabeprazole",
    "levoxyl": "levothyroxine",
    "crestor": "rosuvastatin",
    "pravachol": "pravastatin",
    "vytorin": "ezetimibe",
    "norvasc": "amlodipine",
    "cardizem": "diltiazem",
    "procardia": "nifedipine",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "tenormin": "atenolol",
    "coreg": "carvedilol",
    "diovan": "valsartan",
    "cozaar": "losartan",
    "avapro": "irbesartan",
    "benicar": "olmesartan",
    "lasix": "furosemide",
    "microzide": "hydrochlorothiazide",
    "aldactone": "spironolactone",
    "glucotrol": "glipizide",
    "amaryl": "glimepiride",
    "actos": "pioglitazone",
    "avandia": "rosiglitazone",
    "januvia": "sitagliptin",
    "victoza": "liraglutide",
    "lantus": "insulin glargine",
    "humalog": "insulin lispro",
    "novolog": "insulin aspart",
    "paracetamol": "acetaminophen",
    "blood thinner": "anticoagulants",
    "blood thinners": "anticoagulants",
    "anticoagulant": "anticoagulants",
    "anticoagulants": "anticoagulants",
    "birth control": "contraceptive agents",
    "contraceptive": "contraceptive agents",
    "contraceptives": "contraceptive agents",
    "oral contraceptive": "contraceptive agents",
    "oral contraceptives": "contraceptive agents",
    "oral contraceptive pill": "contraceptive agents",
    "oral contraceptive pills": "contraceptive agents",
    "antibiotic": "anti-bacterial agents",
    "antibiotics": "anti-bacterial agents",
    "antibacterial": "anti-bacterial agents",
    "anti-bacterial": "anti-bacterial agents",
    "nsaid": "anti-inflammatory agents, non-steroidal",
    "nsaids": "anti-inflammatory agents, non-steroidal",
    "nonsteroidal anti-inflammatory drug": "anti-inflammatory agents, non-steroidal",
    "nonsteroidal anti-inflammatory drugs": "anti-inflammatory agents, non-steroidal",
    "blood pressure medication": "antihypertensive agents",
    "blood pressure medications": "antihypertensive agents",
    "blood pressure medicine": "antihypertensive agents",
    "blood pressure meds": "antihypertensive agents",
    "bp medication": "antihypertensive agents",
    "bp meds": "antihypertensive agents",
}
