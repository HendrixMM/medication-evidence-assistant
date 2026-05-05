# Medical Disclaimers Reference

Guardrail responses must carry unambiguous medical disclaimers that align with FDA, EMA, and professional society expectations for investigational tools. The language below can be composed verbatim or templated, provided the intent and scope are preserved.

## Core Disclaimer Language

- **Research Context Only** – “This content supports pharmaceutical and translational research workflows and must not replace individualized clinical judgment or prescribing decisions by licensed professionals.”[[1]](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/medical-product-communications-guidance-industry)
- **Emergency Direction** – “For medical emergencies or urgent symptoms, contact local emergency services or seek in-person care immediately; this system cannot diagnose, triage, or respond to emergencies.”[[2]](https://www.ama-assn.org/system/files/ama-electronic-communication-guidelines.pdf)
- **Patient Variability** – “Drug response and safety vary by patient-specific factors (comorbidities, pharmacogenomics, renal/hepatic function, concomitant therapies); do not extrapolate findings to individual patients without clinician oversight.”[[3]](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers)
- **Evidence Status** – “Summaries may include emerging, preliminary, or non-peer-reviewed evidence; verify against primary literature and approved labeling before applying to practice.”[[1]](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/medical-product-communications-guidance-industry)

## Trigger-Based Augmentations

| Triggered Content                                                                              | Required Augmentation                                                                                                                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dosing, titration, or therapeutic drug monitoring                                              | Emphasize that dosing decisions require full patient data review and must be ordered or adjusted solely by credentialed prescribers.[[2]](https://www.ama-assn.org/system/files/ama-electronic-communication-guidelines.pdf)                                                |
| High-risk drug interactions (e.g., CYP3A4, QT prolongation, boxed warnings)                    | Highlight need for pharmacist/physician review, enhanced monitoring plans, and risk-benefit deliberation before therapy changes.[[3]](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers) |
| Off-label, investigational, emergency-use, or compassionate-use scenarios                      | Instruct users to confirm regulatory status, IRB approvals, and informed consent processes before proceeding.[[4]](https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-compassionate-use-medicines-human-use_en.pdf)                                      |
| Vulnerable populations (pediatrics, geriatrics, pregnancy/lactation, renal/hepatic impairment) | Note that population-specific evidence is required and specialist consultation is mandatory; extrapolation from adult data may be unsafe.[[5]](https://www.nichd.nih.gov/health/clinical-research/clinical-studies/vulnerable-populations)                                  |

## Implementation Guidance

1. Apply the core research disclaimer on every medical or pharmacological response, regardless of perceived risk.
2. Layer trigger-based language in the order of highest risk (investigational status → interaction severity → dosing → population-specific notes) to maintain clarity.
3. Do not duplicate disclaimers; reuse standardized paragraphs to preserve idempotency and auditability.
4. Reference complementary guardrails (`drug_interaction_guidelines.md`, `regulatory_compliance.md`) when interaction analysis, labeling status, or privacy obligations are present.

## References

1. U.S. Food and Drug Administration. _Medical Product Communications That Are Consistent With the FDA-Required Labeling._ 2018. Available at: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/medical-product-communications-guidance-industry
2. American Medical Association. _Guidelines for Physician-Patient Electronic Communication._ 2022. Available at: https://www.ama-assn.org/system/files/ama-electronic-communication-guidelines.pdf
3. U.S. Food and Drug Administration. _Drug Development and Drug Interactions: Table of Substrates, Inhibitors and Inducers._ 2024. Available at: https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers
4. European Medicines Agency. _Guideline on Compassionate Use of Medicines for Human Use._ 2021. Available at: https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-compassionate-use-medicines-human-use_en.pdf
5. Eunice Kennedy Shriver National Institute of Child Health and Human Development. _Research Involving Vulnerable Populations._ 2022. Available at: https://www.nichd.nih.gov/health/clinical-research/clinical-studies/vulnerable-populations
