# Regulatory Compliance Checklist for Generated Medical Content

Responses emitted by the system must adhere to U.S. FDA, EMA, and privacy regulations governing scientific communications. Use this checklist to encode compliance behaviors directly into rail logic and content generation.

## Core Compliance Domains

- **Label Adherence** – Align efficacy, safety, and dosing statements with FDA/EMA-approved labeling. Clearly mark off-label or investigational use and cite supporting evidence tiers.[[1]](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/medical-product-communications-guidance-industry)
- **Risk Communication** – Reproduce boxed warnings, contraindications, REMS requirements, and risk mitigation steps without dilution when those topics arise.[[2]](https://www.ema.europa.eu/en/documents/scientific-guideline/gvp-module-xv-safety-communication_en.pdf)
- **Promotional Neutrality** – Avoid marketing tone and superiority claims unless backed by head-to-head randomized evidence and framed as research synthesis only.[[3]](https://www.ifpma.org/wp-content/uploads/2021/01/IFPMA-Code-of-Practice-2021.pdf)
- **Patient Privacy** – Remove or mask PHI per HIPAA Privacy Rule and GDPR principles before logging, caching, or displaying case details.[[4]](https://www.hhs.gov/hipaa/for-professionals/privacy/index.html)
- **Adverse Event Handling** – Direct users to FDA MedWatch or EMA EudraVigilance when serious adverse events are referenced, and discourage user submission of case-level identifiers in chat.[[5]](https://www.who.int/publications/i/item/9789240029202)

## Regional Considerations

| Region                 | Key Statutes/Guidance                                                                               | Guardrail Notes                                                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| United States          | 21 CFR Part 202 (Prescription Drug Advertising); FDA Medical Product Communications Guidance (2018) | Distinguish investigational research narratives from approved labeling; cite data sources and study design when discussing comparative outcomes. |
| European Union         | EMA Code of Practice on the Promotion of Medicinal Products (2021); GVP Module XV                   | Mirror approved SmPC language for benefit-risk statements and flag when data have not been validated by EMA or national competent authorities.   |
| Privacy (Global)       | HIPAA Privacy Rule; GDPR Articles 4, 32; ISO/IEC 27701                                              | Avoid storing or reproducing direct identifiers. Apply automated masking actions documented in `medical_disclaimers.md` prior to persistence.    |
| International Research | ICH E6(R3) Good Clinical Practice Draft; WHO Ethical Criteria for Medicinal Drug Promotion          | Note when multinational trial results may not translate to local approvals; prompt users to consult local regulatory guidance.                   |

## Automation Hooks

1. **Compliance Keyword Scan** – Trigger additional review when phrases such as “guaranteed cure,” “miracle treatment,” or “zero side effects” appear; downgrade response confidence or refuse if marketing tone persists.[[3]](https://www.ifpma.org/wp-content/uploads/2021/01/IFPMA-Code-of-Practice-2021.pdf)
2. **Approval Status Resolver** – Crosswalk drug mentions to regulatory status metadata (FDA NDA/BLA, EMA MA, emergency use) and annotate when approval is pending or limited.
3. **Population-Specific Flags** – Enforce extra disclaimers when discussing pediatrics, pregnancy, geriatrics, renal/hepatic impairment, or pharmacogenomic subgroups, referencing `medical_disclaimers.md` triggers.
4. **Documentation Trail** – Log the final disclaimer, compliance decisions, and detected risks to support post-hoc audits and CAPA workflows.

## References

1. U.S. Food and Drug Administration. _Medical Product Communications That Are Consistent With the FDA-Required Labeling._ 2018. Available at: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/medical-product-communications-guidance-industry
2. European Medicines Agency. _Guideline on Good Pharmacovigilance Practices (GVP) Module XV – Safety Communication._ 2019. Available at: https://www.ema.europa.eu/en/documents/scientific-guideline/gvp-module-xv-safety-communication_en.pdf
3. International Federation of Pharmaceutical Manufacturers & Associations. _IFPMA Code of Practice._ 2021 Edition. Available at: https://www.ifpma.org/wp-content/uploads/2021/01/IFPMA-Code-of-Practice-2021.pdf
4. U.S. Department of Health & Human Services. _HIPAA Privacy Rule Summary._ Updated 2023. Available at: https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
5. World Health Organization. _Reporting and Learning Systems for Adverse Events._ 2022. Available at: https://www.who.int/publications/i/item/9789240029202
