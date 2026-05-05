# Drug Interaction Safety Guidelines

Use this playbook when synthesizing or validating interaction findings across drug-drug, drug-food, or drug-device combinations. The goal is to surface clinically relevant risk while deferring prescribing decisions to human experts.

## Interaction Assessment Workflow

1. **Normalize Interacting Entities** – Resolve brand/generic names, biologics, and nutraceuticals to RxNorm, WHO ATC, or EMA Common Terminology to anchor downstream lookups.[[1]](https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-investigation-drug-interactions_en.pdf)
2. **Mechanistic Classification** – Identify pharmacokinetic mechanisms (CYP450, UGT, transporter modulation) and pharmacodynamic synergies/antagonisms. Flag CYP3A4, CYP2D6, OATP1B1, and P-glycoprotein events for heightened review.[[2]](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers)
3. **Severity and Evidence Attribution** – Assign severity using the standardized scale below and append the highest-quality evidence tier (clinical trial, observational, case report, in vitro) to support transparency.[[3]](https://www.lexicomp.com/lco/librarian)
4. **Clinical Consequence Summary** – Describe expected patient-level impacts (e.g., QT prolongation, serotonin syndrome, graft rejection risk) with onset timing where known.[[4]](https://pubmed.ncbi.nlm.nih.gov/?term=drug+interaction+monitoring)
5. **Monitoring and Mitigation Signals** – Highlight monitoring labs, ECGs, serum drug levels, or dose titration frameworks that clinicians routinely consider, while reiterating that prescribing decisions require licensed oversight.[[3]](https://www.lexicomp.com/lco/librarian)

## Standardized Severity Scale

| Severity        | Definition                                                                               | Typical Guidance                                                                                                                                           |
| --------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contraindicated | Documented harm outweighs benefit; concurrent use prohibited in labeling or guidelines.  | Recommend alternative therapy and state that the combination is contraindicated pending specialist review.                                                 |
| Major           | Life-threatening or permanent damage is possible; requires urgent clinical intervention. | Advise immediate prescriber review, strong monitoring (e.g., telemetry, INR q24h), and consider temporary therapy hold or substitution.                    |
| Moderate        | Exacerbation of condition or significant change in therapeutic response is likely.       | Suggest considering dose adjustments, spacing doses, or implementing targeted monitoring (serum levels, organ function tests) under clinician supervision. |
| Minor           | Limited clinical relevance; minimal risk or theoretical basis only.                      | Document the interaction for completeness and note that routine monitoring is generally sufficient.                                                        |

## Monitoring and Dose-Adjustment Framework

- **Laboratory Monitoring** – Surface lab strategies aligned with interaction class (e.g., INR for warfarin, trough levels for calcineurin inhibitors, QTc for torsadogenic pairs).[[2]](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers)
- **Dose Modification Signals** – When evidence shows ≥30% AUC/Cmax changes, recommend that clinicians consider titration, temporary interruption, or switching to an alternative agent, while explicitly refusing to calculate individualized dosing.[[1]](https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-investigation-drug-interactions_en.pdf)
- **Duration and Onset Notes** – Capture whether interaction risk is immediate (e.g., CYP inhibition) or delayed (enzyme induction) so clinicians can time monitoring appropriately.[[4]](https://pubmed.ncbi.nlm.nih.gov/?term=drug+interaction+monitoring)
- **Population Sensitivity** – Flag populations with altered pharmacokinetics (renal/hepatic impairment, pediatrics, pregnancy) and cross-reference `medical_disclaimers.md` for tailored language.[[5]](https://ashpublications.org/book/content/1992632/Drug-Drug-Interactions-in-Oncology)

## Guardrail Enforcement Rules

- Refuse user requests to start, stop, or adjust therapies directly; route back to educating on documented risks.
- Escalate warnings for boxed warning agents, drugs with narrow therapeutic indices (warfarin, tacrolimus, theophylline), and biologics with immunogenicity concerns.
- Distinguish between evidence-backed interactions and theoretical signals; note “data limited” when only in vitro or case reports support the interaction.
- Append monitoring/dose-adjustment language only once per response, leveraging the disclaimer idempotency guard.

## Recommended Data Sources

- FDA Drug Interaction Database, labeling sections 7 (Drug Interactions) and 12 (Clinical Pharmacology).[[2]](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers)
- EMA _Guideline on the Investigation of Drug Interactions_ for European regulatory expectations.[[1]](https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-investigation-drug-interactions_en.pdf)
- Clinical pharmacology compendia (Lexicomp, Micromedex, AHFS) for continuously curated severity ratings and management recommendations.[[3]](https://www.lexicomp.com/lco/librarian)
- Peer-reviewed literature tracked via PubMed, especially for oncology, transplant, and anti-infective regimens where new interactions emerge frequently.[[4]](https://pubmed.ncbi.nlm.nih.gov/?term=drug+interaction+monitoring)

## References

1. European Medicines Agency. _Guideline on the Investigation of Drug Interactions._ 2023. Available at: https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-investigation-drug-interactions_en.pdf
2. U.S. Food and Drug Administration. _Drug Development and Drug Interactions: Regulatory Guidance._ 2024. Available at: https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers
3. Wolters Kluwer. _Lexicomp Online: Drug Interaction Analysis._ Accessed 2024. Available at: https://www.lexicomp.com/lco/librarian
4. National Library of Medicine. _PubMed Clinical Queries – Drug Interaction Monitoring._ Accessed 2024. Available at: https://pubmed.ncbi.nlm.nih.gov/?term=drug+interaction+monitoring
5. American Society of Clinical Oncology. _Drug Interactions in Oncology Practice._ 2022. Available at: https://ashpublications.org/book/content/1992632/Drug-Drug-Interactions-in-Oncology
