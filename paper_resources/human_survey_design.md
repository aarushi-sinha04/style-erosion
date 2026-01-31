# Human Evaluation Study Design
**Objective**: To validate if Adversarial Attacks (T5 Paraphrasing) preserve meaning and fluency better than naive attacks, and if they fool humans as well as models.

## 1. Study Setup
- **Platform**: Prolific (Targeting English Native Speakers).
- **Sample Size**: 50 Participants x 20 Pairs = 1000 annotations.
- **Cost Estimate**: ~£150 (based on £9/hr).

## 2. Task Design
Participants will be presented with pairs of texts:
- **Original Text (A)**
- **Transformed Text (B)**

**Questions**:
1.  **Semantic Preservation (Likert 1-5)**:
    -   *How much of the original meaning is preserved in Text B?*
    -   1: Completely different meaning.
    -   5: Identical meaning.
2.  **Fluency (Likert 1-5)**:
    -   *Is Text B grammatically correct and natural?*
    -   1: Gibberish.
    -   5: Perfect English.
3.  **Detectability (Binary)**:
    -   *Do you think Text B was written by a machine?*
    -   [ ] Yes
    -   [ ] No

## 3. Conditions
We will compare 3 conditions (Blind):
1.  **Control**: Original Text vs. Human Rewrite (Gold Standard).
2.  **Naive Attack**: Original Text vs. Synonym Replacement (Baseline).
3.  **Proposed Attack**: Original Text vs. T5 Paraphraser (Phase 3 Method).

## 4. Hypothesis
-   **Semantic Preservation**: T5 > Synonym (p < 0.05).
-   **Fluency**: T5 > Synonym (p < 0.05).
-   **AI Detection**: Humans will detect T5 less often than Synonym attacks.

## 5. Output for Paper
-   Table comparing Mean Opinion Scores (MOS).
-   Bar chart of "AI Detection Rate".
