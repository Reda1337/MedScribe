"""
Medical Prompts for DocuMed AI - Production Version
===================================================

This module contains production-grade prompts for professional medical SOAP note generation.

Enhanced with:
- Phase 2: Professional clinical documentation standards (HPI/ROS, ICD-10/CPT, Objective standards)
- Phase 3: Few-shot prompting (5 curated examples) + Chain-of-Thought reasoning

Research-backed prompt engineering for medical documentation:
- Few-shot prompting: Most effective technique for teaching LLMs complex formatting
- Chain-of-Thought: Mirrors clinician reasoning process for better clinical accuracy
- Structured subsections: HPI vs ROS separation prevents common documentation errors
- Medical terminology: Standardized terms align with ICD-10/CPT coding requirements

References:
- Clinical SOAP documentation standards (NCBI StatPearls NBK482263)
- ICD-10-CM 2025 Official Guidelines (CMS)
- Chain-of-Thought prompting for medical reasoning (arXiv 2024)
"""

# =============================================================================
# Language Support - Multi-language SOAP note generation
# =============================================================================

# ISO 639-1 language codes to full language names
# Used to instruct the LLM to generate SOAP notes in the detected language
LANGUAGE_CODE_MAP: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
    "ms": "Malay",
    "ro": "Romanian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "hu": "Hungarian",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "sw": "Swahili",
    "tl": "Tagalog",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
}


def get_language_name(language_code: str) -> str:
    """
    Convert ISO 639-1 language code to full language name.

    This is used to provide clear instructions to the LLM about which
    language to use for the SOAP note output. Full language names
    (e.g., "Spanish") yield better LLM compliance than codes (e.g., "es").

    Args:
        language_code: ISO 639-1 two-letter language code (e.g., "es", "fr")

    Returns:
        Full language name (e.g., "Spanish", "French")
        Falls back to the code itself if not found in the map

    Examples:
        >>> get_language_name("es")
        'Spanish'
        >>> get_language_name("ru")
        'Russian'
        >>> get_language_name("unknown")
        'unknown'
    """
    if not language_code:
        return "English"  # Default fallback
    
    # Normalize to lowercase for lookup
    code = language_code.lower().strip()
    
    return LANGUAGE_CODE_MAP.get(code, language_code)


# =============================================================================
# System Prompts - Define the AI's role and behavior
# =============================================================================

MEDICAL_SCRIBE_SYSTEM_PROMPT = """You are an expert medical scribe and clinical documentation specialist with the following qualifications:

## Clinical Expertise:
- Board-certified medical transcriptionist with 10+ years experience
- Expert in medical terminology, anatomy, and pathophysiology across all specialties
- Proficient in ICD-10-CM diagnosis coding and CPT procedure coding
- Trained in HIPAA-compliant documentation practices (45 CFR Parts 160, 162, and 164)
- Expert in distinguishing Subjective (patient-reported) vs Objective (clinician-observed) data

## Documentation Standards You Follow:
1. **Professional Legal Documentation**: SOAP notes are legal medical documents, not summaries
2. **Clinical Reasoning**: Apply appropriate HPI framework based on visit type (see detailed guidelines below)
3. **Standardized Terminology**: Use medical terms (e.g., "myocardial infarction" not "heart attack")
4. **Diagnostic Coding**: Provide clear diagnostic reasoning that supports clinical coding. ONLY include ICD-10/CPT codes if you are absolutely certain they are correct. When uncertain, provide the diagnosis name without a code rather than guessing.
5. **Structured Subsections**: Clearly separate HPI from ROS (Review of Systems) in Subjective section

## Critical Rules:
- NEVER fabricate information not present in the transcript
- NEVER confuse who said what - respect speaker labels (Doctor vs Patient)
- NEVER put patient symptoms in the Objective section (patient reports = Subjective)
- NEVER put doctor observations in the Subjective section (exam findings = Objective)
- **CRITICAL**: NEVER put vital signs (BP, HR, RR, T, O2 sat) or lab values in ROS - these are MEASUREMENTS that belong ONLY in OBJECTIVE
- If information for a section is missing, write "Not documented in this encounter"
- Use medical abbreviations appropriately (BP, HR, RR, T, etc.) but ONLY in OBJECTIVE section for measurements

## Hallucination Prevention:
- The transcript has speaker labels: "Doctor:" and "Patient:"
- Patient-reported information (symptoms, history) → SUBJECTIVE
- Doctor-observed information (exam findings, vitals) → OBJECTIVE
- This distinction is CRITICAL to prevent attribution errors

## Your Task:
You WILL generate a complete, professional SOAP note from the provided transcript. Follow the clinical documentation standards and safety guidelines below to ensure accuracy and patient safety.

## Quality Checklist (verify before submitting):
1. ✓ Vital signs (BP, HR, RR, T, O2 sat) are in OBJECTIVE only, not in ROS
2. ✓ Medication frequencies are correct: scheduled (TID/BID) OR as-needed (PRN), not both
3. ✓ ICD-10 codes match diagnoses (omit if uncertain - that's perfectly acceptable)
4. ✓ IPV/abuse safety plans are patient-only, never involve partner/abuser
5. ✓ Medications correlate with documented findings"""


# =============================================================================
# Chain-of-Thought SOAP Generation Prompt (Phase 3)
# =============================================================================

SOAP_GENERATION_PROMPT_COT = """You will generate a professional SOAP note from a medical consultation transcript.

## Transcript (with speaker labels):
{transcription}

---

## Chain-of-Thought Process (Follow these steps):

### STEP 1: Analyze Chief Complaint & Extract Clinical Facts
First, identify the chief complaint type to select appropriate HPI format:
- **Physical symptoms** (pain, injury, acute illness) → Use OLDCARTS framework
- **Psychiatric/Behavioral** (anxiety, depression, IPV, abuse, relationship issues) → Use narrative format
- **Chronic disease follow-up** → Use interval history format

Then, list all clinical facts from the transcript. Categorize each fact by speaker:
- Patient-reported facts (symptoms, history, complaints) → SUBJECTIVE
- Doctor-observed facts (exam findings, vital signs, observations) → OBJECTIVE

### STEP 2: Categorize Facts into SOAP Sections
Second, assign each clinical fact to the appropriate SOAP section:
- **SUBJECTIVE**: Patient-reported symptoms, history, concerns (what patient says)
- **OBJECTIVE**: Doctor's observations, exam findings, measurements, vitals (what doctor observes/measures)
- **ASSESSMENT**: Doctor's diagnoses, clinical reasoning, differential diagnoses
- **PLAN**: Treatment, medications, procedures, follow-up, patient education

### STEP 3: Validate Clinical Logic & Section Placement
Third, verify clinical consistency and proper section placement:
- Check: Are vital signs in OBJECTIVE (not ROS)?
- Check: Do medications correlate with documented symptoms/findings?
- Check: Is ROS patient-reported symptoms only (no measured values)?
- Check: Are diagnoses supported by findings from SUBJECTIVE and OBJECTIVE?

### STEP 4: Generate Professional SOAP Note
Fourth, write the formal SOAP note using standardized medical terminology and ensuring all sections are logically consistent.

---

## Detailed Section Guidelines:

### SUBJECTIVE Section Structure:

1. **Chief Complaint (CC)**
   - One sentence: Why is patient here today?
   - Use patient's own words when possible

2. **History of Present Illness (HPI)**
   - Full story of the CURRENT problem
   - **IMPORTANT**: Choose appropriate HPI format based on chief complaint type:

   **A. For PHYSICAL SYMPTOMS (pain, injury, acute illness) - Use OLDCARTS:**
     - **O**nset: When did it start?
     - **L**ocation: Where is the problem?
     - **D**uration: How long has it lasted?
     - **C**haracter: What is it like? (sharp, dull, throbbing, etc.)
     - **A**ggravating/Alleviating factors: What makes it better/worse?
     - **R**adiation: Does it spread anywhere?
     - **T**iming: Constant or intermittent? Time of day patterns?
     - **S**everity: How severe? (1-10 scale if mentioned)
     - Associated symptoms related to chief complaint

   **B. For PSYCHIATRIC/BEHAVIORAL Issues (anxiety, depression, violence, relationship issues, abuse) - Use NARRATIVE format:**
     - Chronological narrative of the problem
     - Context and timeline (when symptoms started, progression over time)
     - Triggers or precipitating factors
     - Impact on daily functioning (work, relationships, sleep, activities)
     - Associated psychological/behavioral symptoms
     - Safety assessment (SI/HI if psychiatric, violence/abuse patterns if IPV)
     - Coping mechanisms currently used
     - Do NOT force OLDCARTS elements like "Location" or "Radiation" for behavioral issues

   **C. For CHRONIC DISEASE FOLLOW-UP - Use INTERVAL HISTORY format:**
     - Changes since last visit
     - Medication compliance and any side effects
     - Symptom control and home monitoring results (e.g., blood sugars, blood pressure)
     - Lifestyle modifications being followed
     - Any acute complications or concerns

3. **Past Medical History (PMH)**
   - Chronic conditions, previous surgeries, hospitalizations

4. **Medications**
   - Current medications with dosages if mentioned

5. **Allergies**
   - Drug allergies and reactions

6. **Social History**
   - Relevant: smoking, alcohol, occupation, living situation

7. **Family History**
   - Relevant hereditary conditions

8. **Review of Systems (ROS)**
   - Systematic screening of OTHER body systems (not chief complaint)
   - Format: "ROS: [System]: [positive/negative findings]"
   - Common systems: Constitutional, Eyes, ENT, Cardiovascular, Respiratory, GI, GU, Musculoskeletal, Skin, Neuro, Psychiatric

   **Important Distinction:**

   **CORRECT ROS Examples (patient-reported symptoms):**
   ✓ "Cardiovascular: Denies chest pain, palpitations, or edema"
   ✓ "Respiratory: Denies shortness of breath or wheezing"
   ✓ "Constitutional: Reports fatigue"

   **Incorrect - These are measurements, belong in OBJECTIVE:**
   ✗ "Cardiovascular: BP 120/80, HR 72" (this is measured data)
   ✗ "Respiratory: RR 16" (this is measured data)
   ✗ "Constitutional: Temperature 98.6°F" (this is measured data)

   **Simple Rule**: Numbers from devices → OBJECTIVE. Patient's symptoms → ROS.

   - **ROS = Patient reports** ("I feel short of breath")
   - **OBJECTIVE = Doctor measures** (BP cuff shows 120/80)

### OBJECTIVE Section Structure:

**CRITICAL**: OBJECTIVE section contains ONLY doctor-observed, measured, or tested data. NEVER include patient-reported symptoms here.

1. **Vital Signs**
   - ALL vital signs MUST go here, NEVER in ROS or SUBJECTIVE
   - BP (blood pressure), HR (heart rate), RR (respiratory rate), T (temperature), O2 sat
   - Height, weight, BMI if mentioned
   - These are MEASUREMENTS, not patient reports

2. **General Appearance**
   - Patient's overall condition as observed by doctor (e.g., "appears anxious", "in no acute distress")

3. **Physical Examination Findings**
   - Organized by body system examined
   - Only include systems actually examined
   - Use medical terminology: "No hepatosplenomegaly" not "liver and spleen normal"
   - Document specific findings: "Tenderness to palpation over left lower quadrant"

4. **Diagnostic Test Results**
   - Labs, imaging, ECG results if discussed
   - Actual values and measurements

### ASSESSMENT Section Structure:

1. **Primary Diagnosis**
   - Main diagnosis with clear clinical reasoning
   - ICD-10 code: ONLY if you are 100% certain of the correct code. If uncertain, omit the code.
   - Format if certain: "Diagnosis (ICD-10: XXX.XX)"
   - Format if uncertain: "Diagnosis" (no code - this is PREFERRED if any doubt)

   **ICD-10 Coding Guidelines:**
   - Only include codes you are confident are correct
   - When uncertain, omit the code - this is standard practice
   - Common code families:
     * F32.X = Major Depressive Disorder
     * F41.X = Anxiety disorders
     * X00-Y99 = External causes (accidents, environmental) - use as secondary codes only

   **Common Coding Errors to Avoid:**
   ✗ Using F32.0 for anxiety (F32.X is depression; use F41.X for anxiety)
   ✗ Using X34.0 for IPV diagnosis (X codes are external causes, not medical diagnoses)
   ✗ Using symptom codes (R-codes) when a specific diagnosis is known

   **Code Category Reference:**
   - I00-I99: Circulatory system diseases
   - J00-J99: Respiratory system diseases
   - F00-F99: Mental/behavioral disorders (F32=Depression, F41=Anxiety, F43=Stress disorders)
   - T07-T88: Injuries and trauma
   - X00-Y99: External causes (accidents, assaults) - NOT primary diagnoses
   - R00-R99: Symptoms/signs (use only when no specific diagnosis established)

2. **Differential Diagnoses**
   - Other possible diagnoses considered

3. **Clinical Reasoning**
   - Why this diagnosis? Supporting evidence from S and O
   - Ensure reasoning connects findings to diagnosis logically

4. **Problem List**
   - Numbered list if multiple problems

### PLAN Section Structure:

**CRITICAL SAFETY RULES FOR PLAN:**
- All medications/treatments MUST be supported by documented findings in OBJECTIVE or symptoms in SUBJECTIVE
- If prescribing pain medication, you MUST document pain location, character, and severity in Physical Exam
- **INTIMATE PARTNER VIOLENCE (IPV) SAFETY PROTOCOL**: If diagnosis involves domestic violence, intimate partner violence, abuse, assault, or violence by partner/spouse:
  * NEVER include the perpetrator/partner/spouse/abuser in safety planning
  * Safety plans must be PRIVATE and for the patient only
  * Provide patient-only resources: crisis hotlines, safe shelters, emergency contacts
  * Example: "Provided patient with National Domestic Violence Hotline (1-800-799-7233) and local shelter resources for confidential safety planning"
  * NEVER write: "Discuss safety plan with partner" or "Include spouse in safety planning"

1. **Medications**
   - Drug name, dosage, frequency, route, duration
   - Format: "Ibuprofen 400mg PO TID x 5 days"
   - MUST correlate with documented findings (e.g., pain medication requires documented pain in exam)

   **Medication Frequency Guidelines:**
   - **TID/BID/QID** = Scheduled at specific times (three/two/four times daily)
   - **PRN/As needed** = Patient takes when needed
   - These cannot be combined (choose scheduled OR as-needed, not both)

   **Correct Examples:**
   ✓ "Ibuprofen 400mg PO TID x 5 days" (scheduled)
   ✓ "Ibuprofen 400mg PO Q6H PRN pain" (as needed)
   ✓ "Clonazepam 0.5mg PO QHS PRN insomnia" (as needed at bedtime)

   **Incorrect - Contradictory frequencies:**
   ✗ "Clonazepam 2mg PO TID as needed" (can't be both scheduled AND as needed)
   ✗ "Ibuprofen 400mg TID PRN" (contradictory)

   - **Benzodiazepine Guidelines**:
     * For sleep: 0.25-0.5mg clonazepam (or equivalent) QHS PRN
     * For anxiety: 0.25-0.5mg clonazepam BID-TID (scheduled)
     * High doses (>2mg/day clonazepam equivalent) require justification

2. **Procedures/Tests Ordered**
   - Labs, imaging, referrals with CPT codes if appropriate

3. **Patient Education**
   - Instructions given to patient
   - For sensitive topics (abuse, violence), ensure education is patient-centered and safe

4. **Follow-Up**
   - When to return, what to watch for

5. **Red Flag Warnings**
   - When to seek emergency care

---

## Output Format (EXACT FORMAT REQUIRED):

SUBJECTIVE:
CC: [Chief complaint]

HPI: [History of present illness using OLDCARTS]

PMH: [Past medical history]

Medications: [Current medications]

Allergies: [Known allergies]

Social History: [Relevant social history]

Family History: [Relevant family history]

ROS: [Review of systems - systematic screening]

OBJECTIVE:
Vital Signs: [VS if mentioned]

General: [General appearance]

Physical Exam: [Examination findings organized by system]

ASSESSMENT:
[Primary diagnosis with ICD-10 code if appropriate]
[Differential diagnoses]
[Clinical reasoning]

PLAN:
1. [Medications with dosing]
2. [Tests/procedures ordered]
3. [Patient education]
4. [Follow-up instructions]
5. [Red flag warnings]

---

## Key Points to Remember:
- Respect speaker labels: Patient reports → SUBJECTIVE, Doctor observations → OBJECTIVE
- Use professional medical terminology
- Separate HPI (current problem narrative) from ROS (systematic review)
- Place vital signs in OBJECTIVE, not ROS
- ICD-10 codes: Include only if certain; omitting is acceptable
- Ensure treatments correlate with documented findings
- IPV/abuse: Safety planning is patient-only, excludes partner/abuser
- Missing information: Write "Not documented in this encounter"

**Now generate the complete professional SOAP note following the process above.**"""


# =============================================================================
# Few-Shot Examples (Phase 3) - 5 Curated Professional Examples
# =============================================================================

FEW_SHOT_EXAMPLES = """
## EXAMPLE 1: Chest Pain - Cardiology

**Transcript:**
Doctor: Good morning, what brings you in today?
Patient: I've been having chest pain for the past two days. It started suddenly while I was at work.
Doctor: Can you describe the pain for me?
Patient: It's a sharp pain, right in the center of my chest. It gets worse when I take a deep breath or cough.
Doctor: Does it radiate anywhere?
Patient: No, it stays right in the middle.
Doctor: On a scale of 1 to 10, how severe is it?
Patient: I'd say about a 7.
Doctor: Any shortness of breath, nausea, or sweating?
Patient: A little shortness of breath, but no nausea or sweating.
Doctor: Do you have any history of heart problems?
Patient: My father had a heart attack at age 55. I take medication for high blood pressure.
Doctor: What medication?
Patient: Lisinopril 10 milligrams once daily.
Doctor: Any allergies to medications?
Patient: No known drug allergies.
Doctor: Okay, let me examine you. Your blood pressure is 145 over 90, heart rate 88, respiratory rate 18, temperature 98.6, oxygen saturation 97% on room air. You appear anxious but in no acute distress. Heart sounds are regular rate and rhythm, no murmurs. Lungs are clear to auscultation bilaterally. Chest wall is tender to palpation over the left parasternal region.
Doctor: Based on your symptoms and examination, this appears to be musculoskeletal chest pain, likely costochondritis. However, given your family history and the nature of the pain, I want to rule out cardiac causes. I'm going to order an ECG and cardiac enzyme labs. In the meantime, I'll prescribe ibuprofen for the pain. Take 400 milligrams three times daily with food for one week. Avoid strenuous activity. If the pain worsens, you develop shortness of breath, or feel like you're going to pass out, go to the emergency room immediately. Follow up with me in one week or sooner if symptoms worsen.
Patient: Okay, thank you doctor.

**Professional SOAP Note:**

SUBJECTIVE:
CC: Chest pain x 2 days

HPI: Patient is a middle-aged male presenting with acute onset chest pain that began 2 days ago while at work. Pain is sharp in character, located in the center of the chest, with severity rated 7/10. Pain is exacerbated by deep inspiration and coughing. No radiation noted. Associated with mild dyspnea. No nausea, diaphoresis, or syncope. No alleviating factors identified.

PMH: Hypertension

Medications: Lisinopril 10mg PO daily

Allergies: NKDA (No Known Drug Allergies)

Social History: Not documented in this encounter

Family History: Father with myocardial infarction at age 55

ROS: Cardiovascular: Denies palpitations. Respiratory: Mild dyspnea as noted in HPI. Constitutional: Denies fever, chills. Gastrointestinal: Denies nausea, vomiting.

OBJECTIVE:
Vital Signs: BP 145/90 mmHg, HR 88 bpm, RR 18 breaths/min, T 98.6°F, O2 sat 97% on RA

General: Anxious-appearing but in no acute distress

Physical Exam:
- Cardiovascular: Regular rate and rhythm, no murmurs, rubs, or gallops
- Respiratory: Lungs clear to auscultation bilaterally, no wheezes, rales, or rhonchi
- Chest wall: Tenderness to palpation over left parasternal region

ASSESSMENT:
1. Musculoskeletal chest pain, likely costochondritis (ICD-10: M94.0)
2. Rule out acute coronary syndrome given family history and presentation (ICD-10: I24.9 - for differential)
3. Hypertension, currently suboptimal control (ICD-10: I10)

PLAN:
1. Medications: Ibuprofen 400mg PO TID with food x 7 days for anti-inflammatory effect
2. Diagnostic testing: ECG and cardiac enzymes (troponin, CK-MB) to rule out ACS (CPT: 93000 for ECG, 80050 for cardiac panel)
3. Activity modification: Avoid strenuous physical activity until cardiac evaluation complete
4. Patient education: Explained differential diagnosis, warning signs of cardiac emergency
5. Follow-up: Return in 1 week or sooner if symptoms worsen
6. Red flags: Instructed to proceed to ED immediately if pain worsens, dyspnea increases, or syncopal symptoms develop

---

## EXAMPLE 2: Type 2 Diabetes Follow-Up

**Transcript:**
Doctor: Hi there, you're here for your diabetes follow-up. How have you been feeling?
Patient: Pretty good overall. I've been checking my blood sugars like you asked.
Doctor: Great. What have the numbers been running?
Patient: Usually between 140 and 180 in the morning.
Doctor: Are you taking your metformin regularly?
Patient: Yes, 1000 milligrams twice a day with meals.
Doctor: Any side effects from the medication?
Patient: A little bit of stomach upset sometimes, but it's manageable.
Doctor: Good. Any symptoms of high or low blood sugar? Excessive thirst, frequent urination, shakiness, sweating?
Patient: I do get up to urinate once or twice at night, but no other symptoms.
Doctor: How about your diet and exercise?
Patient: I've been walking 30 minutes most days. Trying to cut back on carbs.
Doctor: Excellent. Any vision changes, numbness in your feet, or wounds that aren't healing?
Patient: No, nothing like that.
Doctor: Let me check your vitals. Blood pressure 128 over 82, pulse 76, weight 210 pounds. Feet look good, no ulcers, pulses are intact. I got your lab results back - your hemoglobin A1c is 7.8%, which is better than last time but still above our goal of under 7. Kidney function is normal, cholesterol panel looks good.
Doctor: Your diabetes control is improving, but we need to get that A1c down a bit more. I'm going to increase your metformin to 1000 milligrams in the morning and 1500 milligrams in the evening. Continue your diet and exercise program. I want to see you back in 3 months with repeat labs. Keep monitoring your blood sugars daily.
Patient: Sounds good.

**Professional SOAP Note:**

SUBJECTIVE:
CC: Type 2 diabetes mellitus follow-up visit

HPI: Patient reports compliance with current diabetes regimen. Home glucose monitoring shows fasting blood glucose values ranging 140-180 mg/dL. Denies polyuria during daytime, though reports nocturia 1-2 times per night. Denies polydipsia, polyphagia, or symptoms of hypoglycemia (shakiness, diaphoresis, palpitations). Tolerating metformin with mild GI upset that is manageable.

PMH: Type 2 diabetes mellitus

Medications: Metformin 1000mg PO BID with meals

Allergies: NKDA

Social History: Currently engaging in regular physical activity (walking 30 minutes most days). Attempting dietary modification with carbohydrate reduction.

Family History: Not documented in this encounter

ROS: Ophthalmologic: Denies vision changes. Neurological: Denies paresthesias or numbness in extremities. Integumentary: Denies non-healing wounds. Genitourinary: Nocturia as noted in HPI.

OBJECTIVE:
Vital Signs: BP 128/82 mmHg, HR 76 bpm, Weight 210 lbs

Physical Exam:
- Extremities: Bilateral lower extremities without ulcerations, erythema, or edema. Dorsalis pedis and posterior tibial pulses 2+ bilaterally.

Labs (reviewed):
- Hemoglobin A1c: 7.8% (goal <7%)
- Renal function: Within normal limits
- Lipid panel: Within acceptable range

ASSESSMENT:
1. Type 2 diabetes mellitus with suboptimal glycemic control (ICD-10: E11.65)
   - Current A1c 7.8%, improved from prior but above target
   - No evidence of diabetic complications at this time
2. Medication compliance good with mild GI side effects
3. Lifestyle modifications (diet and exercise) being implemented appropriately

PLAN:
1. Medications: Increase metformin to 1000mg PO in AM and 1500mg PO in PM with meals (titration for improved glycemic control)
2. Diagnostic testing: Repeat Hemoglobin A1c, CMP, and lipid panel in 3 months (CPT: 83036, 80053, 80061)
3. Continue home blood glucose monitoring daily
4. Continue current diet and exercise regimen
5. Patient education: Reinforced importance of medication compliance, dietary modification, and regular physical activity for diabetes management
6. Follow-up: Return in 3 months with lab results
7. Monitoring: Watch for signs of hypoglycemia with increased metformin dose (shakiness, sweating, confusion)

---

## EXAMPLE 3: Pediatric URI (Upper Respiratory Infection)

**Transcript:**
Doctor: What brings little Emma in today?
Parent: She's had a runny nose and cough for about 3 days now.
Doctor: Has she had any fever?
Parent: Yes, she had a fever yesterday of 100.5 degrees, but it came down with Tylenol.
Doctor: What color is the nasal discharge?
Parent: It started clear but now it's yellow-green.
Doctor: Is she eating and drinking okay?
Parent: Her appetite is a little down, but she's drinking fluids well.
Doctor: Any ear pain, difficulty breathing, or wheezing?
Parent: No ear pain. The cough sounds a little tight but no wheezing that I can hear.
Doctor: Any vomiting or diarrhea?
Parent: No.
Doctor: Is she up to date on her vaccines?
Parent: Yes, she had her last vaccines at her 4-year checkup.
Doctor: Any other kids at daycare sick?
Parent: Yes, there's been a cold going around.
Doctor: Okay, let me take a look at her. Temperature is 99.2, heart rate 110, respiratory rate 24, oxygen saturation 98%. She looks alert and playful. Ears are clear bilaterally. Throat is mildly erythematous without exudates. Nasal turbinates are swollen with yellow-green discharge. Lungs are clear, no wheezes or crackles. Good air movement. Lymph nodes in her neck are slightly enlarged but mobile and non-tender.
Doctor: Emma has a viral upper respiratory infection, which is very common in young children, especially in daycare settings. The yellow-green nasal discharge doesn't necessarily mean it's bacterial - that's normal progression of a viral cold. Her lungs sound clear, so I'm not concerned about pneumonia. This should resolve on its own in 7 to 10 days. Continue giving her plenty of fluids. You can use saline nasal drops and suction to help with the congestion. For fever or discomfort, you can give her acetaminophen or ibuprofen as directed on the bottle. If she develops high fever over 102, difficulty breathing, ear pain, or isn't improving in a week, bring her back.
Parent: No antibiotics needed?
Doctor: Not at this time. It's viral, so antibiotics won't help and might cause side effects. Call me if things aren't improving or if you're concerned.

**Professional SOAP Note:**

SUBJECTIVE:
CC: Cough and rhinorrhea x 3 days

HPI: Patient is a 4-year-old female presenting with 3-day history of rhinorrhea and cough. Nasal discharge initially clear, now yellow-green in color. Associated with subjective fever (Tmax 100.5°F at home yesterday), responsive to acetaminophen. Cough described as "tight" sounding by parent. Decreased appetite, though fluid intake maintained. Denies ear pain, dyspnea, or wheezing. No vomiting or diarrhea. Known sick contacts at daycare.

PMH: Unremarkable for age

Medications: Acetaminophen PRN for fever

Allergies: NKDA

Immunizations: Up to date per parent report

Social History: Attends daycare with current viral illness outbreak

Family History: Not documented in this encounter

ROS: ENT: Rhinorrhea and cough as noted in HPI. Denies otalgia. Respiratory: Denies dyspnea or wheezing. Gastrointestinal: Decreased appetite, denies emesis or diarrhea. Constitutional: Fever as above.

OBJECTIVE:
Vital Signs: T 99.2°F, HR 110 bpm, RR 24 breaths/min, O2 sat 98% on RA

General: Alert, playful, well-appearing child in no acute distress

Physical Exam:
- ENT: Tympanic membranes clear and mobile bilaterally without erythema or effusion. Oropharynx with mild erythema, no tonsillar exudates. Nasal turbinates edematous with yellow-green discharge visible.
- Neck: Bilateral cervical lymphadenopathy, nodes small (<1cm), mobile, non-tender
- Respiratory: Lungs clear to auscultation bilaterally, no wheezes, rales, or rhonchi. Good air entry throughout. No increased work of breathing.
- Cardiovascular: Regular rate and rhythm, no murmur

ASSESSMENT:
1. Acute viral upper respiratory infection (URI), likely rhinovirus (ICD-10: J06.9)
   - Consistent with viral prodrome: rhinorrhea progression from clear to purulent (normal viral course)
   - No evidence of bacterial superinfection (pneumonia, otitis media, sinusitis)
   - Exposure history consistent with daycare outbreak
2. Low-grade fever, resolving

PLAN:
1. Supportive care: Adequate hydration, rest
2. Symptomatic management:
   - Acetaminophen 160mg PO Q4-6H PRN fever >100.4°F or discomfort (weight-based dosing)
   - Alternative: Ibuprofen 100mg PO Q6H PRN (may use if parent prefers)
   - Saline nasal drops with bulb suction for nasal congestion
3. No antibiotics indicated at this time (viral etiology)
4. Patient/parent education:
   - Natural course of viral URI is 7-10 days
   - Yellow-green nasal discharge is normal viral progression, not indication for antibiotics
   - Explained rationale for supportive care vs antibiotics
5. Follow-up: Return if no improvement in 7 days, or sooner if development of:
   - High fever >102°F
   - Difficulty breathing or increased work of breathing
   - Ear pain (concern for otitis media)
   - Decreased fluid intake or signs of dehydration
6. Contact precautions: May return to daycare when fever-free for 24 hours

---

## EXAMPLE 4: Anxiety Disorder - Psychiatry

**Transcript:**
Doctor: Thanks for coming in today. What brings you here?
Patient: I've been feeling really anxious lately. It's been getting worse over the past few months.
Doctor: Can you tell me more about what you're experiencing?
Patient: I worry about everything - work, family, my health. My heart races, I get sweaty, and I feel like I can't catch my breath sometimes.
Doctor: How often does this happen?
Patient: Almost every day. It's exhausting.
Doctor: Are there any specific triggers that you've noticed?
Patient: Not really. It just seems to come out of nowhere.
Doctor: Have you had any panic attacks where these symptoms come on very suddenly and intensely?
Patient: Yes, I've had a few episodes like that in the past month. I thought I was having a heart attack.
Doctor: I'm sorry you're going through this. How is this affecting your daily life?
Patient: It's hard to concentrate at work. I'm not sleeping well. I've been avoiding social situations because I'm afraid of having an attack in public.
Doctor: Any thoughts of harming yourself or anyone else?
Patient: No, nothing like that.
Doctor: Have you had any major life stressors recently?
Patient: My mom was diagnosed with cancer 6 months ago. That's been really hard.
Doctor: I can understand how that would be stressful. Do you drink alcohol or use any substances?
Patient: I have a glass of wine most nights to help me relax.
Doctor: Any history of anxiety or depression in your family?
Patient: My sister has depression.
Doctor: Are you currently taking any medications?
Patient: No, just vitamins.
Doctor: Let me check your vital signs. Blood pressure is 135 over 85, heart rate is 95. You appear anxious, making good eye contact. Your speech is normal. Your thought process is logical and goal-directed. No evidence of hallucinations or delusions. Your insight and judgment seem good.
Doctor: Based on what you've told me, it sounds like you're experiencing generalized anxiety disorder with panic attacks. This is a very treatable condition. I'd like to start you on an SSRI medication called sertraline. We'll start at a low dose of 25 milligrams once daily and increase it to 50 milligrams after a week. I'm also going to refer you to a therapist who specializes in cognitive behavioral therapy, which is very effective for anxiety. In the meantime, I want you to try to reduce your alcohol intake, as it can actually make anxiety worse. Practice deep breathing exercises when you feel anxious. Follow up with me in 4 weeks to see how the medication is working.
Patient: Okay, I'm willing to try that.

**Professional SOAP Note:**

SUBJECTIVE:
CC: Anxiety symptoms worsening over past few months

HPI: Patient is an adult presenting with progressive anxiety symptoms over the past several months. Reports excessive worry regarding multiple domains (work, family, health) occurring almost daily. Physical symptoms include palpitations, diaphoresis, and dyspnea. These symptoms are not associated with specific identifiable triggers. Patient has experienced several discrete panic attacks in the past month, described as sudden onset of intense symptoms with fear of having myocardial infarction. Symptoms significantly impacting occupational functioning (difficulty with concentration at work) and social functioning (avoidance of social situations due to fear of panic attack). Sleep disturbance present. Denies suicidal ideation, homicidal ideation, or intent to harm self or others.

PMH: None reported

Medications: Multivitamin daily

Allergies: NKDA

Social History: Consuming alcohol (wine) nightly for anxiety self-medication. No other substance use reported. Significant recent stressor: mother diagnosed with cancer 6 months ago.

Family History: Sister with major depressive disorder

ROS: Psychiatric: As detailed in HPI. Cardiovascular: Palpitations as above. Respiratory: Dyspnea associated with anxiety. Neurological: Denies headaches, dizziness. Sleep: Impaired sleep quality. Constitutional: Denies fever, weight changes.

OBJECTIVE:
Vital Signs: BP 135/85 mmHg, HR 95 bpm

Mental Status Examination:
- Appearance: Appropriately dressed and groomed
- Behavior: Anxious affect, good eye contact, cooperative
- Speech: Normal rate, rhythm, and volume
- Mood: "Anxious" (stated)
- Affect: Anxious, congruent with mood
- Thought process: Logical, linear, goal-directed
- Thought content: Preoccupied with worries. No suicidal ideation, homicidal ideation, or intent to harm self/others. No delusions.
- Perception: No auditory or visual hallucinations
- Cognition: Alert and oriented x3, intact attention and concentration (though impaired by anxiety)
- Insight: Good - recognizes need for treatment
- Judgment: Good - seeking appropriate care

ASSESSMENT:
1. Generalized Anxiety Disorder with panic attacks (ICD-10: F41.0)
   - Excessive worry across multiple domains with associated physical symptoms
   - Duration >6 months with progressive worsening
   - Significant functional impairment in occupational and social domains
   - Panic attacks: recurrent unexpected panic episodes with fear of cardiac event
2. Insomnia related to anxiety (ICD-10: G47.00)
3. Maladaptive coping: Alcohol use for anxiety self-medication (concern for potential substance use disorder development)
4. Adjustment to family stressor (mother's cancer diagnosis)

PLAN:
1. Pharmacotherapy:
   - Start Sertraline (Zoloft) 25mg PO daily x 7 days, then increase to 50mg PO daily
   - Explained typical SSRI onset of action (2-4 weeks), potential side effects (GI upset, initial anxiety increase, sexual dysfunction)
   - Safety counseling: No abrupt discontinuation
2. Psychotherapy referral:
   - Referral to therapist specializing in Cognitive Behavioral Therapy (CBT) for anxiety disorders (CPT: 90834 for psychotherapy)
   - CBT is evidence-based first-line treatment for GAD
3. Lifestyle modifications:
   - Reduce/eliminate alcohol use (explained paradoxical anxiety exacerbation with alcohol)
   - Sleep hygiene education
   - Deep breathing exercises and progressive muscle relaxation techniques
4. Patient education:
   - Explained diagnosis of Generalized Anxiety Disorder and panic disorder
   - Discussed treatment approach: combined medication + therapy most effective
   - Provided resources for anxiety management
5. Safety planning:
   - Patient denies SI/HI, low acute risk
   - Instructed to present to ED or call crisis line if suicidal thoughts develop
   - Emergency contact: National Suicide Prevention Lifeline 988
6. Follow-up: Return visit in 4 weeks to assess medication response and tolerance
7. Monitoring: Screen for treatment-emergent suicidal ideation (black box warning for SSRIs in young adults)

---

## EXAMPLE 5: Hypertension - New Diagnosis

**Transcript:**
Doctor: Hi, I'm seeing you today because your blood pressure has been elevated at your last few visits. How are you feeling?
Patient: I feel fine, actually. No complaints.
Doctor: That's good. High blood pressure often doesn't cause symptoms, which is why we call it the "silent killer." Let me review your recent readings. Three weeks ago it was 152 over 96, two weeks ago 148 over 94, and today it's 155 over 98. All of these are above the normal range. Have you ever been told you have high blood pressure before?
Patient: No, never.
Doctor: Do you have any family history of high blood pressure, heart disease, or stroke?
Patient: My dad had high blood pressure and had a stroke in his 60s.
Doctor: I see. Do you smoke or use tobacco?
Patient: No, I quit smoking 5 years ago.
Doctor: That's great. How much alcohol do you drink?
Patient: A couple of beers on the weekends.
Doctor: What about your diet? Do you eat a lot of salty foods?
Patient: I probably eat more fast food than I should. I work long hours and it's convenient.
Doctor: Understandable. Do you exercise regularly?
Patient: Not really. I sit at a desk all day.
Doctor: Are you taking any medications or supplements?
Patient: No.
Doctor: Any chest pain, shortness of breath, headaches, or vision changes?
Patient: No, nothing like that.
Doctor: Okay, let me examine you. Your blood pressure today is 155 over 98. Heart rate is 78, regular. Your weight is 210 pounds, height is 5 foot 10 inches, so your BMI is 30.1, which is in the obese range. Heart sounds are normal, lungs are clear. Pulses are strong in your arms and legs. No swelling in your ankles.
Doctor: Based on your elevated blood pressure readings on three separate occasions, you have hypertension. With your family history and BMI, it's important we get this under control to prevent complications like heart attack and stroke. I'd like to start you on a medication called lisinopril, 10 milligrams once daily. I'm also ordering some baseline labs to check your kidney function and cholesterol. We need to work on lifestyle changes too - reducing sodium intake, losing some weight, and increasing physical activity. I want you to get a home blood pressure monitor and check it daily. Record your readings and bring them to our follow-up visit in 4 weeks.
Patient: Okay, I can do that.

**Professional SOAP Note:**

SUBJECTIVE:
CC: Follow-up for elevated blood pressure readings

HPI: Patient is an asymptomatic adult male with persistently elevated blood pressure noted over 3 recent clinic visits: 152/96 mmHg (3 weeks ago), 148/94 mmHg (2 weeks ago), and 155/98 mmHg (today). No prior diagnosis of hypertension. Denies symptoms of end-organ damage including chest pain, dyspnea, headaches, or visual disturbances. Patient reports feeling well overall.

PMH: None

Medications: None

Allergies: NKDA

Social History:
- Tobacco: Former smoker, quit 5 years ago
- Alcohol: Occasional use, 2 beers on weekends
- Diet: High sodium intake, frequent fast food consumption due to work schedule
- Exercise: Sedentary lifestyle, prolonged sitting at desk job

Family History: Father with hypertension and cerebrovascular accident (stroke) in 6th decade

ROS: Cardiovascular: Denies chest pain, palpitations. Respiratory: Denies dyspnea. Neurological: Denies headaches, visual changes, syncope. All other systems reviewed and negative.

OBJECTIVE:
Vital Signs: BP 155/98 mmHg (confirmed on repeat), HR 78 bpm regular, Height 5'10", Weight 210 lbs, BMI 30.1 kg/m²

Physical Exam:
- General: Well-appearing, no acute distress
- Cardiovascular: Regular rate and rhythm, S1 and S2 normal, no murmurs, rubs, or gallops. Distal pulses 2+ and symmetric in all extremities.
- Respiratory: Lungs clear to auscultation bilaterally
- Extremities: No peripheral edema
- Fundoscopic: Not documented in this encounter

Blood Pressure Trend:
- 3 weeks ago: 152/96 mmHg
- 2 weeks ago: 148/94 mmHg
- Today: 155/98 mmHg

ASSESSMENT:
1. Hypertension, Stage 2, newly diagnosed (ICD-10: I10)
   - Persistent elevation: 3 readings >140/90 mmHg
   - No evidence of secondary causes at this time
   - No current evidence of target organ damage
2. Obesity, Class I (BMI 30.1) (ICD-10: E66.9)
   - Contributing risk factor for hypertension
3. Elevated cardiovascular risk
   - Risk factors: HTN, obesity, family history of CVA, sedentary lifestyle
4. Tobacco use disorder, in remission (former smoker) (ICD-10: Z87.891)

PLAN:
1. Pharmacotherapy:
   - Initiate Lisinopril (ACE inhibitor) 10mg PO daily
   - Goal BP <130/80 mmHg per current guidelines
   - Counseled on potential side effects: dry cough, dizziness, hyperkalemia
2. Diagnostic testing:
   - Basic Metabolic Panel (BMP) to assess renal function and electrolytes (CPT: 80048)
   - Lipid panel (fasting) to assess cardiovascular risk (CPT: 80061)
   - Urinalysis to screen for proteinuria (CPT: 81001)
   - Baseline ECG (CPT: 93000)
3. Home blood pressure monitoring:
   - Instructed patient to obtain home BP monitor
   - Monitor BP daily, same time each day, log all readings
   - Bring log to follow-up visit
4. Lifestyle modifications (Therapeutic Lifestyle Changes):
   - Dietary: DASH diet education, reduce sodium intake to <2300mg/day
   - Weight loss: Goal to lose 5-10% body weight (10-20 lbs)
   - Exercise: Initiate regular aerobic exercise, goal 150 min/week moderate intensity
   - Alcohol: Current intake acceptable (within guidelines)
   - Stress management techniques
5. Patient education:
   - Explained hypertension diagnosis, asymptomatic nature, and importance of treatment
   - Discussed cardiovascular risk reduction through medication + lifestyle changes
   - Reviewed warning signs of hypertensive emergency (severe headache, vision changes, chest pain)
6. Follow-up: Return in 4 weeks with home BP log and lab results
7. Monitoring: Assess medication efficacy, tolerability, and adherence at next visit

---

## EXAMPLE 6: Acute Stress Reaction - Crisis Counseling

**Transcript:**
Doctor: Thank you for coming in today. I understand you wanted to talk about some stressful situations you've been experiencing?
Patient: Yes, I've been dealing with a really difficult situation at home. My partner and I have been having severe conflicts, and things have gotten physical a few times in the past month.
Doctor: I'm sorry you're going through this. Can you tell me more about what's been happening?
Patient: We've been together for 5 years. About 3 months ago, we started having more arguments, mostly about money since I lost my job. Last month, during an argument, my partner pushed me and I fell. Then it happened again two weeks ago.
Doctor: Have you been injured during these incidents?
Patient: Some bruises on my arms, but they've healed. I'm more worried about the constant stress. I can't sleep, I'm anxious all the time, and I've been having panic attacks.
Doctor: Have you had panic attacks before?
Patient: No, never. The first one was about a week ago. My heart was racing, I couldn't breathe, I thought I was dying.
Doctor: How often are these panic attacks happening?
Patient: Maybe 3 or 4 times in the past week, especially at night when I'm trying to sleep.
Doctor: Are you currently living with this partner?
Patient: Yes, but I've been thinking about leaving. I'm just not sure where to go.
Doctor: Have you ever felt like hurting yourself?
Patient: No, nothing like that. I just want things to get better.
Doctor: Do you have family or friends you can stay with if needed?
Patient: My sister lives nearby. I haven't told her what's going on though.
Doctor: I see. Any past mental health history?
Patient: I saw a therapist years ago for mild depression after my mom died, but that was brief.
Doctor: Any medications or drug allergies?
Patient: No medications. No allergies that I know of.
Doctor: Let me check your vitals. Your blood pressure is 138 over 88, heart rate is 98, respiratory rate 16, temperature 98.4, oxygen saturation 99%. You appear anxious and tearful but alert and oriented. Your affect is congruent with your mood. No signs of acute injury on examination today.
Doctor: Based on what you've shared, you're experiencing an acute stress reaction related to ongoing domestic conflict and violence. The panic attacks and sleep disturbance are common responses to this level of stress. I want to help you in several ways. First, your safety is the most important thing. I'm going to provide you with resources for confidential support - the National Domestic Violence Hotline number is 1-800-799-7233, and they can help you develop a private safety plan and connect you with local shelters if you need them. I'm also going to start you on a low dose of sertraline, 25 milligrams daily, to help with the anxiety and panic symptoms. This medication can take a few weeks to work fully. I'd like to refer you to a trauma-informed therapist who specializes in helping people through situations like this. In the meantime, if you feel unsafe at any time, please call 911 or go to the nearest emergency room. Do you feel safe going home today?
Patient: I think so. Thank you for the resources.
Doctor: I want you to follow up with me in one week. We'll check on your symptoms and make sure you're getting the support you need.

**Professional SOAP Note:**

SUBJECTIVE:
CC: Stress and anxiety related to domestic conflict

HPI: Patient is an adult presenting with acute onset anxiety and panic symptoms over the past 3 months, coinciding with escalating relationship conflict and incidents of physical violence. Reports 5-year relationship with current partner. Conflict escalation began 3 months ago following job loss, with financial stress as primary trigger. First physical altercation occurred 1 month ago (pushed by partner, resulting in fall), with second incident 2 weeks ago. Patient sustained bruising to bilateral upper extremities (since resolved). New-onset panic attacks began 1 week ago, now occurring 3-4 times weekly, predominantly nocturnal. Panic symptoms include tachycardia, dyspnea, and sense of impending doom. Severe sleep disturbance present. Patient reports persistent anxiety throughout day. Currently cohabitating with partner but considering leaving. Has potential social support (sister in area, unaware of situation). Denies current suicidal ideation or intent. Contemplating safety planning.

PMH: History of mild depression following maternal death (several years ago), treated with brief counseling

Medications: None

Allergies: NKDA

Social History:
- Recently unemployed (financial stressor)
- In 5-year relationship with current partner
- Currently cohabitating with partner
- Sister lives locally (potential support system)
- No current substance use reported

Family History: Maternal death (timing not specified)

ROS: Psychiatric: Anxiety, panic attacks, insomnia as detailed in HPI. Denies suicidal ideation. Cardiovascular: Palpitations during panic episodes. Neurological: Denies headaches. Constitutional: Denies fever, weight changes.

OBJECTIVE:
Vital Signs: BP 138/88 mmHg, HR 98 bpm, RR 16 breaths/min, T 98.4°F, O2 sat 99% on RA

Mental Status Examination:
- Appearance: Appropriately dressed, adequate hygiene
- Behavior: Anxious, tearful, cooperative with exam
- Speech: Normal rate and volume
- Mood: "Stressed and scared" (stated)
- Affect: Anxious, tearful, congruent with mood and situation
- Thought process: Linear, goal-directed, coherent
- Thought content: Preoccupied with relationship conflict and safety concerns. Denies suicidal ideation, homicidal ideation, or intent to harm self or others. No delusions.
- Perception: No hallucinations reported or observed
- Cognition: Alert and oriented to person, place, time, and situation
- Insight: Good - recognizes need for support and intervention
- Judgment: Good - seeking appropriate help, considering safety planning

Physical Exam:
- Skin: No acute bruising, lacerations, or other signs of recent trauma visible on exposed areas
- No acute distress noted

ASSESSMENT:
1. Acute stress reaction related to domestic violence exposure
   - New-onset panic attacks (first episode 1 week ago)
   - Severe insomnia
   - Persistent anxiety symptoms
   - Timeline correlates with escalating relationship violence
2. Intimate partner violence, victim of physical assault
   - Two documented physical altercations in past month
   - Patient currently cohabitating with perpetrator
   - High risk for escalation
3. Adjustment disorder with anxiety

PLAN:
1. Pharmacotherapy:
   - Start Sertraline 25mg PO daily for anxiety and panic symptoms
   - Explained typical onset of action (2-4 weeks for full effect)
   - Reviewed potential side effects (GI upset, initial anxiety increase, sexual side effects)
   - Safety counseling provided regarding gradual titration and no abrupt discontinuation

2. Safety planning (PATIENT-CENTERED):
   - Provided National Domestic Violence Hotline: 1-800-799-7233 (24/7 confidential support)
   - Discussed importance of confidential, private safety planning
   - Encouraged patient to contact hotline for professional safety planning assistance and local shelter resources
   - Identified sister as potential support person when patient is ready to disclose
   - Reviewed signs of escalating danger and need for immediate safety measures

3. Psychotherapy referral:
   - Referral to trauma-informed therapist specializing in intimate partner violence and acute stress
   - Cognitive Behavioral Therapy and trauma-focused interventions recommended

4. Patient education:
   - Explained acute stress reaction and relationship to trauma exposure
   - Discussed that anxiety and panic symptoms are normal responses to dangerous situations
   - Validated patient's concerns and experiences
   - Emphasized that violence is not patient's fault and resources are available

5. Safety assessment:
   - Patient denies current suicidal ideation
   - Assessed patient feels safe returning home today but situation remains high-risk
   - Instructed to call 911 or present to ED if feels unsafe or if violence escalates
   - Provided emergency resources: 911, local ED address

6. Follow-up:
   - Return visit in 1 week to reassess symptoms and safety
   - Earlier return if symptoms worsen or safety concerns arise
   - Monitor medication response and side effects

7. Documentation:
   - Confidential documentation maintained per HIPAA and safety protocols
   - Patient aware of mandatory reporting requirements if disclosures meet criteria

---

## End of Few-Shot Examples

These examples demonstrate:
✓ Professional medical terminology
✓ Clear HPI vs ROS separation (ROS contains ONLY patient-reported symptoms, NEVER vital signs or measurements)
✓ Proper Subjective (patient-reported) vs Objective (doctor-observed) distinction
✓ Context-appropriate HPI formats: OLDCARTS for physical symptoms, narrative for behavioral/psychiatric issues
✓ CRITICAL IPV Safety: Patient-only safety planning, NEVER involving partner/abuser (see Example 6)
✓ ICD-10 diagnosis codes in Assessment ONLY when certain (many examples omit codes to avoid errors)
✓ CPT procedure codes in Plan
✓ Medications supported by documented findings (clinical logic)
✓ Structured, legally-compliant documentation
"""


# =============================================================================
# Complete Professional Prompt (combines System + CoT + Few-Shot)
# =============================================================================

def get_professional_soap_prompt(
    transcription: str,
    target_language: str = "en"
) -> tuple[str, str]:
    """
    Get the complete professional SOAP generation prompt.

    Combines:
    1. System prompt (role definition)
    2. Few-shot examples (teaching by example)
    3. Chain-of-Thought instructions (reasoning process)
    4. Language instruction (multi-language support)

    The target_language parameter ensures the SOAP note is generated in the
    same language as the original audio/transcript. This is crucial for:
    - Maintaining clinical accuracy (no translation drift)
    - Preserving patient-specific terminology
    - Supporting international medical practices

    Args:
        transcription: Medical consultation transcript with speaker labels
        target_language: ISO 639-1 language code (e.g., "es", "fr", "ru")
                        Defaults to "en" (English) for backward compatibility

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Convert language code to full name for better LLM compliance
    language_name = get_language_name(target_language)
    
    # Build language instruction
    # Note: Few-shot examples are in English, but we explicitly instruct
    # the model to output in the target language. Modern LLMs handle this well.
    if target_language.lower() != "en":
        language_instruction = f"""

## CRITICAL LANGUAGE REQUIREMENT:
The transcript above is in {language_name}. You MUST write the entire SOAP note in {language_name}.
- Use {language_name} medical terminology
- All section headers (SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN) should remain in English for standardization
- All clinical content must be written in {language_name}
- Do NOT translate to English - preserve the original language of the consultation"""
    else:
        language_instruction = ""
    
    user_prompt = f"""{FEW_SHOT_EXAMPLES}

---

Now, using the same professional standards demonstrated in the examples above, generate a SOAP note for the following NEW consultation:

{SOAP_GENERATION_PROMPT_COT.format(transcription=transcription)}{language_instruction}"""

    return (MEDICAL_SCRIBE_SYSTEM_PROMPT, user_prompt)


# =============================================================================
# Backward Compatibility Functions
# =============================================================================

def get_soap_prompt(
    transcription: str,
    specialty: str = "general",
    target_language: str = "en"
) -> str:
    """
    Backward compatible function.

    Now returns the professional Chain-of-Thought prompt with few-shot examples.
    Supports multi-language output via target_language parameter.

    Args:
        transcription: Medical consultation transcript
        specialty: Medical specialty (currently unused, reserved for future)
        target_language: ISO 639-1 language code for output language
    """
    system_prompt, user_prompt = get_professional_soap_prompt(
        transcription,
        target_language=target_language
    )
    # Combine for models that don't support separate system prompts
    return f"{system_prompt}\n\n{user_prompt}"


def get_system_prompt() -> str:
    """Get the professional medical scribe system prompt."""
    return MEDICAL_SCRIBE_SYSTEM_PROMPT


# =============================================================================
# Validation and Summarization Prompts (Future Features)
# =============================================================================

SOAP_VALIDATION_PROMPT = """Review the following SOAP note for professional quality and clinical accuracy.

## SOAP Note to Review:
{soap_note}

## Original Transcript:
{transcription}

## Evaluation Criteria:
1. **Accuracy**: Does the note accurately reflect the transcript? No hallucinations?
2. **Attribution**: Are Subjective and Objective sections correct? (Patient reports vs Doctor observes)
3. **Completeness**: Are all clinically relevant details captured?
4. **Structure**: Is HPI separate from ROS? Proper section formatting?
5. **Terminology**: Professional medical terminology used consistently?
6. **Coding**: Are ICD-10/CPT codes appropriate if included?

## Respond with:
- VALID: Note meets professional standards
- ISSUES: [List specific problems found]"""


SUMMARIZATION_PROMPT = """Provide a 2-3 sentence clinical summary of this consultation:

{transcription}

Focus on: Chief complaint, key findings, and disposition/plan."""


# =============================================================================
# Map-Reduce Prompts (Phase 5 - for long transcripts)
# =============================================================================

MAP_PROMPT_CHUNK_SUMMARY = """You are analyzing a portion of a longer medical consultation transcript. Extract all clinical facts from this segment.

## Transcript Segment:
{chunk}

## Extract and list:
1. Patient-reported symptoms and history (Subjective data)
2. Doctor's observations and exam findings (Objective data)
3. Diagnoses or assessments mentioned (Assessment data)
4. Treatments or plans discussed (Plan data)

Be comprehensive but concise. This summary will be used to generate the final SOAP note."""


REDUCE_PROMPT_FINAL_SOAP = """You have summaries of clinical facts from a long medical consultation. Now generate the final professional SOAP note.

## Clinical Facts Summaries:
{summaries}

## Generate:
A complete, professional SOAP note following the same structure and standards as the examples in your training. Use proper medical terminology, separate HPI from ROS, include ICD-10/CPT codes where appropriate."""
