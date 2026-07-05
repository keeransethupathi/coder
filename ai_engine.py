from pydantic import BaseModel, Field
from typing import List
import os
import json
import ollama
from google import genai
from google.genai import types

# ----------------- PYDANTIC STRUCTURED OUTPUT SCHEMA -----------------

class ICD10CharBreakdown(BaseModel):
    position: int = Field(description="Character position (1 to 7)")
    character: str = Field(description="The alphanumeric character value in the code")
    name: str = Field(description="The name of the ICD-10-PCS axis (e.g. Section, Body System, Root Operation, Body Part, Approach, Device, Qualifier)")
    value_description: str = Field(description="Description of what this character code means in this context")

class ICD10PCSCode(BaseModel):
    code: str = Field(description="The complete 7-character ICD-10-PCS code")
    procedure_name: str = Field(description="Name of the procedure in the report mapped to this code")
    confidence: float = Field(description="Confidence score (0.0 to 1.0) of this mapping")
    justification: str = Field(description="Clinical explanation of why this code was selected based on operative findings")
    breakdown: List[ICD10CharBreakdown] = Field(description="The 7-character breakdown of the ICD-10-PCS code")

class SurgicalSummary(BaseModel):
    preoperative_diagnosis: str = Field(description="Pre-operative diagnosis mentioned in the report")
    postoperative_diagnosis: str = Field(description="Post-operative diagnosis mentioned in the report")
    procedures_performed: List[str] = Field(description="List of procedures performed")
    surgical_team: str = Field(description="Names of surgeon, assistant, and anesthesiologist if available")
    anesthesia: str = Field(description="Type of anesthesia used (e.g. General, Local, Epidural)")
    key_findings: List[str] = Field(description="Bullet points of major surgical findings and observations")
    specimens_removed: List[str] = Field(description="Specimens removed for pathology (e.g. gallbladder, appendix), or empty list")
    estimated_blood_loss: str = Field(description="Estimated blood loss (EBL) mentioned, or 'Not specified'")
    complications: str = Field(description="Complications encountered, or 'None'")
    operative_narrative_summary: str = Field(description="A concise summary of the surgical steps described in the report")

class ReportAnalysisResult(BaseModel):
    summary: SurgicalSummary = Field(description="Structured clinical summary of the report")
    icd10_codes: List[ICD10PCSCode] = Field(description="Suggested ICD-10-PCS procedure codes with breakdowns")

# ----------------- DUAL BACKEND AI IMPLEMENTATION -----------------

def analyze_surgical_report_ollama(report_text: str, model_name: str = "meditron:latest", host_url: str = None) -> ReportAnalysisResult:
    """
    Interfaces with a local Ollama instance to analyze the surgical report text.
    """
    client = ollama
    if host_url:
        client = ollama.Client(host=host_url)
        
    prompt = f"""
You are a senior clinical coding expert specializing in ICD-10-PCS (Procedure Coding System).
Analyze the following surgical report and generate a structured JSON response matching the requested schema.

Guidelines for ICD-10-PCS Mapping:
- ICD-10-PCS codes must be exactly 7 characters long.
- Break down each character from position 1 to 7 detailing:
  1. Section (e.g., 0 for Medical and Surgical)
  2. Body System (e.g., F for Hepatobiliary System and Pancreas, D for Gastrointestinal)
  3. Root Operation (e.g., T for Resection, D for Extraction, B for Excision)
  4. Body Part (e.g., 4 for Gallbladder, J for Appendix)
  5. Approach (e.g., 4 for Percutaneous Endoscopic, 0 for Open)
  6. Device (e.g., Z for No Device)
  7. Qualifier (e.g., Z for No Qualifier)
- Ensure high mapping accuracy based on clinical guidelines.

Surgical Report:
\"\"\"
{report_text}
\"\"\"
"""

    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        format=ReportAnalysisResult.model_json_schema()
    )
    
    return ReportAnalysisResult.model_validate_json(response['message']['content'])


def analyze_surgical_report_gemini(report_text: str, api_key: str, model_name: str = "gemini-2.5-flash") -> ReportAnalysisResult:
    """
    Interfaces with Google Gemini via the GenAI SDK to analyze the surgical report.
    """
    if not api_key:
        raise ValueError("API Key is required to call the Gemini AI engine.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are a senior clinical coding expert specializing in ICD-10-PCS (Procedure Coding System).
Analyze the following surgical report and generate a structured response according to the requested schema.

Guidelines for ICD-10-PCS Mapping:
- ICD-10-PCS codes must be exactly 7 characters long.
- Break down each character from position 1 to 7 detailing:
  1. Section (e.g., 0 for Medical and Surgical)
  2. Body System (e.g., F for Hepatobiliary System and Pancreas, D for Gastrointestinal)
  3. Root Operation (e.g., T for Resection, D for Extraction, B for Excision)
  4. Body Part (e.g., 4 for Gallbladder, J for Appendix)
  5. Approach (e.g., 4 for Percutaneous Endoscopic, 0 for Open)
  6. Device (e.g., Z for No Device)
  7. Qualifier (e.g., Z for No Qualifier)
- Ensure high mapping accuracy based on clinical guidelines.

Surgical Report:
\"\"\"
{report_text}
\"\"\"
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReportAnalysisResult,
            temperature=0.1
        )
    )
    
    return ReportAnalysisResult.model_validate_json(response.text)

# ----------------- MOCK DATA ENGINE (FALLBACK FOR DEMO) -----------------

def get_mock_analysis(report_text: str) -> ReportAnalysisResult:
    """
    Generates mock analysis for common surgical reports.
    Provides immediate usability when no local LLM or API Key is active.
    """
    text_lower = report_text.lower()
    
    if "cholecystectomy" in text_lower or "gallbladder" in text_lower:
        # Laparoscopic Cholecystectomy Mock
        summary = SurgicalSummary(
            preoperative_diagnosis="Acute cholecystitis with cholelithiasis",
            postoperative_diagnosis="Acute cholecystitis with cholelithiasis",
            procedures_performed=["Laparoscopic cholecystectomy"],
            surgical_team="Dr. Sarah Jenkins, MD (Surgeon); Dr. Robert Patel, MD (Assistant)",
            anesthesia="General endotracheal anesthesia",
            key_findings=[
                "Gallbladder was acutely distended, erythematous, and bound by dense inflammatory adhesions to the omentum.",
                "Multiple faceted gallstones were palpated in the gallbladder body.",
                "Cystic duct and cystic artery were isolated, clipped, and divided without incident."
            ],
            specimens_removed=["Gallbladder"],
            estimated_blood_loss="Minimal (less than 20 mL)",
            complications="None",
            operative_narrative_summary="Pneumoperitoneum was established. The gallbladder was retracted. The triangle of Calot was dissected, isolating the cystic duct and artery. These structures were secured using titanium clips and divided. The gallbladder was dissected off the cystic plate and extracted via the umbilical port."
        )
        
        icd10 = ICD10PCSCode(
            code="0FT44ZZ",
            procedure_name="Laparoscopic cholecystectomy",
            confidence=0.98,
            justification="Complete removal of the gallbladder (Resection) via laparoscopic approach (Percutaneous Endoscopic) with no device and no qualifier.",
            breakdown=[
                ICD10CharBreakdown(position=1, character="0", name="Section", value_description="Medical and Surgical"),
                ICD10CharBreakdown(position=2, character="F", name="Body System", value_description="Hepatobiliary System and Pancreas"),
                ICD10CharBreakdown(position=3, character="T", name="Root Operation", value_description="Resection (Cutting out or off, without replacement, all of a body part)"),
                ICD10CharBreakdown(position=4, character="4", name="Body Part", value_description="Gallbladder"),
                ICD10CharBreakdown(position=5, character="4", name="Approach", value_description="Percutaneous Endoscopic (Laparoscopic)"),
                ICD10CharBreakdown(position=6, character="Z", name="Device", value_description="No Device"),
                ICD10CharBreakdown(position=7, character="Z", name="Qualifier", value_description="No Qualifier")
            ]
        )
        return ReportAnalysisResult(summary=summary, icd10_codes=[icd10])
        
    elif "appendectomy" in text_lower or "appendix" in text_lower:
        # Open Appendectomy Mock
        summary = SurgicalSummary(
            preoperative_diagnosis="Acute appendicitis",
            postoperative_diagnosis="Acute ruptured appendicitis",
            procedures_performed=["Open appendectomy"],
            surgical_team="Dr. Marcus Aurelius, MD (Surgeon)",
            anesthesia="General anesthesia",
            key_findings=[
                "Appendix was noted to be highly inflamed, gangrenous, and perforated at the tip.",
                "Localized purulent fluid was present in the right lower quadrant, which was aspirated and cultured."
            ],
            specimens_removed=["Appendix"],
            estimated_blood_loss="Minimal (approx. 30 mL)",
            complications="None",
            operative_narrative_summary="A McBurney incision was made in the right lower quadrant. The appendix was identified and mobilized. The mesoappendix was dissected, and the appendiceal artery was clamped, divided, and ligated with 2-0 silk sutures. A crush clamp was applied to the base of the appendix, and a 2-0 chromic tie was placed around the crushed base. The appendix was amputated. The appendiceal stump was cauterized and inverted into the cecum with a purse-string suture of 3-0 silk."
        )
        
        icd10 = ICD10PCSCode(
            code="0DTJ0ZZ",
            procedure_name="Open appendectomy",
            confidence=0.95,
            justification="Complete removal of the appendix (Resection) via an open incision (Open) with no device and no qualifier.",
            breakdown=[
                ICD10CharBreakdown(position=1, character="0", name="Section", value_description="Medical and Surgical"),
                ICD10CharBreakdown(position=2, character="D", name="Body System", value_description="Gastrointestinal System"),
                ICD10CharBreakdown(position=3, character="T", name="Root Operation", value_description="Resection (Cutting out or off, without replacement, all of a body part)"),
                ICD10CharBreakdown(position=4, character="J", name="Body Part", value_description="Appendix"),
                ICD10CharBreakdown(position=5, character="0", name="Approach", value_description="Open (Direct cutting through skin and subcutaneous layers)"),
                ICD10CharBreakdown(position=6, character="Z", name="Device", value_description="No Device"),
                ICD10CharBreakdown(position=7, character="Z", name="Qualifier", value_description="No Qualifier")
            ]
        )
        return ReportAnalysisResult(summary=summary, icd10_codes=[icd10])
        
    elif "knee" in text_lower or "arthroplasty" in text_lower:
        # Total Knee Arthroplasty Mock
        summary = SurgicalSummary(
            preoperative_diagnosis="Severe osteoarthritis, left knee",
            postoperative_diagnosis="Severe osteoarthritis, left knee",
            procedures_performed=["Total knee arthroplasty, left knee"],
            surgical_team="Dr. Linda Hamilton, MD (Surgeon); Dr. Ken Wahl, MD (Anesthesiologist)",
            anesthesia="Spinal anesthesia with IV sedation",
            key_findings=[
                "Severe tricompartmental cartilage loss, particularly pronounced on the medial tibial plateau.",
                "Large osteophytic spurs on both distal femur and proximal tibia.",
                "Valgus deformity corrected during reconstruction."
            ],
            specimens_removed=["Resected bone fragments and osteophytes"],
            estimated_blood_loss="Approximately 100 mL",
            complications="None",
            operative_narrative_summary="A midline longitudinal incision was made. The knee joint was entered via a medial parapatellar arthrotomy. Standard cuts were made to the distal femur and proximal tibia. Sizing was confirmed, and cement was mixed. The femoral and tibial components and polyethylene insert were cemented into position. Patella resurfacing was performed."
        )
        
        icd10 = ICD10PCSCode(
            code="0SRD0JZ",
            procedure_name="Total knee arthroplasty, left knee",
            confidence=0.97,
            justification="Replacement of the left knee joint surface (Replacement) with a synthetic implant (Synthetic Substitute) via open approach (Open) with no qualifier.",
            breakdown=[
                ICD10CharBreakdown(position=1, character="0", name="Section", value_description="Medical and Surgical"),
                ICD10CharBreakdown(position=2, character="S", name="Body System", value_description="Lower Joints"),
                ICD10CharBreakdown(position=3, character="R", name="Root Operation", value_description="Replacement (Putting in or on biological or synthetic material that physically takes the place and/or function of all or a portion of a body part)"),
                ICD10CharBreakdown(position=4, character="D", name="Body Part", value_description="Knee Joint, Left"),
                ICD10CharBreakdown(position=5, character="0", name="Approach", value_description="Open"),
                ICD10CharBreakdown(position=6, character="J", name="Device", value_description="Synthetic Substitute"),
                ICD10CharBreakdown(position=7, character="Z", name="Qualifier", value_description="No Qualifier")
            ]
        )
        return ReportAnalysisResult(summary=summary, icd10_codes=[icd10])
        
    else:
        # General Mock fallback
        summary = SurgicalSummary(
            preoperative_diagnosis="Suspected localized pathology",
            postoperative_diagnosis="Localized tissue inflammation, benign",
            procedures_performed=["Diagnostic biopsy of local tissue"],
            surgical_team="Dr. John Doe, MD (Surgeon)",
            anesthesia="Local infiltration anesthesia",
            key_findings=["Slight tissue swelling and hypervascularity", "No macroscopically malignant tissue observed"],
            specimens_removed=["Tissue biopsy specimen"],
            estimated_blood_loss="Negligible",
            complications="None",
            operative_narrative_summary="Local anesthetic was infiltrated. A small incision was made, and a small biopsy of the inflamed tissue was obtained. Hemostasis was secured using electrocautery, and the skin was closed with a simple suture."
        )
        
        icd10 = ICD10PCSCode(
            code="0HBTXZZ",
            procedure_name="Biopsy of skin/subcutaneous tissue",
            confidence=0.85,
            justification="Excision of a portion of skin/subcutaneous tissue for diagnostic purposes (Excision, Qualifier: Diagnostic) using external approach (External) with no device.",
            breakdown=[
                ICD10CharBreakdown(position=1, character="0", name="Section", value_description="Medical and Surgical"),
                ICD10CharBreakdown(position=2, character="H", name="Body System", value_description="Skin and Breast"),
                ICD10CharBreakdown(position=3, character="B", name="Root Operation", value_description="Excision (Cutting out or off, without replacement, a portion of a body part)"),
                ICD10CharBreakdown(position=4, character="T", name="Body Part", value_description="Subcutaneous Tissue and Fascia, Trunk"),
                ICD10CharBreakdown(position=5, character="X", name="Approach", value_description="External"),
                ICD10CharBreakdown(position=6, character="Z", name="Device", value_description="No Device"),
                ICD10CharBreakdown(position=7, character="Z", name="Qualifier", value_description="No Qualifier")
            ]
        )
        return ReportAnalysisResult(summary=summary, icd10_codes=[icd10])
