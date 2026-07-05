import streamlit as st
import pandas as pd
import json
import os
import ollama
from dotenv import load_dotenv

# Set page config
st.set_page_config(
    page_title="SurgiCoder AI - Clinical Report Coding Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
default_api_key = os.getenv("GEMINI_API_KEY", "")

# Import engines
from nlp_engine import extract_entities, render_html_markup
from ai_engine import (
    analyze_surgical_report_ollama, 
    analyze_surgical_report_gemini, 
    get_mock_analysis, 
    ReportAnalysisResult
)

# Custom Premium Styling Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Page Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header Container Styling */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .app-title {
        color: #f8fafc;
        font-size: 36px;
        font-weight: 700;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin: 0;
    }
    
    /* Status indicators */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid;
    }
    
    .status-ready {
        background-color: rgba(16, 185, 129, 0.08);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.2);
    }
    
    .status-mock {
        background-color: rgba(245, 158, 11, 0.08);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.2);
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        display: inline-block;
    }
    
    .pulse-green {
        background-color: #34d399;
        animation: pulse-green-anim 1.8s infinite;
    }
    
    .pulse-orange {
        background-color: #fbbf24;
        animation: pulse-orange-anim 1.8s infinite;
    }
    
    @keyframes pulse-green-anim {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.5); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }
    
    @keyframes pulse-orange-anim {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.5); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(251, 191, 36, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(251, 191, 36, 0); }
    }
    
    /* Summary Cards */
    .summary-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .card-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        border-bottom: 1px solid #334155;
        padding-bottom: 4px;
    }
    
    .card-value {
        color: #f1f5f9;
        font-size: 15px;
        line-height: 1.6;
    }
    
    /* Code Breakdown Grid Styles */
    .breakdown-grid {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 10px;
        margin: 16px 0;
    }
    
    .breakdown-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 12px 6px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .breakdown-box:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    .breakdown-label {
        font-size: 10px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .breakdown-char {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .breakdown-desc {
        font-size: 11px;
        color: #cbd5e1;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- PRELOADED OPERATIVE REPORT SAMPLES -----------------
SAMPLES = {
    "Laparoscopic Cholecystectomy": """OPERATIVE REPORT

PREOPERATIVE DIAGNOSIS: Symptomatic cholelithiasis, acute cholecystitis.
POSTOPERATIVE DIAGNOSIS: Symptomatic cholelithiasis, acute cholecystitis.
PROCEDURE PERFORMED: Laparoscopic cholecystectomy.
SURGEON: Dr. Sarah Jenkins, MD
ANESTHESIA: General endotracheal anesthesia.
SPECIMENS: Gallbladder.
ESTIMATED BLOOD LOSS: 15 mL.
COMPLICATIONS: None.

OPERATIVE INDICATIONS: The patient is a 45-year-old female who presented to the emergency room with severe right upper quadrant pain radiating to her back, accompanied by nausea. Ultrasound demonstrated thickened gallbladder wall and cholelithiasis. Consent was obtained.

OPERATIVE SUMMARY: The patient was brought to the operating room, placed in the supine position, and general anesthesia was induced. The abdomen was prepped and draped in the usual sterile manner. 

A 10-mm umbilical incision was made, and pneumoperitoneum was established using carbon dioxide gas to a pressure of 15 mmHg. A 10-mm laparoscope was introduced, and abdominal inspection showed no visceral injury. Three additional ports were placed under direct vision: a 10-mm port subxiphoid, and two 5-mm ports in the right upper quadrant.

The gallbladder was retracted superiorly. Moderate inflammatory adhesions between the gallbladder and omentum were dissected using a harmonic scalpel. Calot's triangle was dissected, clearly isolating the cystic duct and cystic artery. The critical view of safety was fully achieved. Two titanium clips were placed proximally on the cystic duct and one distally, and it was divided. The cystic artery was clipped twice proximally, once distally, and divided.

The gallbladder was then dissected off the liver bed using electrocautery. Hemostasis of the liver bed was confirmed. The gallbladder was placed in an EndoBag and retrieved through the umbilical incision. The abdomen was irrigated. All ports were removed, CO2 evacuated, and incisions closed. The patient tolerated the procedure well and was transferred to the recovery room.""",

    "Open Appendectomy": """OPERATIVE REPORT

PREOPERATIVE DIAGNOSIS: Acute appendicitis.
POSTOPERATIVE DIAGNOSIS: Acute gangrenous appendicitis.
PROCEDURE PERFORMED: Open appendectomy.
SURGEON: Dr. Marcus Aurelius, MD
ANESTHESIA: General anesthesia.
SPECIMENS: Appendix.
ESTIMATED BLOOD LOSS: 25 mL.
COMPLICATIONS: None.

OPERATIVE INDICATIONS: The patient is an 18-year-old male presenting with classical symptoms of acute appendicitis, including migratory right lower quadrant pain, localized tenderness at McBurney's point, and leukocytosis.

OPERATIVE SUMMARY: The patient was positioned supine, anesthetized, prepped, and draped. A 5-cm oblique incision was made over McBurney's point. The incision was carried down through the skin and subcutaneous tissue. The external oblique aponeurosis was split along the line of its fibers, and the internal oblique and transversus abdominis muscles were parted bluntly. The peritoneum was identified, grasped, and opened.

On entering the peritoneal cavity, a small amount of purulent fluid was encountered and aspirated. The cecum was identified and mobilized. The appendix was noted to be swollen, highly injected, and gangrenous at the distal tip, but without frank rupture. 

The appendix was delivered into the wound. The mesoappendix was dissected, and the appendiceal artery was clamped, divided, and ligated with 2-0 silk sutures. A crush clamp was applied to the base of the appendix, and a 2-0 chromic tie was placed around the crushed base. The appendix was amputated. The appendiceal stump was cauterized and inverted into the cecum with a purse-string suture of 3-0 silk.

The surgical field was irrigated with warm saline. The wound was closed in layers: the peritoneum was closed with 2-0 Vicryl, muscles closed in layers, and the skin closed with subcuticular monocryl. The patient was extubated and transferred in stable condition.""",

    "Total Knee Arthroplasty (Left)": """OPERATIVE REPORT

PREOPERATIVE DIAGNOSIS: Severe osteoarthritis, left knee.
POSTOPERATIVE DIAGNOSIS: Severe osteoarthritis, left knee.
PROCEDURE PERFORMED: Total knee arthroplasty, left knee.
SURGEON: Dr. Linda Hamilton, MD
ANESTHESIA: Spinal anesthesia with IV sedation.
SPECIMENS: Resected bone fragments.
ESTIMATED BLOOD LOSS: 100 mL.
COMPLICATIONS: None.

OPERATIVE INDICATIONS: The patient is a 68-year-old male with progressive, debilitating pain in the left knee, unresponsive to conservative treatments. Preoperative radiographs showed severe medial joint space narrowing and bone-on-bone contact.

OPERATIVE SUMMARY: The patient was identified, spinal anesthesia administered, and a tourniquet applied to the left thigh. The left lower extremity was prepped and draped.

The tourniquet was inflated to 250 mmHg. A midline longitudinal skin incision was made over the patella, and entry into the joint was achieved using a medial parapatellar arthrotomy. The patella was everted laterally. Large osteophytes were removed from the femur and tibia.

The distal femoral cutting guide was aligned, and distal femoral bone cuts were made. Sizing was performed, and anterior, posterior, and chamfer cuts were completed. Attention was turned to the tibia. A tibial cutting guide was positioned, and proximal tibial cuts were executed perpendicular to the mechanical axis.

Trial components were inserted: a size 5 femoral component, a size 5 tibial component, and a 10-mm polyethylene insert. Knee range of motion showed excellent stability, tracking, and full extension. The joint was copiously irrigated.

Bone surfaces were dried. Methyl methacrylate cement was prepared. The femoral component and tibial tray were cemented in place. The excess cement was cleared. The patella was prepared and resurfaced with a cemented polyethylene patellar component. After cement curing, the tourniquet was deflated. Excellent hemostasis was verified.

The joint was irrigated again. The medial parapatellar incision was closed with 1-0 Vicryl. Subcutaneous tissue was approximated, and the skin closed with surgical staples. A sterile dressing was applied."""
}

# ----------------- SIDEBAR & BACKEND CONFIGURATION -----------------

st.sidebar.title("🛠️ AI Codification Engine")
backend_choice = st.sidebar.radio(
    "Choose LLM Backend",
    ["☁️ Google Gemini API (Cloud Ready)", "🖥️ Local Ollama (HIPAA-Safe / Offline)"],
    index=0,
    help="Select whether to use the cloud-based Google Gemini API or a local offline Ollama LLM setup."
)

api_key = ""
model_option = ""
force_mock = False
ollama_host = ""

if "gemini" in backend_choice.lower():
    api_key = st.sidebar.text_input(
        "Gemini API Key", 
        value=default_api_key, 
        type="password",
        help="Input your Gemini API key from Google AI Studio. If empty, the app defaults to demo mock mode."
    )
    model_option = st.sidebar.selectbox(
        "Select Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )
    if not api_key.strip():
        st.sidebar.warning("No API Key entered. Running in Mock fallback mode.")
        force_mock = True
else:
    ollama_host = st.sidebar.text_input(
        "Ollama Host URL", 
        value="http://localhost:11434",
        help="Endpoint for local Ollama server."
    )
    
    # Check Ollama connection status
    ollama_connected = False
    downloaded_models = []
    try:
        client = ollama.Client(host=ollama_host)
        models_data = client.list()
        downloaded_models = [m.get("model") for m in models_data.get("models", [])]
        ollama_connected = True
    except Exception:
        pass
        
    if ollama_connected and downloaded_models:
        model_option = st.sidebar.selectbox(
            "Select Ollama Model",
            downloaded_models,
            index=0 if "meditron:latest" not in downloaded_models else downloaded_models.index("meditron:latest")
        )
        force_mock = st.sidebar.checkbox("Force Demo Mode (Mock data)", value=False)
    else:
        st.sidebar.warning("🔌 Local Ollama Offline. Defaulting to mock demo fallback mode.")
        model_option = "Mock fallback mode"
        force_mock = True

st.sidebar.divider()
st.sidebar.title("📄 Report Input Selector")
template_selection = st.sidebar.selectbox(
    "Choose Template / Input Method",
    ["Select a template...", "Laparoscopic Cholecystectomy", "Open Appendectomy", "Total Knee Arthroplasty (Left)", "Custom Surgical Report"],
    index=0
)

# Populate report text area based on dropdown selection
report_placeholder = "Paste your surgical or operative report details here..."
if template_selection in SAMPLES:
    initial_text = SAMPLES[template_selection]
else:
    initial_text = ""

# ----------------- APP HEADER RENDERING -----------------

# Render status indicators depending on chosen backend and status
if "gemini" in backend_choice.lower() and not force_mock:
    status_html = (
        f'<div class="status-pill status-ready">'
        f'<span class="pulse-dot pulse-green"></span>'
        f'Gemini Active ({model_option})'
        f'</div>'
    )
elif "ollama" in backend_choice.lower() and ollama_connected and not force_mock:
    status_html = (
        f'<div class="status-pill status-ready">'
        f'<span class="pulse-dot pulse-green"></span>'
        f'Local Ollama Active ({model_option})'
        f'</div>'
    )
else:
    status_html = (
        '<div class="status-pill status-mock">'
        '<span class="pulse-dot pulse-orange"></span>'
        'SurgiCoder Demo Mode (Mock Fallback)'
        '</div>'
    )

header_html = (
    f'<div class="app-header">'
    f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">'
    f'<div>'
    f'<h1 class="app-title">🏥 SurgiCoder AI</h1>'
    f'<p class="app-subtitle">Surgical Report Parser & Structured ICD-10-PCS Auto-Coding Dashboard</p>'
    f'</div>'
    f'<div>{status_html}</div>'
    f'</div>'
    f'</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

# ----------------- MAIN REPORT INPUT SECTION -----------------

st.subheader("✍️ Operative Report Entry")
report_input = st.text_area(
    "Input Surgical Report", 
    value=initial_text, 
    height=250, 
    placeholder=report_placeholder
)

run_cols = st.columns([1, 4])
with run_cols[0]:
    analyze_button = st.button("🚀 Analyze & Map Codes", width="stretch", type="primary")

# Session state initialization to store analysis outputs
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "nlp_entities" not in st.session_state:
    st.session_state.nlp_entities = None

# Button click handler
if analyze_button:
    if not report_input.strip() or report_input == report_placeholder:
        st.error("Please insert surgical report text to analyze.")
    else:
        # Run local fast NLP
        with st.spinner("Extracting clinical concepts with local spaCy model..."):
            try:
                entities = extract_entities(report_input)
                st.session_state.nlp_entities = entities
            except Exception as nlp_err:
                st.error(f"Local NLP extraction warning: {nlp_err}")
                st.session_state.nlp_entities = []

        # Run AI mapping
        with st.spinner(f"Generating clinical summary and mapping codes..."):
            try:
                if force_mock:
                    results = get_mock_analysis(report_input)
                elif "gemini" in backend_choice.lower():
                    results = analyze_surgical_report_gemini(report_input, api_key.strip(), model_option)
                else:
                    results = analyze_surgical_report_ollama(report_input, model_option, ollama_host)
                st.session_state.analysis_results = results
            except Exception as ai_err:
                st.error(f"Execution Error: {ai_err}")
                st.session_state.analysis_results = None

# ----------------- DISPLAY INTERACTIVE TABS -----------------

if st.session_state.analysis_results:
    tab1, tab2, tab3 = st.tabs([
        "📊 Clinical Summary & ICD-10-PCS Codes", 
        "🔍 Highlighted Medical Entities (NLP)", 
        "💾 Export & JSON Output"
    ])
    
    res = st.session_state.analysis_results
    
    with tab1:
        # Clinical Summary Subsection
        st.subheader("📋 Structured Clinical Summary")
        
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            st.markdown(f"""
            <div class="summary-card">
                <div class="card-title">Pre-operative Diagnosis</div>
                <div class="card-value">{res.summary.preoperative_diagnosis}</div>
            </div>
            <div class="summary-card">
                <div class="card-title">Post-operative Diagnosis</div>
                <div class="card-value">{res.summary.postoperative_diagnosis}</div>
            </div>
            <div class="summary-card">
                <div class="card-title">Procedures Performed</div>
                <div class="card-value">
                    <ul style="margin: 0; padding-left: 20px;">
                        {"".join(f"<li>{p}</li>" for p in res.summary.procedures_performed)}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with sum_col2:
            st.markdown(f"""
            <div class="summary-card">
                <div class="card-title">Surgical Team & Anesthesia</div>
                <div class="card-value">
                    <strong>Team:</strong> {res.summary.surgical_team}<br/>
                    <strong>Anesthesia:</strong> {res.summary.anesthesia}
                </div>
            </div>
            <div class="summary-card">
                <div class="card-title">Metrics & Safety</div>
                <div class="card-value">
                    <strong>Estimated Blood Loss (EBL):</strong> {res.summary.estimated_blood_loss}<br/>
                    <strong>Complications:</strong> {res.summary.complications}
                </div>
            </div>
            <div class="summary-card">
                <div class="card-title">Specimens Removed</div>
                <div class="card-value">
                    {", ".join(res.summary.specimens_removed) if res.summary.specimens_removed else "None"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="summary-card">
            <div class="card-title">Operative Narrative Summary</div>
            <div class="card-value" style="font-style: italic;">"{res.summary.operative_narrative_summary}"</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical Findings bullets
        if res.summary.key_findings:
            st.markdown("##### 📌 Key Findings & Observations")
            for find in res.summary.key_findings:
                st.markdown(f"- {find}")
                
        st.divider()
        
        # ICD-10-PCS Auto-Coding Subsection
        st.subheader("🏷️ Suggested ICD-10-PCS Procedure Codes")
        
        if not res.icd10_codes:
            st.warning("No ICD-10-PCS code mappings generated.")
        else:
            for idx, code_obj in enumerate(res.icd10_codes):
                # Clean and capitalize code
                cleaned_code = code_obj.code.strip().upper()
                
                # Render code card header
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0f172a 0%, #020617 100%); border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-top: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 10px;">
                        <span style="font-size: 20px; font-weight: 700; color: #38bdf8;">Code Mapped: {cleaned_code}</span>
                        <span style="background-color: rgba(56, 189, 248, 0.1); color: #38bdf8; font-size: 13px; font-weight: 600; padding: 3px 10px; border-radius: 9999px; border: 1px solid rgba(56, 189, 248, 0.2);">
                            Confidence: {code_obj.confidence * 100:.1f}%
                        </span>
                    </div>
                    <div style="margin-top: 12px; color: #e2e8f0; font-size: 15px;">
                        <strong>Procedure:</strong> {code_obj.procedure_name}
                    </div>
                    <div style="margin-top: 6px; color: #94a3b8; font-size: 14px; line-height: 1.5;">
                        <strong>Justification:</strong> {code_obj.justification}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual Grid Breakdown of the 7 characters
                st.markdown("###### 🔍 7-Character Axis Code Breakdown")
                
                # Make sure the breakdown is sorted and complete
                breakdown_list = sorted(code_obj.breakdown, key=lambda x: x.position)
                
                # Generate Grid items
                grid_items = ""
                for char in breakdown_list:
                    grid_items += f"""
                    <div class="breakdown-box">
                        <div class="breakdown-label">Pos {char.position}</div>
                        <div class="breakdown-char">{char.character}</div>
                        <div class="breakdown-label">{char.name}</div>
                        <div class="breakdown-desc" title="{char.value_description}">{char.value_description}</div>
                    </div>
                    """
                
                # Render grid
                st.markdown(f'<div class="breakdown-grid">{grid_items}</div>', unsafe_allow_html=True)
                
                # Render a detailed mapping table for readability/export
                table_data = []
                for char in breakdown_list:
                    table_data.append({
                        "Position": f"Char {char.position}",
                        "Character Code": char.character,
                        "ICD-10-PCS Axis Name": char.name,
                        "Axis Description Value": char.value_description
                    })
                
                st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)
                st.write("")
                
    with tab2:
        # Highlighted Medical Entities via local spacy
        st.subheader("🔍 Local NER Concept Visualizer")
        st.caption("Extracted and classified using spaCy's biomedical engine (en_core_sci_sm) running locally.")
        
        # Legend of entity types
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; border-radius: 4px; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); display: inline-block;"></span>
                <span style="font-size: 13px; color: #60a5fa; font-weight: 600;">Surgical Procedure</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; border-radius: 4px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); display: inline-block;"></span>
                <span style="font-size: 13px; color: #34d399; font-weight: 600;">Anatomy / Organ</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; border-radius: 4px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); display: inline-block;"></span>
                <span style="font-size: 13px; color: #f87171; font-weight: 600;">Pathology / Diagnosis</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; border-radius: 4px; background: rgba(167, 139, 250, 0.15); border: 1px solid rgba(167, 139, 250, 0.4); display: inline-block;"></span>
                <span style="font-size: 13px; color: #a78bfa; font-weight: 600;">General Clinical Concept</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.nlp_entities is not None:
            html_markup = render_html_markup(report_input, st.session_state.nlp_entities)
            st.markdown(html_markup, unsafe_allow_html=True)
        else:
            st.info("Run analysis to view entities.")
            
    with tab3:
        st.subheader("💾 Export Structured Data")
        st.caption("Download the compiled output schema matching standard billing/coding ingestion schemas.")
        
        # Format result to dict/json string
        output_dict = res.model_dump()
        # Add NLP entities
        if st.session_state.nlp_entities:
            output_dict["extracted_entities"] = st.session_state.nlp_entities
            
        json_str = json.dumps(output_dict, indent=2)
        
        # Download button
        st.download_button(
            label="⬇️ Download Analysis JSON",
            data=json_str,
            file_name="surgical_analysis_output.json",
            mime="application/json",
            width="stretch"
        )
        
        st.markdown("##### Structured JSON Payload Preview")
        st.code(json_str, language="json")
else:
    st.info("👈 Choose a template report or input your own surgical text in the sidebar / input field, then click 'Analyze & Map Codes' to begin.")
