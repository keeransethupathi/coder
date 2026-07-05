import html
import re

# Global cache for the NLP model
_nlp = None
spacy_available = False

try:
    import spacy
    spacy_available = True
except ImportError:
    pass

def get_nlp_model():
    """Loads and caches the spaCy model with error tolerance."""
    global _nlp, spacy_available
    if not spacy_available:
        return None
        
    if _nlp is None:
        try:
            # Try to load the scientific model if available
            _nlp = spacy.load("en_core_sci_sm")
        except Exception:
            try:
                # Try fallback to standard small model
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                _nlp = None
    return _nlp

def classify_entity(text):
    """
    Categorizes generic scientific entities into distinct clinical classes
    using high-performance medical suffix and keyword heuristics.
    """
    text_lower = text.lower()
    
    procedure_hints = [
        "ectomy", "otomy", "plasty", "oscopy", "rraphy", "pexy", "centesis",
        "repair", "excision", "excis", "resection", "resect", "incision", "incis", 
        "insertion", "insert", "removal", "remov", "dissection", "dissect", "biopsy", 
        "drainage", "drain", "bypass", "closure", "close", "debridement", "debrid",
        "amputation", "anastomosis", "reconstruction", "reconstruct", "suture", "sutur",
        "irrigate", "irrigat", "prep"
    ]
    
    anatomy_hints = [
        "gallbladder", "appendix", "colon", "knee", "joint", "bone", "artery",
        "vein", "muscle", "nerve", "skin", "liver", "stomach", "lung", "heart",
        "abdomen", "abdominal", "peritoneal", "fascia", "duct", "cystic", "bile",
        "intestine", "bowel", "sternum", "tissue", "lobes", "tendon", "ligament",
        "cartilage", "femur", "tibia", "patella", "meniscus"
    ]
    
    pathology_hints = [
        "itis", "osis", "pathy", "oma", "emia", "uria", "iasis",
        "bleeding", "rupture", "calculus", "calculi", "stone", "stones", "mass",
        "hernia", "inflammation", "lesion", "tear", "fracture", "infection",
        "adhesions", "effusion", "obstruction", "stenosis", "ischemia", "necrosis"
    ]
    
    if any(hint in text_lower for hint in procedure_hints):
        return "PROCEDURE"
    elif any(hint in text_lower for hint in pathology_hints):
        return "PATHOLOGY"
    elif any(hint in text_lower for hint in anatomy_hints):
        return "ANATOMY"
    else:
        return "CLINICAL_CONCEPT"

def extract_entities(text):
    """
    Extracts medical entities from the raw report text.
    Falls back to a high-speed regex keyword scanner if spaCy is unavailable.
    """
    if not text.strip():
        return []
        
    nlp = get_nlp_model()
    
    # If spaCy model loaded successfully, use it
    if nlp is not None:
        try:
            doc = nlp(text)
            entities = []
            seen_spans = set()
            
            for ent in doc.ents:
                start, end = ent.start_char, ent.end_char
                if any(start >= s and end <= e for s, e in seen_spans):
                    continue
                seen_spans.add((start, end))
                
                ent_text = ent.text.strip()
                if not ent_text:
                    continue
                    
                category = classify_entity(ent_text)
                entities.append({
                    "start": start,
                    "end": end,
                    "text": ent_text,
                    "label": category
                })
            entities.sort(key=lambda x: x["start"])
            return entities
        except Exception:
            pass
            
    # Fallback: High-speed keyword scanner (perfect for offline/unconfigured cloud containers)
    entities = []
    keywords = [
        ("laparoscopic cholecystectomy", "PROCEDURE"),
        ("cholecystectomy", "PROCEDURE"),
        ("appendectomy", "PROCEDURE"),
        ("arthroplasty", "PROCEDURE"),
        ("cholecystitis", "PATHOLOGY"),
        ("appendicitis", "PATHOLOGY"),
        ("osteoarthritis", "PATHOLOGY"),
        ("gallbladder", "ANATOMY"),
        ("appendix", "ANATOMY"),
        ("knee", "ANATOMY"),
        ("liver parenchyma", "ANATOMY"),
        ("mesoappendix", "ANATOMY"),
        ("patella", "ANATOMY"),
        ("bleeding", "PATHOLOGY")
    ]
    
    # Simple regex scanner to find start and end indices of terms
    for term, label in keywords:
        for match in re.finditer(re.escape(term), text, re.IGNORECASE):
            entities.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "label": label
            })
            
    # Sort and remove overlapping matches
    entities.sort(key=lambda x: x["start"])
    refined_entities = []
    last_end = -1
    for ent in entities:
        if ent["start"] >= last_end:
            refined_entities.append(ent)
            last_end = ent["end"]
            
    return refined_entities

def render_html_markup(text, entities):
    """
    Generates premium HTML markup highlighting clinical concepts in glassmorphic badges.
    """
    if not text:
        return ""
        
    style_map = {
        "PROCEDURE": {
            "bg": "rgba(59, 130, 246, 0.15)",      # Translucent blue
            "border": "rgba(59, 130, 246, 0.4)",
            "color": "#60a5fa",
            "badge_bg": "rgba(59, 130, 246, 0.25)"
        },
        "PATHOLOGY": {
            "bg": "rgba(239, 68, 68, 0.15)",      # Translucent red
            "border": "rgba(239, 68, 68, 0.4)",
            "color": "#f87171",
            "badge_bg": "rgba(239, 68, 68, 0.25)"
        },
        "ANATOMY": {
            "bg": "rgba(16, 185, 129, 0.15)",      # Translucent green
            "border": "rgba(16, 185, 129, 0.4)",
            "color": "#34d399",
            "badge_bg": "rgba(16, 185, 129, 0.25)"
        },
        "CLINICAL_CONCEPT": {
            "bg": "rgba(167, 139, 250, 0.15)",    # Translucent purple
            "border": "rgba(167, 139, 250, 0.4)",
            "color": "#a78bfa",
            "badge_bg": "rgba(167, 139, 250, 0.25)"
        }
    }
    
    html_parts = []
    current_idx = 0
    
    # Ensure entities are sorted and formatted correctly
    # Supports both Pydantic models (with start, end attributes) and dicts (with keys)
    formatted_ents = []
    for ent in entities:
        if hasattr(ent, "start"):
            formatted_ents.append({
                "start": ent.start,
                "end": ent.end,
                "text": ent.text,
                "label": ent.label
            })
        elif isinstance(ent, dict):
            formatted_ents.append(ent)
            
    formatted_ents.sort(key=lambda x: x["start"])
    
    for ent in formatted_ents:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        
        # Verify indices range bounds
        if start < current_idx or end > len(text) or start > end:
            continue
            
        # Add non-entity text preceding this entity
        if start > current_idx:
            html_parts.append(html.escape(text[current_idx:start]))
            
        # Add highlighted entity
        ent_text = text[start:end]
        styles = style_map.get(label, style_map["CLINICAL_CONCEPT"])
        
        highlighted = (
            f'<mark style="'
            f'background-color: {styles["bg"]}; '
            f'border: 1px solid {styles["border"]}; '
            f'color: {styles["color"]}; '
            f'padding: 3px 6px; '
            f'margin: 0 2px; '
            f'border-radius: 6px; '
            f'display: inline-flex; '
            f'align-items: center; '
            f'gap: 6px; '
            f'font-weight: 500;'
            f'">'
            f'{html.escape(ent_text)}'
            f'<span style="'
            f'font-size: 0.7em; '
            f'text-transform: uppercase; '
            f'letter-spacing: 0.05em; '
            f'background-color: {styles["badge_bg"]}; '
            f'color: {styles["color"]}; '
            f'padding: 1px 4px; '
            f'border-radius: 4px; '
            f'font-weight: 700;'
            f'">'
            f'{label.replace("_", " ")}'
            f'</span>'
            f'</mark>'
        )
        html_parts.append(highlighted)
        current_idx = end
        
    # Append the remaining text
    if current_idx < len(text):
        html_parts.append(html.escape(text[current_idx:]))
        
    rendered = "".join(html_parts).replace("\n", "<br/>")
    
    wrapper_html = (
        f'<div style="'
        f'background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); '
        f'border: 1px solid #334155; '
        f'border-radius: 12px; '
        f'padding: 20px; '
        f'color: #e2e8f0; '
        f'font-family: \'Outfit\', \'Segoe UI\', sans-serif; '
        f'line-height: 1.8; '
        f'font-size: 15px; '
        f'box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'
        f'">'
        f'{rendered}'
        f'</div>'
    )
    return wrapper_html
