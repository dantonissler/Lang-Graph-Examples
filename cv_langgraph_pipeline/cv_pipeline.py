# -*- coding: utf-8 -*-
# cv_pipeline.py — Pipeline LangGraph para análise de currículos (PDF)
from pathlib import Path
import os, re, json, yaml
from typing import Any, Optional, List, Dict
from datetime import datetime

import pandas as pd

# ============== 1) SETUP ==============
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "output"
OUT_JSON_DIR = OUTPUT_DIR / "json"

JSONL_PATH = OUTPUT_DIR / "resultado.jsonl"
EXCEL_PATH = OUTPUT_DIR / "log.xlsx"
ONTOLOGY_PATH = OUTPUT_DIR / "ontology.yaml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("[SETUP]")
print("  BASE_DIR     :", BASE_DIR)
print("  DATA_DIR     :", DATA_DIR, "| exists:", DATA_DIR.exists())
print("  OUTPUT_DIR   :", OUTPUT_DIR, "| exists:", OUTPUT_DIR.exists())

# ============== 2) DEPENDÊNCIAS ==============
OCR_AVAILABLE = True
PDF_RENDER_AVAILABLE = True
try:
    import pypdf
except Exception as e:
    raise RuntimeError("Instale pypdf: pip install pypdf") from e

try:
    import pypdfium2 as pdfium
except Exception as e:
    PDF_RENDER_AVAILABLE = False
    print("[WARN] pypdfium2 ausente — OCR pode ficar limitado.")

try:
    import pytesseract
    from PIL import Image
except Exception:
    OCR_AVAILABLE = False
    print("[WARN] pytesseract/Pillow ausentes — sem OCR.")

# LangGraph (com fallback)
try:
    from langgraph.graph import StateGraph, START, END
    USING_LANGGRAPH = True
except Exception as e:
    USING_LANGGRAPH = False
    print("[LangGraph] WARN:", e, "→ usando fallback sequencial.")

    class StateGraph:
        def __init__(self, state_cls): self.nodes = {}
        def add_node(self, name, fn): self.nodes[name] = fn
        def add_edge(self, a, b): pass  # sequencial simples
        def compile(self):
            def _runner(state):
                for n in list(self.nodes.keys()):
                    state = self.nodes[n](state)
                return state
            return type("MockApp", (object,), {"invoke": _runner})
    START = "START"; END = "END"

# LLM (OpenAI via LangChain)
USE_LLM = True
try:
    from langchain_openai import ChatOpenAI
except Exception as e:
    USE_LLM = False
    print("[LLM] WARN: langchain-openai não disponível. Enriquecimento LLM será ignorado.")

# ============== 3) ONTOLOGIA DEFAULT ==============
DEFAULT_ONTOLOGY = {
    "entities": {
        "nome_candidato": {
            "patterns_any": [r"(?i)\b(nome|candidate|candidato|curriculum vitae)\s*[:\-]?\s*([A-ZÁ-Ú][A-Za-zÁ-Úá-ú\s'\-]{3,})"],
            "value_format": "{g2}", "type": "text"
        },
        "email": {
            "patterns_any": [r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"],
            "value_format": "{match}", "type": "text"
        },
        "telefone": {
            "patterns_any": [r"(?i)(\+?\d{1,3}\s*)?(\(?\d{2}\)?\s*)?\d{4,5}[-\s]?\d{4}"],
            "value_format": "{match}", "type": "text"
        },
        "linkedin": {
            "patterns_any": [r"(?i)https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_\/]+"],
            "value_format": "{match}", "type": "text"
        },
        "github": {
            "patterns_any": [r"(?i)https?://(www\.)?github\.com/[A-Za-z0-9\-_]+"],
            "value_format": "{match}", "type": "text"
        },
        "escolaridade": {
            "patterns_any": [r"(?i)\b(gradua[cç][aã]o|bacharelado|licenciatura|p[oó]s[- ]gradua[cç][aã]o|especializa[cç][aã]o|mestrado|doutorado)\b"],
            "value_format": "{match}", "type": "label"
        },
        "anos_experiencia": {
            "patterns_any": [r"(?i)(\d{1,2})\s*(anos?|yrs?)\s*(de\s*)?experi[êe]ncia"],
            "value_format": "{g1}", "type": "number"
        },
        "ultima_empresa": {
            "patterns_any": [r"(?i)(empresa|companhia|organization|org)\s*[:\-]\s*([A-ZÁ-Ú0-9][^\n]{2,60})"],
            "value_format": "{g2}", "type": "text"
        },
        "ultimo_cargo": {
            "patterns_any": [r"(?i)(cargo|posi[cç][aã]o|position|fun[cç][aã]o)\s*[:\-]\s*([A-ZÁ-Ú][^\n]{2,60})"],
            "value_format": "{g2}", "type": "text"
        },
        "habilidades_tecnicas": {
            "patterns_any": [r"(?i)\b(python|java|javascript|typescript|c\+\+|c#|go|rust|sql|postgresql|mysql|mongodb|redis|docker|kubernetes|aws|gcp|azure|spark|hadoop|pandas|numpy|scikit-learn|tensorflow|pytorch)\b"],
            "value_format": "{match}", "type": "list"
        }
    }
}

# ============== 4) MODELOS ==============
from pydantic import BaseModel, Field

class Section(BaseModel):
    title: str = ""
    start_idx: int = 0
    end_idx: int = 0
    text: str = ""

class Fator(BaseModel):
    key: str
    value: Any
    source: str = "rule"
    confidence: float = 1.0
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    evidence: Optional[str] = None

class PipelineState(BaseModel):
    filepath: Optional[str] = None
    raw_text: str = ""
    sections: List[Section] = Field(default_factory=list)
    fatores: List[Fator] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)

# ============== 5) UTILS ==============
def read_pdf_text_native(p: Path) -> str:
    try:
        reader = pypdf.PdfReader(str(p))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""

PDF_RENDER_DPI = int(os.environ.get("PDF_RENDER_DPI", "300"))
OCR_LANGS = os.environ.get("OCR_LANGS", "por+eng")
OCR_MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", "300"))

def render_pdf_pages(p: Path, dpi: int = PDF_RENDER_DPI):
    if not PDF_RENDER_AVAILABLE:
        return []
    scale = max(1.0, float(dpi)/150.0)
    images = []
    pdf = pdfium.PdfDocument(str(p))
    for i in range(len(pdf)):
        pil = pdf[i].render(scale=scale).to_pil()
        images.append(pil)
    return images

def ocr_image(img: "Image.Image", lang: str = OCR_LANGS) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except Exception:
        try:
            return pytesseract.image_to_string(img)
        except Exception:
            return ""

def read_pdf_with_ocr_if_needed(p: Path) -> str:
    native = read_pdf_text_native(p)
    if len((native or "").strip()) >= OCR_MIN_CHARS:
        return native
    pages = render_pdf_pages(p)
    if not pages:  # sem render → retorna nativo mesmo
        return native
    ocr_texts = [ocr_image(img) for img in pages]
    merged = "\n\n".join(ocr_texts).strip()
    return merged or native

def naive_sections(text: str, max_len: int = 8000) -> List[Section]:
    # Divide por cabeçalhos comuns de currículo ou fatiamento
    markers = r"(?im)^\s*(experi[êe]ncia|experiences?|forma[cç][aã]o|education|skills|habilidades|projetos|projects)\b"
    parts = re.split(markers, text)
    if len(parts) <= 1:
        secs, start = [], 0
        for i in range(0, len(text), max_len):
            chunk = text[i:i+max_len]
            secs.append(Section(title=f"sec_{len(secs)+1}", start_idx=start, end_idx=start+len(chunk), text=chunk))
            start += len(chunk)
        return secs or [Section(title="full", start_idx=0, end_idx=len(text), text=text)]
    secs, start = [], 0
    it = iter(parts)
    lead = next(it)
    if lead.strip():
        secs.append(Section(title="intro", start_idx=start, end_idx=start+len(lead), text=lead)); start += len(lead)
    for title, chunk in zip(it, it):
        full = title + "\n" + chunk
        secs.append(Section(title=title, start_idx=start, end_idx=start+len(full), text=full))
        start += len(full)
    return secs

def dedupe_fatores(fatores: List[Fator]) -> List[Fator]:
    seen, out = set(), []
    for f in fatores:
        sig = (f.key, json.dumps(f.value, ensure_ascii=False, sort_keys=True))
        if sig in seen: continue
        seen.add(sig); out.append(f)
    return out

# ============== 6) ONTOLOGY LOADER ==============
def load_ontology() -> Dict[str, Any]:
    if not ONTOLOGY_PATH.exists():
        with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_ONTOLOGY, f, allow_unicode=True, sort_keys=False)
        print("[ONTOLOGY] Default gerado:", ONTOLOGY_PATH)
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

ONT = load_ontology()

# ============== 7) NÓS DO GRAFO ==============
def n_load_pdf(state: PipelineState) -> PipelineState:
    p = Path(state.filepath or "")
    if p.suffix.lower() != ".pdf":
        raise ValueError("Apenas .pdf é suportado.")
    text = read_pdf_with_ocr_if_needed(p)
    state.raw_text = text
    state.logs.append(f"load_pdf:{p.name}:{len(text)}")
    return state

def n_sectionize(state: PipelineState) -> PipelineState:
    state.sections = naive_sections(state.raw_text)
    state.logs.append(f"sectionize:{len(state.sections)}")
    return state

def _apply_entity_patterns(section: Section, key: str, spec: dict) -> List[Fator]:
    fatores = []
    pats = spec.get("patterns_any", []) or []
    ent_type = (spec.get("type") or "").lower()
    for pat in pats:
        for m in re.finditer(pat, section.text, flags=re.I):
            groups = m.groups() or ()
            if "value_format" in spec:
                fmt = spec["value_format"]
                repl = {"match": m.group(0)}
                for i, g in enumerate(groups, start=1):
                    repl[f"g{i}"] = g
                val = fmt.format(**repl)
            else:
                val = True if ent_type == "bool" else (groups[0] if groups else m.group(0))
            fatores.append(Fator(
                key=key, value=val, source="rule", confidence=1.0,
                span_start=section.start_idx + m.start(),
                span_end=section.start_idx + m.end(),
                evidence=section.text[max(0, m.start()-60): m.end()+60],
            ))
    return fatores

def n_apply_rules(state: PipelineState) -> PipelineState:
    fatores = []
    entities = ONT.get("entities", {}) or {}
    for sec in state.sections:
        for key, spec in entities.items():
            fatores.extend(_apply_entity_patterns(sec, key, spec))
    state.fatores = fatores
    state.logs.append(f"apply_rules:{len(fatores)}")
    return state

# ---- LLM enrichment ----
def n_llm_enrich(state: PipelineState) -> PipelineState:
    if not USE_LLM or not os.getenv("OPENAI_API_KEY"):
        state.logs.append("llm_enrich:skipped")
        return state
    try:
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
        prompt = (
            "Você é um recrutador técnico. A partir do texto abaixo, retorne um JSON conciso com campos "
            "top_skills (lista), senioridade (junior/pleno/senior), resumo_profissional (3-4 linhas), "
            "areas_de_interesse (lista). Texto do currículo:\n---\n{texto}\n---\nJSON:"
        )
        resp = llm.invoke(prompt.format(texto=state.raw_text))
        content = getattr(resp, "content", str(resp))
        m = re.search(r"\{.*\}", content, flags=re.S)
        data = json.loads(m.group(0)) if m else {"resposta": content}
        for k, v in data.items():
            state.fatores.append(Fator(key=f"llm_{k}", value=v, source="llm", confidence=0.85))
        state.logs.append("llm_enrich:ok")
    except Exception as e:
        state.logs.append(f"llm_enrich:error:{type(e).__name__}:{e}")
    return state

def n_normalize(state: PipelineState) -> PipelineState:
    normed = []
    skills = set()
    for f in state.fatores:
        v = f.value
        if f.key in {"email", "linkedin", "github"} and isinstance(v, str):
            v = v.strip().lower()
        if f.key == "habilidades_tecnicas":
            skills.add(str(v).lower())
            continue
        normed.append(Fator(**{**f.dict(), "value": v}))
    if skills:
        normed.append(Fator(key="habilidades_tecnicas", value=sorted(skills), source="rule_merge"))
    state.fatores = dedupe_fatores(normed)
    state.logs.append("normalize:emails+links+skills_merge")
    return state

def n_persist(state: PipelineState) -> PipelineState:
    record = {"file": Path(state.filepath).name if state.filepath else "", "fatores": [f.dict() for f in state.fatores], "logs": state.logs}
    with open(JSONL_PATH, "a", encoding="utf-8") as w:
        w.write(json.dumps(record, ensure_ascii=False) + "\n")
    per_file = OUT_JSON_DIR / (Path(state.filepath).stem + ".json")
    with open(per_file, "w", encoding="utf-8") as w:
        json.dump(record, w, ensure_ascii=False, indent=2)
    state.logs.append("persist:jsonl+json")
    return state

# ============== 8) EXECUÇÃO EM LOTE ==============
def _first(fatores: List[Fator], key: str):
    for f in fatores:
        if f.key == key:
            return f.value
    return None

def run_single(path: str | Path, app) -> PipelineState:
    st = PipelineState(filepath=str(path))
    out = app.invoke(st)
    return PipelineState(**out) if isinstance(out, dict) else out

def run_batch(app, data_dir: Path = DATA_DIR, clear_previous: bool = True) -> pd.DataFrame:
    files = sorted([*data_dir.glob("*.pdf")])
    print("[RUN] PDFs encontrados:", len(files))
    if clear_previous and JSONL_PATH.exists(): JSONL_PATH.unlink()
    rows = []
    for p in files:
        try:
            st = run_single(p, app)
            fatores = st.fatores
            row = {
                "file": p.name,
                "nome_candidato": _first(fatores, "nome_candidato"),
                "email": _first(fatores, "email"),
                "telefone": _first(fatores, "telefone"),
                "linkedin": _first(fatores, "linkedin"),
                "github": _first(fatores, "github"),
                "escolaridade": _first(fatores, "escolaridade"),
                "anos_experiencia": _first(fatores, "anos_experiencia"),
                "ultima_empresa": _first(fatores, "ultima_empresa"),
                "ultimo_cargo": _first(fatores, "ultimo_cargo"),
                "habilidades_tecnicas": _first(fatores, "habilidades_tecnicas"),
                "llm_top_skills": _first(fatores, "llm_top_skills"),
                "llm_senioridade": _first(fatores, "llm_senioridade"),
                "llm_resumo_profissional": _first(fatores, "llm_resumo_profissional"),
                "llm_areas_de_interesse": _first(fatores, "llm_areas_de_interesse"),
            }
            rows.append(row)
        except Exception as e:
            rows.append({"file": p.name, "error": f"{type(e).__name__}: {e}"})
    df = pd.DataFrame(rows)
    df.to_excel(EXCEL_PATH, index=False)
    print("[RUN] BI salvo em:", EXCEL_PATH)
    return df

# ============== 9) COMPILAR E RODAR ==============
if __name__ == "__main__":
    g = StateGraph(PipelineState)
    g.add_node("load_pdf", n_load_pdf)
    g.add_node("sectionize", n_sectionize)
    g.add_node("apply_rules", n_apply_rules)
    g.add_node("llm_enrich", n_llm_enrich)
    g.add_node("normalize", n_normalize)
    g.add_node("persist", n_persist)

    try:
        g.add_edge(START, "load_pdf")
        g.add_edge("load_pdf", "sectionize")
        g.add_edge("sectionize", "apply_rules")
        g.add_edge("apply_rules", "llm_enrich")
        g.add_edge("llm_enrich", "normalize")
        g.add_edge("normalize", "persist")
        g.add_edge("persist", END)
    except Exception:
        pass

    app = g.compile()
    print("[GRAPH] Compilado. Engine:", "LangGraph" if 'USING_LANGGRAPH' in globals() and USING_LANGGRAPH else "Fallback")
    print("Nós:", ["load_pdf", "sectionize", "apply_rules", "llm_enrich", "normalize", "persist"])

    df = run_batch(app, DATA_DIR)
    print("[DONE] Processados:", len(df))
