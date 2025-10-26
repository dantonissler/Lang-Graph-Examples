# 📄 CV LangGraph Pipeline
Pipeline completo com **LangGraph** para **análise de currículos em PDF**, extraindo campos essenciais e gerando saídas estruturadas (JSON por candidato, JSONL de auditoria e Excel para BI).

## ✨ Destaques
- 🧩 **Ontologia YAML** com *regex* determinísticas.
- 🤖 **IA generativa (OpenAI via LangChain)** para enriquecimento semântico (skills, senioridade, resumo).
- 🧠 **LangGraph** como orquestrador (com fallback sequencial).
- 🧾 Saídas: `output/json/*.json`, `output/resultado.jsonl`, `output/log.xlsx`.

## 📂 Estrutura
```
cv_langgraph_pipeline/
├── Data/                  # coloque aqui os PDFs dos currículos
├── output/
│   ├── json/
│   ├── resultado.jsonl
│   ├── log.xlsx
│   └── ontology.yaml
├── cv_pipeline.py
├── requirements.txt
└── README.md
```

## 🚀 Como rodar
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="sua_chave"   # Windows (PowerShell): $Env:OPENAI_API_KEY="sua_chave"

python cv_pipeline.py
```

## 🧠 Campos extraídos (exemplos)
- `nome_candidato`, `email`, `telefone`, `linkedin`, `github`
- `escolaridade`, `anos_experiencia`, `ultima_empresa`, `ultimo_cargo`
- `habilidades_tecnicas` (regex) + **enriquecimento LLM**: `top_skills`, `senioridade`, `resumo_profissional`

## 🛠️ Customização
Edite `output/ontology.yaml` para ajustar *regex* e chaves. Você também pode ampliar o nó `llm_enrich` para retornar outros campos.

## 🔒 Observações
- Entrada suportada: **apenas `.pdf`** (texto nativo ou com OCR).
- Para PDFs imagem, instale Tesseract localmente e ajuste o PATH se necessário.
