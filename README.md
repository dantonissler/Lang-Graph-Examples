````markdown
# 🧩 LangGraph Examples — Estudos com Grafos de IA

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Active-green.svg)

---

## 🎯 Objetivo

Este repositório demonstra, de forma progressiva, como criar **pipelines inteligentes baseados em grafos** com o framework **LangGraph** — usado em **aplicações de IA com estado, memória e ferramentas**.

---

## 📂 Estrutura

| Arquivo | Conceito |
|----------|-----------|
| `01_basic_chain.py` | Cadeia simples com um nó de prompt. |
| `02_agent_with_tools.py` | Agente que chama funções externas (tools). |
| `03_branching_graph.py` | Fluxo condicional de execução. |
| `04_memory_and_state.py` | Manutenção de memória e estado. |
| `05_streaming_workflow.py` | Streaming de tokens (respostas em tempo real). |

---

## 🚀 Como executar

1️⃣ Crie um ambiente:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\\Scripts\\activate    # Windows
````

2️⃣ Instale dependências:

```bash
pip install -r requirements.txt
```

3️⃣ Configure sua variável de ambiente com a API da OpenAI:

```bash
export OPENAI_API_KEY="sua_chave_aqui"
```

4️⃣ Execute um exemplo:

```bash
python 01_basic_chain.py
```

---

## 🧠 Conceitos principais

* **LangGraph** — constrói fluxos dinâmicos de IA baseados em grafos.
* **LLMNode / ToolNode / PromptNode** — nós executáveis que interagem com modelos ou funções.
* **State** — mantém o contexto e resultados intermediários.
* **Streaming** — respostas em tempo real.
* **Branching** — caminhos diferentes dependendo do estado.

---

## 👨‍💻 Autor

**Danton Rodrigues**
💼 *Estudos em Inteligência Artificial, Automação e Grafos de Linguagem*
📫 [GitHub](https://github.com/dantonissler)

---

> “O poder do LangGraph está em unir a lógica de fluxo de trabalho com a inteligência dos modelos de linguagem.”

```

---

Posso gerar **todos esses 5 arquivos + o README e requirements.txt** automaticamente e entregar em um `.zip` pronto pra você subir no GitHub (`langgraph_examples.zip`).  
Quer que eu crie agora?
```
