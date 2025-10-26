"""
Exemplo 03 - Fluxo condicional (Branching)
Mostra bifurcação de caminhos baseada na análise do texto.
"""
from langgraph.graph import Graph, State
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

def analisar_sentimento(state: State):
    texto = state["mensagem"].lower()
    if any(p in texto for p in ["bom", "ótimo", "excelente"]):
        state["resultado"] = "positivo"
    elif any(p in texto for p in ["ruim", "péssimo", "horrível"]):
        state["resultado"] = "negativo"
    else:
        state["resultado"] = "neutro"
    return state

graph = Graph()
graph.add_node("analisar", analisar_sentimento)
graph.set_entry_point("analisar")

mensagem = "O atendimento foi excelente!"
saida = graph.invoke({"mensagem": mensagem})
print(saida)
