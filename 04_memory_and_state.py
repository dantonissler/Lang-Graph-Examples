"""
Exemplo 04 - Memória e estado entre execuções.
Mostra como o grafo pode armazenar e atualizar o contexto.
"""
from langgraph.graph import Graph, State
from datetime import datetime

def registrar(state: State):
    historico = state.get("historico", [])
    nova_entrada = f"[{datetime.now().strftime('%H:%M:%S')}] {state['mensagem']}"
    historico.append(nova_entrada)
    state["historico"] = historico
    return state

graph = Graph()
graph.add_node("registrar", registrar)
graph.set_entry_point("registrar")

estado = {"mensagem": "Usuário iniciou a conversa"}
estado = graph.invoke(estado)
estado["mensagem"] = "Usuário solicitou ajuda técnica"
estado = graph.invoke(estado)

print("Histórico armazenado:")
for h in estado["historico"]:
    print(" -", h)
