"""
Exemplo 05 - Streaming em tempo real (LangGraph)
Mostra como o grafo pode emitir respostas parciais.
"""
from langgraph.graph import Graph
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
graph = Graph()
graph.add_node("modelo", model)
graph.set_entry_point("modelo")

print("Streaming iniciado:\n")
for token in graph.stream({"input": "Explique resumidamente o que é IA generativa."}):
    print(token, end="", flush=True)
print("\n\nStreaming finalizado.")
