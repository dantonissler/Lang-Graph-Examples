"""
Exemplo 01 - Cadeia simples (LangGraph)
Demonstra uma pipeline básica com um único nó de prompt.
"""
from langgraph.graph import Graph
from langgraph.graph.nodes import PromptNode
from langchain_openai import ChatOpenAI

# Define modelo e nó
model = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptNode(model, "Explique de forma simples: {topico}")

# Constrói o grafo
graph = Graph()
graph.add_node("explicador", prompt)
graph.set_entry_point("explicador")

# Executa
resposta = graph.invoke({"topico": "aprendizado de máquina"})
print(resposta)
