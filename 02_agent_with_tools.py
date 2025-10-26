"""
Exemplo 02 - Agente com ferramentas externas (LangGraph)
Demonstra integração com uma função Python customizada.
"""
from langgraph.graph import Graph
from langgraph.graph.nodes import ToolNode, LLMNode
from langchain_openai import ChatOpenAI

# Função ferramenta
def calcular_media(valores: list[float]) -> float:
    return sum(valores) / len(valores)

# Nó da ferramenta
tool_node = ToolNode(tools={"media": calcular_media})

# Nó do modelo
model = ChatOpenAI(model="gpt-4o-mini")
llm_node = LLMNode(model, prompt="Use a ferramenta 'media' para calcular a média de {valores}")

# Grafo
graph = Graph()
graph.add_node("ferramenta", tool_node)
graph.add_node("modelo", llm_node)
graph.connect("modelo", "ferramenta")  # saída do modelo → ferramenta
graph.set_entry_point("modelo")

resultado = graph.invoke({"valores": [10, 20, 30, 40]})
print(resultado)
