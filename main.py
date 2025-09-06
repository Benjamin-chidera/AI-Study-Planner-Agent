import streamlit as st
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# langgraph
from langgraph.graph import StateGraph, START, END

st.title("AI Study Planner Agent")
st.subheader("Welcome to the AI Study Planner Agent")


class AgentState(TypedDict):
    topic: str
    study_plan: str
    
def study_topic(state: AgentState) -> AgentState:
    
    
    return state

def retrieve_topic(state: AgentState) -> AgentState:
    
    llm = ChatOllama(model="llama3.1")
    
    message = [
        (
            "system", """
                        You are AI assistant study planner agent who creates study plans related to a topic i give you.
                        
                        Study plans should include the following:
                        Topics to study
                        Books to read
                        Videos to watch
                        Tests to take
                        Assignments to complete
                        Exams to take
                        Projects to complete
                        hours to spend on each topic
                        
        
            """
        ),
        (
            "human", "{topic}"
        )
    ]
    
    chat = ChatPromptTemplate.from_messages(message)
    
    chain = chat | llm | StrOutputParser()
    
    res = chain.invoke({"topic": state["topic"]})
    
    # break res into chunks
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_text(res)
    
    for chunk in chunks:
        state["study_plan"] = chunk
        # st.write(chunk)
    
    
    
    return state    

def create_study_plan(state: AgentState) -> AgentState:
    
    # display the study plan in streamlit
    
    st.write(state["study_plan"])
    
    return state


# initialize state
graph = StateGraph(AgentState)

# add nodes to graph
graph.add_node("topic", study_topic)
graph.add_node("retrieve_topic", retrieve_topic)
graph.add_node("create_study_plan", create_study_plan)

# add edges to graph
graph.add_edge(START, "topic")
graph.add_edge("topic", "retrieve_topic")
graph.add_edge("retrieve_topic", "create_study_plan")
graph.add_edge("create_study_plan", END)

# compile graph
compiled_graph = graph.compile()



topic = st.text_input("Enter your topic:")

create_plan = st.button("Create Plan")

if create_plan:
    if topic:
        compiled_graph.invoke({"topic": topic, "study_plan": ""})
        