from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver # recommended to use (Async)PostgresSaver for production capabilities
from langchain_core.tools import tool
from typing import Annotated, List, TypedDict
from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, RemoveMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
import time
# import logging : not required rn


class ChatBotState(TypedDict):
    """State schema for the chatbot"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    summary: str

class MyHelpfulBot():
    def __init__(self, model="qwen2.5:7b", persist_directory="db"):
        self.llm = ChatOllama(model=model, temperature=0.1)
        self.embedfn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda:0"})
        
        self.vectorstores = {
            "mycoll": Chroma(
                collection_name="mycoll", 
                embedding_function=self.embedfn, 
                persist_directory=persist_directory, 
                collection_metadata={"hnsw:space": "cosine"}),

            "Seismic__horror": Chroma(
                collection_name="Seismic__horror", 
                embedding_function=self.embedfn, 
                persist_directory=persist_directory, 
                collection_metadata={"hnsw:space": "cosine"}),

            "Prison": Chroma(
                collection_name="Prison", 
                embedding_function=self.embedfn, 
                persist_directory=persist_directory, 
                collection_metadata={"hnsw:space": "cosine"})
            #add more ?
        }
        self.tools = [self.create_find_context_tool(self.vectorstores)]
        self.memory = InMemorySaver()
        self.workflow = self._create_workflow()
        self.config = {"configurable": {"thread_id": "conversation-1"}}

    def create_find_context_tool(self, vectorstores):
        @tool
        def find_context(query: str, collname: str = "mycoll", no_of_docs: int = 5) -> str:
            """Finding context via embeddings already stored in directory.
            
            Args:
                query: Input query by user
                collname: Name of the Chroma collection to search in. Available: Seismic__horror, mycoll, Prison
                no_of_docs: No. of documents to be queried upon similarity search
                
            Returns:
                Required context for the user query"""

            try:
                if collname not in vectorstores:
                    return f"Collection '{collname}' not found. Available collections: {list(vectorstores.keys())}"
                
                matches = vectorstores[collname].similarity_search(query, no_of_docs)
                if not matches:
                    return f"No relevant documents found in {collname} for query: {query}"
                
                content = "\n\n".join([doc.page_content for doc in matches])
                return f"Retrieved from {collname}:\n{content}"
            except Exception as e:
                #logger.error(f"Error in find_context: {e}")
                return f"Error retrieving context: {str(e)}"
        return find_context
    
    def _summarize(self, state: ChatBotState) -> ChatBotState:  #make it async?
        """Running summary of the chat history."""
        # incomplete for now, unnecessary for short conversations
        return {**state, "summary": ''}

    
    def _react_agent_node(self, state: ChatBotState) -> ChatBotState:
        """Agent for document querying and reasoning."""

        user_input = state["user_input"]
        messages = state.get("messages", [])
        summary = state["summary"]

        # recent conversation context
        context_messages = []
        for msg in messages:
            # passing all messages for now, assuming small conversations
            if isinstance(msg, (HumanMessage, AIMessage)):
                role = "Human" if isinstance(msg, HumanMessage) else "AI"
                context_messages.append(f"{role}: {msg.content}")
        
        conversation_context = "\n".join(context_messages) if context_messages else ""
        
        system_message = f"""You are a helpful assistant with access to document search tools. 
        First you are to think about which document to search based on the conversation context and current user input.
        Then, use the find_context tool to find relevant context based on user query.
        Finally provide comprehensive answers based on the retrieved context. 
        Be polite, respectful and accurate. If no relevant information is found, say so clearly.

        Available documents = {list(self.vectorstores.keys())}

        Conversation Context: 
        {conversation_context}

        Summary:
        {summary}
        
        Available tools: {[f"{tool.name}: {tool.description}" for tool in self.tools]}
        
        Think step by step and use tools when needed to answer the user's question.""" 

        agent_graph = create_langgraph_react_agent(
            self.llm, 
            self.tools, 
            prompt=system_message
        )
        
        try:
            
            agent_state = {"messages": [HumanMessage(content=user_input)]}
            result = agent_graph.invoke(agent_state)
            
            final_message = result["messages"][-1]
            response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            new_messages = [
                HumanMessage(content=user_input),
                AIMessage(content=response_content)
            ]
            
            existing_messages = state.get("messages", [])
            updated_messages = existing_messages + new_messages
            
            return {
                **state,
                "messages": updated_messages
            }
            
        except Exception as e:
            print(f"Encountered Exception in react agent node: {e}")
            error_response = f"I encountered an error while processing your request: {str(e)}"
            return {
                **state,
                "messages": [
                    HumanMessage(content=user_input),
                    AIMessage(content=error_response)
                ]
            }

    def _create_workflow(self) -> StateGraph:
        """Creating Langgraph workflow"""

        workflow = StateGraph(ChatBotState)
        
        workflow.add_node("summarize_messages", self._summarize)
        workflow.add_node("react_agent", self._react_agent_node)
        # workflow.add_node("conversation_history", self.get_conversation_history)
        
        workflow.add_edge(START, "summarize_messages")
        workflow.add_edge("summarize_messages", "react_agent")
        # workflow.add_edge("react_agent", "conversation_history")
        # workflow.add_edge("conversation_history", END)
        workflow.add_edge("react_agent", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def chat(self, user_input: str) -> str:
        """Main chat function using LangGraph memory management"""
        try:
            # getting existing state from memory or create initial state
            try:
                existing_state = self.workflow.get_state(self.config)
                if existing_state and existing_state.values:
                    # continue existing conversation
                    curr_state = {
                        "messages": existing_state.values.get("messages", []),
                        "user_input": user_input,            
                        "summary": existing_state.values.get("summary", "")
                    }
                else:
                    # start new conversation
                    curr_state = {
                        "messages": [],
                        "user_input": user_input,                        
                        "summary": ""
                    }
            except Exception as e:
                print(f"Could not retrieve existing state: {e}, starting fresh")
                # fallback to new conversation
                curr_state = {
                    "messages": [],
                    "user_input": user_input,                    
                    "summary": ""
                }
            
            result = self.workflow.invoke(curr_state, self.config)
            
            messages = result.get("messages", [])
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I'm sorry, I couldn't process your request properly."
                
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."
        
    def get_full_conversation_history(self):
        """Get the current conversation history"""
        try:
            existing_state = self.workflow.get_state(self.config)
            if existing_state and existing_state.values:
                messages = existing_state.values.get("messages", [])
                history = []
                for msg in messages:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        role = "Human" if isinstance(msg, HumanMessage) else "AI"
                        history.append(f"{role}: {msg.content}")
                return history
            else:
                print("No conversation history found.")
                return([])
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return([])
        
    def get_summary(self):
        """Get updated summary"""
        return 
        
    def clear_memory(self):
        """Clear the conversation memory"""
        try:
            # new thread ID 
            import uuid
            self.config = {"configurable": {"thread_id": f"conversation-{uuid.uuid4()}"}}
            print("Memory cleared - started new conversation thread")
        except Exception as e:
            print(f"Error clearing memory: {e}")


# if __name__ == "__main__":

#     agent = MyHelpfulBot(model="qwen2.5:3b")
#     #print(agent.find_context.args_schema.model_json_schema()) # can add Annotated args 
#     print("Chatbot initialised! Type 'quit' to exit, 'clear' to clear memory.")
#     print("You can ask questions about the documents in your collection.")

#     while True:
#         try:
#             user_input = input("\nYou: ").strip()
            
#             if user_input.lower() in ['quit', 'exit', 'q']:
#                 print("Goodbye!")
#                 #print(agent.get_summary())
#                 print(agent.get_full_conversation_history())
#                 break
#             elif user_input.lower() == 'clear':
#                 agent.clear_memory()
#                 print("Memory cleared!")
#                 continue
#             elif not user_input:
#                 continue
            
#             # Get response from the bot
#             start = time.time()
#             response = agent.chat(user_input)
#             time_taken = time.time()-start
#             print(f"\nBot: {response}\n({time_taken} seconds)")
            
#         except KeyboardInterrupt:
#             print("\nGoodbye!")
#             break
#         except Exception as e:
#             print(f"Error: {e}")
#             print("Please try again.")

    #print("Conversation history: \n\n")
    #print(agent.get_full_conversation_history())

'''
#testing the tool

if __name__ == "__main__":
    agent = MyHelpfulBot()
    
    find_context_tool = agent.create_find_context_tool(agent.vectorstores)
    
    # Test 1: Valid collection and query
    print("=== Test 1: Valid query ===")
    result = find_context_tool.invoke({
        "query": "prisoner", 
        "collname": "Prison", 
        "no_of_docs": 3
    })
    print(f"Result: {result}")
    print(f"Length: {len(result)} characters")
    
    # Test 2: Invalid collection
    print("\n=== Test 2: Invalid collection ===")
    result = find_context_tool.invoke({
        "query": "test", 
        "collname": "NonExistent", 
        "no_of_docs": 3
    })
    print(f"Result: {result}")
    
    # Test 3: Different collection
    print("\n=== Test 3: Different collection ===")
    result = find_context_tool.invoke({
        "query": "horror", 
        "collname": "Seismic__horror", 
        "no_of_docs": 2
    })
    print(f"Result: {result}")


agent = MyHelpfulBot(model="qwen2.5:3b")
print("init!")
print(agent.workflow.get_state(agent.config))
agent.clear_memory()
print(agent.workflow.get_state(agent.config))
print(agent.config["configurable"]["thread_id"])'''
