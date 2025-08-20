from langchain_groq import ChatGroq
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
import os
# import logging : not required rn


class ChatBotState(TypedDict):
    """State schema for the chatbot"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    keyword: str

class MyHelpfulBot():
    def __init__(self, model="deepseek-r1-distill-llama-70b", persist_directory="podak"):
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        self.llm = ChatGroq(
            model=model,
            temperature=0.4,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )
        self.embedfn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda:0"})
        
        self.vectorstores = {
            "podak": Chroma(
                collection_name="podak", 
                embedding_function=self.embedfn, 
                persist_directory=persist_directory, 
                collection_metadata={"hnsw:space": "cosine"})
        }
        print("Number of documents in 'podak' store:", self.vectorstores["podak"]._collection.count())


        self.tools = [self.create_find_context_tool(self.vectorstores)]
        self.memory = InMemorySaver()
        self.workflow = self._create_workflow()
        self.config = {"configurable": {"thread_id": "conversation-1"}}

    def create_find_context_tool(self, vectorstores):
        @tool
        def find_context(query: str, collname: str = "podak", no_of_docs: int = 10, keyword : list = []) -> str:
            """Searches Kotak Mahindra Bank's official documents for verified answers. 
    
        Parameters:
        - query (str): The exact customer question to search for
        - keyword (str): Extracted keyword(s) from user input for filtering

        Returns:
        - str: Document excerpts or "No results found"
            """

            try:
                collname = 'podak'
                if collname not in vectorstores:
                    return f"Collection '{collname}' not found. Available collections: {list(vectorstores.keys())}"

                # print('Keyword : ', keyword)
                # or_q = [{'$contains' : key} for key in keyword]

                # print('query : ', query)
                # print('keyword in :', keyword)
                    
                
                matches = vectorstores[collname].similarity_search(query = query, k = no_of_docs, filter = {'keyword' : {'$in' : keyword}}
                                                                   )
                if not matches:
                    return f"No relevant documents found in {collname} for query: {query}"
                
                content = "\n\n".join([doc.page_content for doc in matches])
                # print('Content : ', content)
                return f"{content}"
            except Exception as e:
                #logger.error(f"Error in find_context: {e}")
                # print(f"Error in find_context: {e}")
                return f"Error retrieving context: {str(e)}"
        return find_context
    
    def _summarize(self, state: ChatBotState) -> ChatBotState:  #make it async?
        """Running keyword of the chat history."""
        # incomplete for now, unnecessary for short conversations
        keyword = self.get_keyword(state)
        print('Summarize key : ', keyword)
        return {**state, "keyword": keyword}

    
    def _react_agent_node(self, state: ChatBotState) -> ChatBotState:
        """Agent for document querying and reasoning."""

        user_input = state["user_input"]
        messages = state.get("messages", [])
        keyword = state["keyword"]


        # recent conversation context
        context_messages = []
        for msg in messages:
            # passing all messages for now, assuming small conversations
            if isinstance(msg, (HumanMessage, AIMessage)):
                role = "Human" if isinstance(msg, HumanMessage) else "AI"
                context_messages.append(f"{role}: {msg.content}")
        
        conversation_context = "\n".join(context_messages) if context_messages else ""
        
        # system_message = f"""You are a helpful assistant with access to document search tools. 
        # First you are to think about which document to search based on the conversation context and current user input.
        # Then, use the find_context tool to find relevant context based on user query.
        # Finally provide comprehensive answers based on the retrieved context. 
        # Be polite, respectful and accurate. If no relevant information is found, say so clearly.

        # Available documents = {list(self.vectorstores.keys())}

        # Conversation Context: 
        # {conversation_context}

        # Keyword:
        # {keyword}
        
        # Available tools: {[f"{tool.name}: {tool.description}" for tool in self.tools]}
        
        # Think step by step and use tools when needed to answer the user's question."""

        system_message = f"""You are a helpful digital assistant for Kotak Mahindra Bank products and services. Your purpose is to provide accurate information to customers in a friendly, easy-to-understand manner.


*Keyword*:{keyword}

**You MUST follow these steps for EVERY query:**
1. **Search First**: Use `find_context` tool with the user's exact query.
2. **Analyze**: Check if the tool returned valid Kotak documents.
3. **Respond**: Answer ONLY using the tool's output. If none found, say so.



**Special Cases Handling:**
- For greetings (hi/hello) or non-banking queries:
  * DO NOT call tools
  * Respond with generic welcome message
  * Example: "Hello! How can I assist you with Kotak banking today?"
- For Bank timings:
    * Respond with 9 to 5 time.
    * Ask the user to contact the local branch office.

        
**Good Response Example:**
"To check your account balance, you can use the Kotak Mobile Banking app. Just log in and your balance will show on the dashboard. You can also get mini-statements there."

**Bad Response Example:**
"Account balances are visible in mobile banking." 
(Too vague, didn't verify with search tool)

**When Information is Unavailable:**
1. First state: "Let me check that for you..." (while running search tool)
2. If nothing found: "I couldn't find this in current resources. For help, you can:"
   - Visit kotak.com
   - Call 1860 266 2666
   - Message in the mobile app

**Tone Rules:**
- Use natural language like "you'll" instead of "you will"
- Keep responses conversational (2-3 sentences max)
- Explain terms simply: "FD means Fixed Deposit - like a savings account that earns higher interest"

**Safety Protocol:**
If any doubt after searching:
1. Admit it: "I want to confirm this for you..."
2. Run the search tool again with different keywords
3. If still unsure: "For accurate help, please contact customer care at 1860 266 2666"

**Tool Usage Requirement:**
- **Every single query** must trigger the search tool first
- Never answer without running the search
- Add the user query in the tool.
- If the tool fails, say: "I'm having trouble accessing that information right now. Please try [alternative option]"

Example Tool Call:
find_context(query="...", keyword="['Date & Time']")

- The keword parameter takes a list of str.

**Final Reminder:**
You exist solely to:
1. Run the search tool
2. Interpret official documents
3. Respond conversationally
4. Escalate when needed
"""

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
                        "keyword": existing_state.values.get("keyword", "")
                    }
                else:
                    # start new conversation
                    curr_state = {
                        "messages": [],
                        "user_input": user_input,                        
                        "keyword": ""
                    }
            except Exception as e:
                print(f"Could not retrieve existing state: {e}, starting fresh")
                # fallback to new conversation
                curr_state = {
                    "messages": [],
                    "user_input": user_input,                    
                    "keyword": ""
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
        
    def get_keyword(self, state: ChatBotState) -> str:
        """Get updated keyword"""
        keyword = ['Shared Visions & Goals', 'Vision & Mission', 'Culture and Employee Attraction in Financial Services', 'Compliance and Trust', "Citizen's Charter for Banks", 'Confidentiality & Usage Restrictions', 'KYC & AML Procedures', 'Identity Verification & Compliance', 'Financial Products & Services', 'Banking Services', 'Banking Service Operations', 'Customer Remedies', 'Customer Service Policies & Ombudsman Scheme Publicization', 'Customer Service Initiatives', 'Bank Website Access Policy', 'Bank Website Access', 'Bank Compensation Policy Access', 'Kotak Bank Security Repossession Policy Access', 'Bank Privacy Policy', 'Privacy Policy', 'Privacy Commitments', 'Consent for Information Disclosure', 'Legal Protections & Defenses', 'Consent and Disclosure', 'Third Party Support Services', 'Privacy Policy & Charter Access', 'Bank Service Standards', 'Services for Vulnerable Groups', 'Banking Facilities for Disabled', 'Bank Death Claim Process & Info Link', 'Bank Communication & Regulations', 'Bank Services', 'Locker Facilities Guide', 'SA/CA Account Safe Deposit Requirements', 'Remittance Services', 'Backup Plans & Power Supply', 'Net Banking Features', 'Kotak Card Services & Features', 'Mobile Banking Features', 'Date & Times', 'WhatsApp Banking Services', 'Email Alerts & Preferences', 'Bank Procedures for NACH Compliance', 'Loan Pricing & Terms', 'Loan Information', 'Branch Office Operations & Customer Service', 'Deliverable Timing', 'Dining Locations', 'FCY Cash Withdrawal & Deposit Timings', 'Date & Times', 'Customer Expectations in Banking', 'Cheque Handling Procedures', 'Cheque Rejection Conditions', 'Secure Financial Transactions', 'Bank Rules', 'Bank Account Actions', 'Customer Service Complaints Timeline', 'Financial Security Measures', 'Bank Alerts Tracking', 'Bank Security Measures', 'Secure Internet Banking Access', 'Secure Online Banking Practices', 'Financial Year Rules', 'Bank Customer Services', 'Customer Feedback & Satisfaction', 'Complaint Resolution Instructions', 'Online Grievance Redressal System', 'ATM Complaint Resolution', 'Grievance Filing Instructions', 'Banking Complaints Resolution', 'Bank Services & Commitments', 'Account Branch Access Methods', 'Reserve Bank of India Exchange Facilities', 'Currency Exchanges', 'Bank Note Exchange Facilities', 'Bank Rules for Mutilated Notes', 'Forgery Handling Procedures', 'Reserve Bank of India Security Features', 'Exchange Issues Reporting', 'Customer Service Politeness', 'Damaged Currency Transactions', 'Reserve Bank Branch Services', 'Bank Branch Exchanges', 'Public Feedback', 'Pensioner Services', 'Civil Pension Benefits', 'Age Ranges & Life Expectancy Milestones', 'Pensions Percentages', 'Pension Credit Notifications & Processes', 'RBI Issue Offices Jurisdictions', 'Location Information', 'Location & Contact Details', 'RBI Issue Department Location', 'Contact Details', 'Geographical Locations', 'Telephone Bhavan, Chandigarh Locations', 'Location', 'Address Information', 'Address & Position', 'General Manager Location & Designation', 'Contact Details & Leadership', 'Uttar Pradesh & Uttarakhand Locations', 'Post Bag Location & DGM Responsibility', 'Locations & GMs', 'Contact Information', 'RBI Personnel Locations & Titles', 'Location & Position at RBI Issue Dept', 'Bakery Department Address', 'Confidentiality & Distribution Restrictions', 'Customer Rights Policies', 'Customer Rights & Fairness', 'Fair Treatment Practices', 'Bank Obligations', 'Ethical Practices & Communication Guidelines', 'Communication Policies & Obligations', 'Customer Product Suitability Policies', 'Confidentiality Conditions', 'Customer Information Disclosure Conditions', 'Bank Compensation Policies', 'Grievance Redressal Processes', 'Bank Complaint Handling Timelines', 'Dispute Resolution & Liability for Banks', 'Refund Policy & Bank Inquiries', 'Confidentiality & Usage Restrictions', 'Grievance Redressal Policy & Processes', 'Fair Treatment & Complaint Handling Principles', 'Customer Support Processes', 'Customer Feedback Processes', 'Continuous Improvement for Customer Experience', 'Banking Complaints & Requests', 'Policy Coverage', 'Scope & Applicability', 'Feedback & Complaint Methods', 'Contact Methods & Concern Registration', 'Complaint Escalation Procedures', 'Customer Issue Resolution', 'Complaint Escalation', 'Contact Details', 'Kotak Bank Contact Information', 'Grievance Redressal Process', 'Internal Ombudsman Role & Compliance', 'Internal Ombudsman Roles', 'Standing Committee Composition & Responsibilities', 'Feedback Evaluation', 'Customer Service Compliance Responsibilities', 'Branch-Level CS Committees Purpose & Establishment', 'Customer Service Committees', 'Customer Service Oversight & Policy Development Processes', 'Customer Complaint Escalation in Banking', 'Complaints & Suggestions Arrangements', 'Resolution Times & Turnaround', 'Bank Complaints', 'Date & Times of TAT Measurement', 'Banking Txn Disputes Complaints', 'CRM System Efficiency', 'Complaints Received', 'Customer Complaint Resolution Process', 'Complaint Handling & MIS Processes', 'Customer Feedback & Complaint Handling', 'Customer Satisfaction & Efficiency Strategies', 'Grievance Handling & Complaints', 'Complaint Records RetentionAccessibility', 'Customer Satisfaction Surveys', 'Customer Feedback Improvement Strategies']
        query = state['user_input']

        keyword_prompt = ChatPromptTemplate.from_template(
            f"""
            **Task**: Extract the most relevant keyword(s) from the predefined list below that matches the user's query. 
            Return ONLY the exact keyword(s) or say "No match found".

            **Rules**:
            1. Strictly use ONLY the keywords from this list: {keyword}
            2. Ignore partial matches or synonyms. 
            3. Return maximum 5 keywords if multiple are relevant.

            **Query**: {query}

            **Output Format** (comma-separated or "No match found"):
            """
        )


        keyword_chain = keyword_prompt | self.llm

        result = keyword_chain.invoke({
            "keywords": ", ".join(keyword),
            "query": query
        })

        # print(result)
        # print('test : ', list(map(lambda x : x.strip(), result.content.split(','))))

        return list(map(lambda x : x.strip(), result.content.split(',')))
        
    def clear_memory(self):
        """Clear the conversation memory"""
        try:
            # new thread ID 
            import uuid
            self.config = {"configurable": {"thread_id": f"conversation-{uuid.uuid4()}"}}
            print("Memory cleared - started new conversation thread")
        except Exception as e:
            print(f"Error clearing memory: {e}")


if __name__ == "__main__":

    agent = MyHelpfulBot()
    #print(agent.find_context.args_schema.model_json_schema()) # can add Annotated args 
    print("Chatbot initialised! Type 'quit' to exit, 'clear' to clear memory.")
    print("You can ask questions about the documents in your collection.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                #print(agent.get_keyword())
                print(agent.get_full_conversation_history())
                break
            elif user_input.lower() == 'clear':
                agent.clear_memory()
                print("Memory cleared!")
                continue
            elif not user_input:
                continue
            
            # Get response from the bot
            start = time.time()
            response = agent.chat(user_input)
            time_taken = time.time()-start
            print(f"\nBot: {response}\n({time_taken} seconds)")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

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
