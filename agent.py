from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from tools import *

#################
## Agent State ##
#################
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

#################
## Build Graph ##
#################
def build_graph(llm):
    """Build a graph

    Returns:
        _type_: _description_
    """
    
    tools = [visit_website_tool,
             search_and_read_web_tool, 
             read_text_tool,
             read_csv_tool, 
             read_excel_tool, 
             extract_text_from_image_assistant,
             describe_image_assistant,
             download_gaia_attachment,
             analyze_mp3_tool]
    
    chat_with_tools = llm.bind_tools(tools)



    def assistant(state: AgentState):
        return {"messages": [chat_with_tools.invoke(state["messages"])]}

    # The graph for the agent is built using a StateGraph, 
    # which allows us to define nodes and edges for the control flow.
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))


    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition
    )
    builder.add_edge("tools", "assistant")

    return builder.compile()


######################
## Agent Definition ##
######################
class BasicAgent:
    """LangGraph agent"""

    def __init__(self, llm):
        print("BasicAgent initialized.")
        self.graph = build_graph(llm)
        self.llm = llm
        self.system_prompt = (
            """You are a general AI assistant. I will ask you a question and you can use tools when relevant. Reply with the answer and nothing else. YOUR ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number, neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string. If unable to analyze a video or image, analyze any textual clues instead.
                        ATTACHMENT PRIORITY:
                            - If the question mentions an attached file, attachment, or filename, you MUST inspect that file before considering web search.
                            - Your first tool call MUST be download_gaia_attachment, passing the filename as [task_id].extension.
                            - After downloading, use the appropriate tool to analyse the file (read_excel_tool, read_csv_tool, analyze_mp3_tool, etc.).
                            - Do NOT use web search to find information that is supposed to be contained in the attachment.
                            - Only use web search if the question explicitly requires external information beyond the attachment.
                        WEB SEARCH RESTRICTIONS:
                            - Use web search only when external information is necessary to answer the question.
                            - Make at most THREE web search calls per task.
                            - Each search must address a specific missing piece of information.
                            - Do not repeat searches with slightly different wording.
                            - If a search returns relevant information, use it. Do not search again just to confirm it.
                            - If a search fails or returns insufficient information, stop searching and provide your best-supported answer.
                            - If a direct URL is provided, call visit_website_tool exactly once first, passing the plain URL.
                        Further rules for tool use:
                            - Use the minimum number of tool calls necessary to establish the answer reliably.
                            - For multi-step questions, identify the necessary subquestions, solve them in order, and use each result to guide the next step. Keep intermediate results internally.
                            - For questions involving identifying an entity through a statistic and retrieving another attribute, first identify the correct entity, then retrieve the requested attribute for that same entity and time period.
                            - Check whether tool results actually contain relevant evidence. Use useful results rather than searching for the same information again.
                            - If a source points to another relevant primary source, follow that source before considering a web-search fallback. Only use the fallback if the expected source cannot be located or accessed.
                            - Never repeat a tool call unless it failed technically and retrying is necessary.
                            - Do not infer specific facts from titles or descriptions without supporting evidence. If the answer cannot be established, provide the best-supported answer from available information.
                            - Before each tool call, identify what information is missing and whether the call is necessary and permitted.
                            - Once the answer is sufficiently supported, stop and return only the answer. Do not explain your research process.
                            - You don't have chess capabilities. If asked about chess, respond with your best guess based on general knowledge, and do not attempt to browse chess-specific sources.
            """)

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question}...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question),
        ]
        try:
            response = self.graph.invoke({"messages": messages},
                                     config={"recursion_limit": 40})

            answer = response["messages"][-1].text

            print(f"Agent returning answer: {answer}")
            return answer

        except GraphRecursionError:
            print("Recursion limit reached. Asking for best guess...")
    
            # Ask the LLM directly, without binding tools
            final_messages = messages + [
                SystemMessage(content=(
                    "You have reached the tool-use limit. "
                    "Do not call any more tools. Based on the information "
                    "already available in the conversation, provide your "
                    "best-supported answer. If uncertain, make your best guess. "
                    "Return only the answer."
                ))
            ]
    
            response = self.llm.model.invoke(final_messages)
            answer = response.text
            print(f"Agent returning answer: {answer}")
            return answer

    