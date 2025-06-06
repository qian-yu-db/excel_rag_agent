from typing import Any, Generator, Optional, Sequence, Union
import uuid
import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################
#LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Update with your system prompt
system_prompt = """You are a helpful assistant"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
CATALOG = "fins_genai"
SCHEMA = "unstructured_data"

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
# for details
tools = []
vector_search_tools = [
        VectorSearchRetrieverTool(
        index_name=f"{CATALOG}.{SCHEMA}.excel_enriched_parsed_text_vs_index",
        tool_name="table_search",
        tool_description="Vector Search Retriever tool for insurance information tables",
    )
]
tools.extend(vector_search_tools)

#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)
    memory = MemorySaver()

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    # def chat_history_cutoff(state: ChatAgentState):
    #     state['messages'] = state['messages'][-5:]
    #     return state

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        #state = chat_history_cutoff(state)
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=memory)


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        config = {"configurable": {"thread_id": "1"}}
        for event in self.agent.stream(request, config, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        config = {"configurable": {"thread_id": "1"}}
        for event in self.agent.stream(request, config, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
