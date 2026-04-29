import os
from dotenv import load_dotenv
import chainlit as cl

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor

load_dotenv()

# ==============================================================================
# ENVIRONMENT TOPOLOGY
# ==============================================================================
#
# HERMES CORE ENVIRONMENT (WebLogic Instance #1)
#   - AdminServer  : 1 admin node
#   - WorkerNode1  : worker node
#   - WorkerNode2  : worker node
#   Deployments (on worker nodes):
#     * hermes_core
#     * hermes_services
#     * hermes_routs
#     * hermes_dispatcher
#
# HERMES APP ENVIRONMENT (WebLogic Instance #2)
#   - AdminServer  : 1 admin node
#   - WorkerNode1  : worker node  <-- expected to be RUNNING
#   - WorkerNode2  : worker node  <-- expected to be SHUTDOWN
#   Deployments:
#     * hermes_app (on WorkerNode1 only)
#
# ==============================================================================


# ==============================================================================
# TOOL MOCKS — hardcoded responses simulating a healthy environment
# ==============================================================================
#
# All tools return realistic but static mock data.
# Status tools always reflect the expected healthy state of each instance.
# Log tools return a plausible snippet of clean INFO-level log lines.
# Action tools (restart, clean) return a successful completion message.
# ==============================================================================


# ── 1. Admin Server Status ────────────────────────────────────────────────────

@tool
def get_admin_server_status(instance: str) -> str:
    """
    Return the current status of the WebLogic Admin Server for the given instance.

    Args:
        instance (str): Which WebLogic instance to query.
                        Accepted values: "core" | "app"
                          - "core" -> Hermes Core environment (4 deployments)
                          - "app"  -> Hermes App environment  (hermes_app only)

    Expected return format:
        A single line describing the Admin Server state.

    Example output:
        AdminServer  RUNNING  health=OK  uptime=14d 3h 22m

    The agent will interpret the state field and report whether the Admin Server
    is healthy, degraded, or down.
    """
    return "AdminServer  RUNNING  health=OK  uptime=14d 3h 22m"


# ── 2. Worker Node Status ─────────────────────────────────────────────────────

@tool
def get_node_status(instance: str) -> str:
    """
    Return the current status of all managed (worker) nodes in the given
    WebLogic instance.

    Args:
        instance (str): "core" | "app"

    Expected return format:
        One line per server: <ServerName>  <State>  <health>  <uptime>

    Example output (core instance):
        WorkerNode1  RUNNING  health=OK  uptime=14d 3h 22m
        WorkerNode2  RUNNING  health=OK  uptime=14d 3h 21m

    Example output (app instance -- only WorkerNode1 should be RUNNING):
        WorkerNode1  RUNNING   health=OK  uptime=14d 3h 22m
        WorkerNode2  SHUTDOWN  health=N/A

    States to watch for: RUNNING | STARTING | SUSPENDING | SUSPENDED |
                         SHUTTING_DOWN | SHUTDOWN | FAILED | UNKNOWN
    The agent will highlight any unexpected state (e.g. WorkerNode2 RUNNING in
    the app environment, or any node FAILED/UNKNOWN in any environment).
    """
    if instance.lower() == "core":
        return (
            "WorkerNode1  RUNNING  health=OK  uptime=14d 3h 22m\n"
            "WorkerNode2  RUNNING  health=OK  uptime=14d 3h 21m"
        )
    else:  # app
        return (
            "WorkerNode1  RUNNING   health=OK  uptime=14d 3h 22m\n"
            "WorkerNode2  SHUTDOWN  health=N/A"
        )


# ── 3. Deployment Status ──────────────────────────────────────────────────────

is_restarted = False

@tool
def get_deployment_status(instance: str) -> str:
    """
    Return the current state of all deployments (applications) across the
    worker nodes of the given WebLogic instance.

    Args:
        instance (str): "core" | "app"

    Expected return format:
        One line per deployment: <AppName>  <State>  <Target(s)>

    Example output (core instance):
        hermes_core        ACTIVE  WorkerNode1, WorkerNode2
        hermes_services    ACTIVE  WorkerNode1, WorkerNode2
        hermes_routs       ACTIVE  WorkerNode1, WorkerNode2
        hermes_dispatcher  ACTIVE  WorkerNode1, WorkerNode2

    Example output (app instance):
        hermes_app  ACTIVE  WorkerNode1

    States to watch for: ACTIVE | PREPARED | UNPREPARED | NEW | RETIRED |
                         FAILED | STATE_CHANGE_PENDING
    The agent will flag any deployment not in ACTIVE state.
    """
    if instance.lower() == "core":
        return (
            "hermes_core        ACTIVE  WorkerNode1, WorkerNode2\n"
            "hermes_services    ACTIVE  WorkerNode1, WorkerNode2\n"
            "hermes_routs       ACTIVE  WorkerNode1, WorkerNode2\n"
            "hermes_dispatcher  ACTIVE  WorkerNode1, WorkerNode2"
        )
    else:  # app
        if is_restarted:
            return "hermes_app  ACTIVE  WorkerNode1"
        else:
            return "hermes_app  FAILED  WorkerNode1"


# ── 4. Get Node Logs ──────────────────────────────────────────────────────────

@tool
def get_node_logs(instance: str, node_name: str, lines: int = 100) -> str:
    """
    Retrieve the tail of the server log for a specific worker node.

    Args:
        instance  (str): "core" | "app"
        node_name (str): Name of the node whose logs you want.
                         Core instance nodes : "WorkerNode1" | "WorkerNode2"
                         App  instance nodes : "WorkerNode1" | "WorkerNode2"
        lines     (int): How many tail lines to retrieve (default: 100).

    Expected return format:
        Raw log content -- timestamped WebLogic log entries.

    Example output:
        ####<Apr 25, 2026 10:01:10,001 AM CEST> <Info> <WebLogicServer> <host> <WorkerNode1> ... <BEA-000360> <Server started in RUNNING mode>
        ####<Apr 25, 2026 10:01:15,210 AM CEST> <Info> <WorkManager> <host> <WorkerNode1> ... <BEA-002900> <Initializing self-tuning thread pool>

    The agent will scan for WARNING / ERROR / CRITICAL / ALERT severity entries
    and summarise them.
    """
    host = "hermes-core-host" if instance.lower() == "core" else "hermes-app-host"
    return (
        f"####<Apr 25, 2026 10:01:10,001 AM CEST> <Info> <WebLogicServer> <{host}> <{node_name}> <main> <<WLS Kernel>> <> <> <1745571670001> <BEA-000360> <Server started in RUNNING mode>\n"
        f"####<Apr 25, 2026 10:01:15,210 AM CEST> <Info> <WorkManager> <{host}> <{node_name}> <main> <<WLS Kernel>> <> <> <1745571675210> <BEA-002900> <Initializing self-tuning thread pool>\n"
        f"####<Apr 25, 2026 10:01:20,455 AM CEST> <Info> <HTTP> <{host}> <{node_name}> <main> <<anonymous>> <> <> <1745571680455> <BEA-101162> <User weblogic connected from 192.168.1.10>\n"
        f"####<Apr 25, 2026 10:01:25,780 AM CEST> <Info> <JDBC> <{host}> <{node_name}> <main> <<WLS Kernel>> <> <> <1745571685780> <BEA-001128> <Connection pool 'oracleDS' has been created>\n"
        f"####<Apr 25, 2026 10:01:30,999 AM CEST> <Info> <EJB> <{host}> <{node_name}> <main> <<WLS Kernel>> <> <> <1745571690999> <BEA-010065> <Message-driven EJB: HermesMessageBean is successfully deployed>\n"
        f"####<Apr 25, 2026 10:01:35,123 AM CEST> <Info> <Management> <{host}> <{node_name}> <main> <<WLS Kernel>> <> <> <1745571695123> <BEA-141281> <No domain-level configuration changes detected>\n"
        f"####<Apr 25, 2026 10:01:40,456 AM CEST> <Info> <HTTP> <{host}> <{node_name}> <main> <<anonymous>> <> <> <1745571700456> <BEA-101163> <Request processed successfully -- GET /hermes/api/health 200 OK (12ms)>"
    )


# ── 5. Restart a Node ─────────────────────────────────────────────────────────

@tool
def restart_node(instance: str, node_name: str) -> str:
    """
    Trigger a graceful restart of a specific worker node and return the
    resulting status message.

    Args:
        instance  (str): "core" | "app"
        node_name (str): Node to restart -- "WorkerNode1" | "WorkerNode2"

    IMPORTANT:
        - Restarting a node in the CORE instance will briefly take 2 of the
          4 deployments offline if both nodes are targeted simultaneously.
          Restart one node at a time.
        - In the APP instance, only WorkerNode1 should ever be restarted
          (WorkerNode2 is expected to remain SHUTDOWN). Restarting WorkerNode2
          would be an anomaly -- the agent will warn before proceeding.

    Expected return format:
        A status message confirming the restart sequence outcome.

    Example output (success):
        Shutdown of WorkerNode1 initiated ... SHUTDOWN
        Start    of WorkerNode1 initiated ... RUNNING
        Restart completed successfully. WorkerNode1 is now RUNNING.
    """
    global is_restarted
    is_restarted = True
    return (
        f"Shutdown of {node_name} initiated ... done\n"
        f"{node_name} state: SHUTTING_DOWN\n"
        f"{node_name} state: SHUTDOWN\n"
        f"Start of {node_name} initiated ... done\n"
        f"{node_name} state: STARTING\n"
        f"{node_name} state: RUNNING\n"
        f"Restart completed successfully. {node_name} is now RUNNING."
    )


# ── 6. Clean Node Logs ────────────────────────────────────────────────────────

@tool
def clean_node_logs(instance: str, node_name: str) -> str:
    """
    Clear (rotate/truncate) the server logs on a specific worker node to free
    up disk space or start a clean logging session.

    Args:
        instance  (str): "core" | "app"
        node_name (str): Node whose logs to clean -- "WorkerNode1" | "WorkerNode2"

    IMPORTANT:
        - This operation is IRREVERSIBLE. Ensure you have archived any logs
          you need before proceeding.
        - The server does NOT need to be stopped for log rotation, but a
          restart of the server causes WebLogic to open a fresh log file
          automatically.

    Expected return format:
        Confirmation lines showing which files were removed and the disk freed.

    Example output (success):
        Removed: $DOMAIN_HOME/servers/WorkerNode1/logs/WorkerNode1.log      (138 MB)
        Removed: $DOMAIN_HOME/servers/WorkerNode1/logs/WorkerNode1.log0001  (200 MB)
        Removed: $DOMAIN_HOME/servers/WorkerNode1/logs/access.log           (54 MB)
        Done. Disk freed: 392 MB. Log directory is now empty.
    """
    return (
        f"Removed: $DOMAIN_HOME/servers/{node_name}/logs/{node_name}.log      (138 MB)\n"
        f"Removed: $DOMAIN_HOME/servers/{node_name}/logs/{node_name}.log0001  (200 MB)\n"
        f"Removed: $DOMAIN_HOME/servers/{node_name}/logs/{node_name}.log0002  (200 MB)\n"
        f"Removed: $DOMAIN_HOME/servers/{node_name}/logs/access.log           (54 MB)\n"
        f"Done. Disk freed: 592 MB. Log directory is now empty."
    )


# ── Tool Registry ─────────────────────────────────────────────────────────────

tools = [
    get_admin_server_status,
    get_node_status,
    get_deployment_status,
    get_node_logs,
    restart_node,
    clean_node_logs,
]


# ==============================================================================
# LLM & AGENT
# ==============================================================================

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-120b:free",
    streaming=True,
    temperature=0
)

SYSTEM_PROMPT = """
You are an expert DevOps assistant managing a Hermes development environment.
You have two WebLogic instances under your supervision:

HERMES CORE  (instance = "core")
  Admin  : AdminServer          -> should always be RUNNING
  Workers: WorkerNode1          -> should always be RUNNING
           WorkerNode2          -> should always be RUNNING
  Deployments (on both workers):
    - hermes_core
    - hermes_services
    - hermes_routs
    - hermes_dispatcher

HERMES APP   (instance = "app")
  Admin  : AdminServer          -> should always be RUNNING
  Workers: WorkerNode1          -> should always be RUNNING
           WorkerNode2          -> should always be SHUTDOWN (expected)
  Deployments:
    - hermes_app  (on WorkerNode1 ONLY)

You also manage one Oracle Database instance (status checks are handled outside
your tools for now -- flag any DB-related issues to the operator manually).

BEHAVIOUR RULES:
2. When you detect an anomaly (e.g. a node in unexpected state, a deployment
   not ACTIVE, disk full warnings in logs), report it clearly and suggest a
   remediation step.
6. In the app instance, if WorkerNode2 is found RUNNING, flag it as an anomaly
   and ask the operator whether to shut it down.
7. Be concise in your summaries -- use checkmark emoji for healthy, warning emoji
   for warnings, and X emoji for errors/failures.
8. If a deployment is not in a RUNNING status, then suggest to the user to restart 
   the node its running on, this usually solves the issue 
""".strip()


@cl.on_chat_start
async def start():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def main(message: cl.Message):
    agent_executor = cl.user_session.get("agent_executor")
    chat_history = cl.user_session.get("chat_history")

    res = await agent_executor.ainvoke(
        {"input": message.content, "chat_history": chat_history},
        config={"callbacks": [cl.LangchainCallbackHandler()]}
    )

    # Append the new turn to history so future invocations have full context
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=res["output"]))
    cl.user_session.set("chat_history", chat_history)

    await cl.Message(content=res["output"]).send()