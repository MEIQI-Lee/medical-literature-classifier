"""Public integration API for the web/backend owner."""

from agent.agent import ClassificationAgent
from agent.factory import create_agent, create_demo_agent, create_medbert_agent
from agent.schemas import AgentConfig, AgentResult
from agent.service import AcademicClassificationAgent, get_agent

__all__ = [
    "ClassificationAgent",
    "AgentConfig",
    "AgentResult",
    "create_agent",
    "create_demo_agent",
    "create_medbert_agent",
    "AcademicClassificationAgent",
    "get_agent",
]
