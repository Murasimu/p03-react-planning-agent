"""ReAct planning agent: observe -> think -> act -> reflect, with hard
iteration limits, self-critique on failure, and graceful degradation.

The agent reasons in explicit steps (Thought / Action / Observation),
never loops forever (max_steps), critiques its own failed actions
(Reflection), and when it cannot finish it returns a best-effort partial
answer instead of spinning.
"""

from .agent import ReActAgent, Step, StepType
from .parser import parse_step, ParseError

__all__ = ["ReActAgent", "Step", "StepType", "parse_step", "ParseError"]
__version__ = "1.0.0"
