"""
Trinity Cognitive Architecture — Classification Schema v1

The Mind-Body contract: Phi-3.5-mini outputs this schema via grammar-constrained
decoding. The Body reads x_jarvis_routing metadata to decide how to render/execute.
"""

import os
from typing import Dict, Any, Optional

SCHEMA_VERSION = 1

CLASSIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer"},
        "intent": {
            "type": "string",
            "enum": [
                "answer",
                "action",
                "multi_step_action",
                "vision_needed",
                "clarify",
                "conversation",
            ],
        },
        "domain": {
            "type": "string",
            "enum": [
                "math", "code", "reasoning", "creative", "general",
                "system", "voice_unlock", "vision", "agentic", "translation",
                "conversation", "surveillance", "workspace",
            ],
        },
        "complexity": {
            "type": "string",
            "enum": ["trivial", "simple", "moderate", "complex", "expert"],
        },
        "requires_vision": {"type": "boolean"},
        "requires_action": {"type": "boolean"},
        "escalate_to_claude": {"type": "boolean"},
        "confidence": {"type": "number"},
        "suggested_actions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "schema_version", "intent", "domain", "complexity",
        "confidence", "requires_vision", "requires_action",
        "escalate_to_claude",
    ],
}

# Domain -> TaskType mapping (feeds into GCPModelSwapCoordinator)
DOMAIN_TO_TASK_TYPE: Dict[str, str] = {
    "math": "math_complex",
    "code": "code_complex",
    "reasoning": "reason_complex",
    "creative": "creative_write",
    "general": "general_chat",
    "system": "voice_command",
    "vision": "multimodal",
    "agentic": "reason_complex",
    "translation": "translate",
    "conversation": "greeting",
    "surveillance": "voice_command",
    "workspace": "voice_command",
    "voice_unlock": "voice_command",
}

# Domains where Phi can both classify AND respond (no specialist needed)
# v242.1: Narrowed to conversation only — system commands need the action
# executor, not a text response from the classifier model.
PHI_SELF_SERVE_DOMAINS = frozenset({"conversation"})

# Minimum confidence to trust classification (below this -> escalate)
MIN_CONFIDENCE_THRESHOLD = float(
    os.environ.get("JARVIS_PHI_MIN_CONFIDENCE", "0.5")
)


def build_classifier_system_prompt(action_registry: Optional[Dict] = None) -> str:
    """Build the Phi classifier system prompt with optional action registry."""

    actions_section = ""
    if action_registry:
        actions_list = ", ".join(sorted(action_registry.keys()))
        actions_section = f"\n\nAvailable actions the Body can execute: {actions_list}"

    return f"""You are a query classifier for the JARVIS AI assistant. Your ONLY job is to classify the user's query into a structured JSON object. Do NOT answer the query — only classify it.

Output JSON with these fields:
- schema_version: always {SCHEMA_VERSION}
- intent: one of [answer, action, multi_step_action, vision_needed, clarify, conversation]
  - "answer": user is asking a question that needs a text response
  - "action": user wants a single system action (open app, lock screen, volume change)
  - "multi_step_action": user wants a sequence of actions (open browser, navigate, click)
  - "vision_needed": user is asking about something visual (screen content, image analysis)
  - "clarify": query is ambiguous, ask user to clarify
  - "conversation": greeting, small talk, or casual interaction
- domain: one of [math, code, reasoning, creative, general, system, voice_unlock, vision, agentic, translation, conversation, surveillance, workspace]
  - CRITICAL: "voice_unlock" is for ANY command to unlock/open/wake the screen, computer, Mac, or laptop via voice. NEVER classify unlock commands as workspace.
- complexity: one of [trivial, simple, moderate, complex, expert]
- requires_vision: true if the query needs to see the screen or an image
- requires_action: true if the query needs the Body to execute a system action
- escalate_to_claude: true if this query is too complex for a 7B local model (multi-step agentic planning, computer use, safety-critical decisions)
- confidence: 0.0 to 1.0, how confident you are in this classification
- suggested_actions: optional list of specific action names the Body should execute. Use for action/multi_step_action intents.
  System actions: ["lock_screen"], ["unlock_screen"], ["open_browser"], ["volume_up"], ["volume_down"]
  Workspace actions: ["fetch_unread_emails"], ["check_calendar_events"], ["send_email"], ["draft_email_reply"], ["create_calendar_event"], ["daily_briefing"], ["workspace_summary"], ["get_contacts"], ["create_document"]{actions_section}

Examples:
- "what's today" -> intent=answer, domain=general, complexity=trivial, confidence=0.95
- "lock my screen" -> intent=action, domain=system, complexity=trivial, confidence=0.99, suggested_actions=["lock_screen"]
- "unlock my screen" -> intent=action, domain=voice_unlock, complexity=trivial, confidence=0.99, requires_action=true, suggested_actions=["unlock_screen"]
- "unlock the mac" -> intent=action, domain=voice_unlock, complexity=trivial, confidence=0.99, requires_action=true, suggested_actions=["unlock_screen"]
- "unlock my computer" -> intent=action, domain=voice_unlock, complexity=trivial, confidence=0.99, requires_action=true, suggested_actions=["unlock_screen"]
- "unlock my laptop" -> intent=action, domain=voice_unlock, complexity=trivial, confidence=0.99, requires_action=true, suggested_actions=["unlock_screen"]
- "wake my screen" -> intent=action, domain=voice_unlock, complexity=trivial, confidence=0.98, requires_action=true, suggested_actions=["unlock_screen"]
- "open my screen" -> intent=action, domain=voice_unlock, complexity=trivial, confidence=0.97, requires_action=true, suggested_actions=["unlock_screen"]
- "what's on my screen" -> intent=vision_needed, domain=vision, requires_vision=true, confidence=0.92
- "what's the derivative of x squared" -> intent=answer, domain=math, complexity=moderate, confidence=0.93
- "open Safari and go to GitHub" -> intent=multi_step_action, domain=agentic, escalate_to_claude=true, confidence=0.88
- "hello" -> intent=conversation, domain=conversation, complexity=trivial, confidence=0.99
- "watch all Chrome windows for changes" -> intent=action, domain=surveillance, requires_action=true, confidence=0.91
- "check my email" -> intent=action, domain=workspace, complexity=simple, confidence=0.93, suggested_actions=["fetch_unread_emails"]
- "what's on my calendar today" -> intent=action, domain=workspace, complexity=simple, confidence=0.91, suggested_actions=["check_calendar_events"]
- "send an email to John" -> intent=action, domain=workspace, complexity=moderate, confidence=0.90, suggested_actions=["send_email"]
- "draft a reply to that email" -> intent=action, domain=workspace, complexity=moderate, confidence=0.88, suggested_actions=["draft_email_reply"]
- "schedule a meeting for tomorrow" -> intent=action, domain=workspace, complexity=moderate, confidence=0.89, suggested_actions=["create_calendar_event"]
- "give me my morning briefing" -> intent=action, domain=workspace, complexity=simple, confidence=0.92, suggested_actions=["daily_briefing"]"""
