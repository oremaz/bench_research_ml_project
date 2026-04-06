"""
LLM-based intent classification for NutriCoach.
Replaces keyword-based routing with structured output from the LLM.
"""

from typing import Literal
from pydantic import BaseModel, Field


class IntentSchema(BaseModel):
    """Structured intent classification result."""
    intent: Literal[
        "general_chat",
        "calculate_targets",
        "meal_planning",
        "log_daily",
        "analyze_progress",
        "profile_update",
    ] = Field(description="The user's primary intent")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the classification")


INTENT_CLASSIFICATION_PROMPT = """Classify the user's intent into exactly one of these categories:

- general_chat: Casual conversation, greetings, general nutrition questions, or anything that doesn't fit other categories
- calculate_targets: Requesting nutrition target calculations (calories, macros, BMR, TDEE)
- meal_planning: Asking for meal plans, meal suggestions, or "what should I eat" type questions
- log_daily: Reporting what they ate, logging meals, tracking weight, recording daily compliance
- analyze_progress: Reviewing trends, weekly progress, asking about how they're doing over time
- profile_update: Changing profile information, updating goals, dietary preferences, or restrictions

User message: "{user_message}"

Classify this message."""


def classify_intent(llm, user_message: str, has_profile: bool = True) -> IntentSchema:
    """
    Classify user intent using LLM structured output.

    Args:
        llm: A LangChain LLM instance (e.g., ChatGoogleGenerativeAI)
        user_message: The user's message to classify
        has_profile: Whether the user has a complete profile

    Returns:
        IntentSchema with the classified intent
    """
    # If no profile, always route to profile_update
    if not has_profile:
        return IntentSchema(
            intent="profile_update",
            confidence=1.0,
            reasoning="User profile is incomplete, needs setup first",
        )

    try:
        classifier = llm.with_structured_output(IntentSchema)
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_message=user_message)
        result = classifier.invoke(prompt)
        return result
    except Exception:
        # Fallback to general_chat on any LLM error
        return IntentSchema(
            intent="general_chat",
            confidence=0.5,
            reasoning="Fallback due to classification error",
        )
