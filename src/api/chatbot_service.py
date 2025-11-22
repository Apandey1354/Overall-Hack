"""
Chatbot Service for Shu Todoroki
Provides project-related assistance with safety and ethics checks.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        print(f"Warning: OpenAI client initialization failed: {exc}")
        openai_client = None


# Project-related keywords and topics
PROJECT_KEYWORDS = [
    "track dna", "track dna profiler", "championship fate matrix", "fate matrix",
    "track performance", "driver performance", "racing", "race", "circuit", "venue",
    "telemetry", "lap time", "driver", "championship", "simulation", "monte carlo",
    "track coach", "coach lab", "scenario lab", "driver embedding", "track clustering",
    "butterfly effect", "track difficulty", "sector", "weather", "strategy",
    "barber", "cota", "indianapolis", "vir", "raceway", "grcup", "gr cup",
    "track type", "technical track", "speed track", "balanced track",
    "transfer learning", "model", "prediction", "forecast", "analysis"
]

# Illegal/unethical topics to filter
ILLEGAL_KEYWORDS = [
    "hack", "illegal", "cheat", "exploit", "unauthorized", "breach", "steal",
    "violence", "harm", "dangerous", "weapon", "drug", "illegal activity"
]


def check_safety_and_relevance(question: str) -> Tuple[bool, Optional[str]]:
    """
    First layer: Check if the question is legal, ethical, and project-related.
    
    Returns:
        (is_safe_and_relevant, error_message)
        - If safe and relevant: (True, None)
        - If unsafe or irrelevant: (False, error_message)
    """
    question_lower = question.lower()
    
    # Check for illegal/unethical content
    for keyword in ILLEGAL_KEYWORDS:
        if keyword in question_lower:
            return False, "I cannot answer questions about illegal or harmful activities. Please ask about the Track DNA Profiler or Championship Fate Matrix project instead."
    
    # Check if question is project-related
    is_project_related = any(keyword in question_lower for keyword in PROJECT_KEYWORDS)
    
    # Also allow general greetings and basic questions
    greetings = ["hello", "hi", "hey", "greetings", "help", "what can you do", "who are you"]
    is_greeting = any(greeting in question_lower for greeting in greetings)
    
    if not (is_project_related or is_greeting):
        return False, "I can only answer questions about the Track DNA Profiler and Championship Fate Matrix project. Please ask about track analysis, driver performance, championship simulation, or related concepts."
    
    return True, None


def get_project_context() -> str:
    """
    Get comprehensive project context for the chatbot.
    """
    return """You are Shu Todoroki, a helpful AI assistant for the Track DNA Profiler & Championship Fate Matrix project.

PROJECT OVERVIEW:
This project combines two powerful AI systems:

1. Track DNA Profiler: A venue-specific AI that learns each track's unique characteristics and predicts driver performance across different circuits. It analyzes:
   - Track difficulty rankings
   - Driver performance by track type (technical, high-speed, balanced)
   - Track-specific patterns and characteristics
   - Sector-by-sector analysis
   - Weather impact on performance

2. Championship Fate Matrix: A season-long simulation engine that:
   - Predicts championship outcomes using Monte Carlo methods
   - Identifies critical pivot points and butterfly effects
   - Allows "what-if" scenario testing
   - Tracks driver standings across multiple races

KEY FEATURES:
- Track Coach Lab: Provides driver-specific advice based on track characteristics and driver skill profiles
- Scenario Lab: Lets users modify race results and see championship impact
- Driver Embeddings: 8-dimensional skill vectors representing driver capabilities
- Track Clustering: Groups tracks by similarity
- Transfer Learning: Predicts performance on new tracks using learned patterns

AVAILABLE TRACKS:
- Barber Motorsports Park
- Circuit of the Americas (COTA)
- Indianapolis Motor Speedway
- Virginia International Raceway (VIR)

DATA SOURCES:
- Race results and standings
- Telemetry data (lap times, sector times)
- Weather conditions
- Driver performance metrics
- Best lap analysis

Your role is to help users understand:
- How the Track DNA Profiler works
- How to interpret championship simulations
- Concepts like driver embeddings, track difficulty, and butterfly effects
- How to use the Coach Lab and Scenario Lab features
- General questions about the project's purpose and functionality

Be friendly, clear, and educational. If asked about something outside the project scope, politely redirect to project-related topics."""


def generate_chatbot_response(question: str, conversation_history: Optional[list] = None) -> str:
    """
    Generate a response from Shu Todoroki chatbot.
    
    Args:
        question: User's question
        conversation_history: Optional list of previous messages in format [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        Response string from the chatbot
    """
    # First layer: Safety and relevance check
    is_safe, error_msg = check_safety_and_relevance(question)
    if not is_safe:
        return error_msg
    
    if not openai_client:
        # Fallback response without OpenAI
        if any(greeting in question.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm Shu Todoroki, your assistant for the Track DNA Profiler & Championship Fate Matrix project. I can help explain concepts, answer questions about track analysis, driver performance, championship simulation, and how to use the various features. What would you like to know?"
        
        return "I'm Shu Todoroki, your assistant for this project. To enable full AI-powered responses, please configure OPENAI_API_KEY in your .env file. For now, I can tell you this is a racing analytics project that analyzes track characteristics and simulates championship outcomes."
    
    # Build conversation context
    messages = [
        {"role": "system", "content": get_project_context()}
    ]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history[-10:])  # Keep last 10 messages for context
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        
        return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
    
    except Exception as exc:
        return f"I encountered an error: {str(exc)}. Please try again or ask about the Track DNA Profiler or Championship Fate Matrix project."

