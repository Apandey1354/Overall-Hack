"""
AI Recommendation Service
Generates AI-powered recommendations that reference Coach Lab insights and Scenario Lab changes.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Any

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


def clean_markdown(text: str) -> str:
    """
    Remove markdown formatting from text to ensure clean, natural output.
    """
    if not text:
        return text
    
    # Remove bold/italic markdown (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove code blocks and inline code
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove headers (# Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


def generate_coach_recommendations(
    track_id: str,
    coach_advice: Dict[str, Any],
    sector_recommendations: List[Dict],
    weather_strategies: Dict[str, Any],
    driver_number: int,
    driver_name: str,
    driver_embedding: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate AI recommendations based on Coach Lab insights.
    """
    if not openai_client:
        return "AI recommendations unavailable (missing OPENAI_API_KEY). Configure OPENAI_API_KEY in your .env file to enable AI-powered insights."

    # Format sector recommendations
    sector_text = "\n".join([
        f"- {s.get('sector', 'Unknown')}: {s.get('focus', 'N/A')} (variance: {s.get('variance', 0):.3f})"
        for s in sector_recommendations[:5]
    ])

    # Format weather strategies
    weather_text = "\n".join([
        f"- {cond}: {str(strategy.get('notes', []))}"
        for cond, strategy in weather_strategies.items()
    ])

    # Format driver embedding data if available
    embedding_text = ""
    if driver_embedding:
        skill_vector = driver_embedding.get('skill_vector', [])
        embedding_text = f"""
DRIVER SKILL PROFILE (from Driver Embeddings):
Technical Proficiency: {driver_embedding.get('technical_proficiency', 0) * 100:.1f}%
High-Speed Proficiency: {driver_embedding.get('high_speed_proficiency', 0) * 100:.1f}%
Consistency Score: {driver_embedding.get('consistency_score', 0) * 100:.1f}%
Weather Adaptability: {driver_embedding.get('weather_adaptability', 0) * 100:.1f}%
Best Track Type: {driver_embedding.get('best_track_type', 'N/A')}
Track-Specific Strengths: {driver_embedding.get('strengths', 'N/A')}

8-Dimensional Skill Vector Breakdown:
- Technical: {skill_vector[0] * 100:.1f}% (skill at technical tracks)
- High-Speed: {skill_vector[1] * 100:.1f}% (skill at speed-focused tracks)
- Consistency: {skill_vector[2] * 100:.1f}% (lap time and position consistency)
- Weather Adaptability: {skill_vector[3] * 100:.1f}% (performance across conditions)
- Tech-Track Performance: {skill_vector[4] * 100:.1f}% (specific to technical tracks)
- Speed-Track Performance: {skill_vector[5] * 100:.1f}% (specific to speed tracks)
- Balanced-Track Performance: {skill_vector[6] * 100:.1f}% (specific to balanced tracks)
- Finish Rate: {skill_vector[7] * 100:.1f}% (race completion rate)
"""

    prompt = f"""You are a professional racing coach providing personalized recommendations to a driver.

Track: {track_id}
Driver: #{driver_number} - {driver_name}
{embedding_text}
COACH LAB INSIGHTS:
Sector Recommendations:
{sector_text}

Weather Strategies:
{weather_text}

Driver-Specific Advice:
- Strengths: {', '.join(coach_advice.get('strengths', []))}
- Weaknesses: {', '.join(coach_advice.get('weaknesses', []))}
- Focus Areas: {chr(10).join('- ' + item for item in coach_advice.get('focus', []))}

Provide 4-5 concise, actionable recommendations that:
1. Reference the driver's specific skill profile from their embeddings (technical proficiency, consistency, weather adaptability, etc.)
2. Connect their skill strengths/weaknesses to the track's characteristics
3. Reference specific sector recommendations from the Coach Lab that match their skill profile
4. Incorporate weather strategy insights, especially considering their weather adaptability score
5. Address areas where their skill profile shows gaps (e.g., if technical proficiency is low but track is technical)
6. Use natural, coaching language tailored to this specific driver's profile

IMPORTANT: Write in plain text only. Use simple bullet points with dashes (-). Do NOT use any markdown formatting like asterisks, bold text, or special characters. Write naturally as if speaking to the driver directly. Each recommendation should be 1-2 sentences.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert racing coach providing clear, actionable advice. Always respond in plain text without any markdown formatting, bold text, or special characters. Write naturally and conversationally."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        if response.choices and response.choices[0].message.content:
            cleaned = clean_markdown(response.choices[0].message.content.strip())
            return cleaned
        return "AI response empty."
    except Exception as exc:
        return f"AI recommendation unavailable: {str(exc)}"


def generate_scenario_recommendations(
    scenario_name: str,
    scenario_description: str,
    adjustments: List[Dict],
    baseline_standings: List[Dict],
    scenario_standings: List[Dict],
    impact_summary: Optional[str] = None,
    target_driver_number: Optional[int] = None,
    driver_embeddings: Optional[List[Dict]] = None,
) -> str:
    """
    Generate AI recommendations based on Scenario Lab changes and their championship impact.
    """
    if not openai_client:
        return "AI recommendations unavailable (missing OPENAI_API_KEY). Configure OPENAI_API_KEY in your .env file to enable AI-powered insights."

    # Format adjustments
    adjustments_text = "\n".join([
        f"- Event {adj.get('event_order', '?')}: Driver #{adj.get('changes', [{}])[0].get('driver_number', '?')} "
        f"changed to position {adj.get('changes', [{}])[0].get('new_position', '?')}"
        for adj in adjustments[:5]
    ])

    # Format top 5 baseline vs scenario
    baseline_top5 = "\n".join([
        f"  {i+1}. Driver #{d.get('driver_number', '?')}: {d.get('season_points', 0):.1f} pts"
        for i, d in enumerate(baseline_standings[:5])
    ])
    
    scenario_top5 = "\n".join([
        f"  {i+1}. Driver #{d.get('driver_number', '?')}: {d.get('season_points', 0):.1f} pts"
        for i, d in enumerate(scenario_standings[:5])
    ])

    # Add driver skill comparison if target driver and embeddings are provided
    driver_skill_analysis = ""
    if target_driver_number and driver_embeddings:
        target_embedding = next((e for e in driver_embeddings if e.get('driver_number') == target_driver_number), None)
        if target_embedding:
            top5_drivers = [d.get('driver_number') for d in scenario_standings[:5]]
            top5_embeddings = [e for e in driver_embeddings if e.get('driver_number') in top5_drivers]
            
            skill_labels = ["Technical", "High-Speed", "Consistency", "Weather", "Tech-Track", "Speed-Track", "Balanced-Track", "Finish-Rate"]
            target_skills = target_embedding.get('skill_vector', [])
            
            # Compare target driver with top 5
            comparisons = []
            for skill_idx, skill_label in enumerate(skill_labels):
                target_val = target_skills[skill_idx] if skill_idx < len(target_skills) else 0
                top5_avg = sum(e.get('skill_vector', [])[skill_idx] if skill_idx < len(e.get('skill_vector', [])) else 0 
                              for e in top5_embeddings) / len(top5_embeddings) if top5_embeddings else 0
                diff = target_val - top5_avg
                if diff > 0.1:
                    comparisons.append(f"{skill_label}: +{(diff * 100):.1f}% vs top 5 average (STRENGTH)")
                elif diff < -0.1:
                    comparisons.append(f"{skill_label}: {(diff * 100):.1f}% vs top 5 average (FOCUS AREA)")
            
            if comparisons:
                driver_skill_analysis = f"""
TARGET DRIVER SKILL ANALYSIS (Driver #{target_driver_number}):
{chr(10).join(comparisons[:6])}

Strong Points: Focus on races that leverage the driver's strengths above.
Focus Areas: Work on improving skills where the driver is below top 5 average.
"""

    prompt = f"""You are a championship strategist analyzing "what-if" scenarios.

SCENARIO: {scenario_name}
Description: {scenario_description}

CHANGES MADE (from Scenario Lab):
{adjustments_text}

CHAMPIONSHIP IMPACT:
Baseline Standings (Top 5):
{baseline_top5}

Scenario Standings (Top 5):
{scenario_top5}

{driver_skill_analysis}

{impact_summary if impact_summary else ''}

Provide 3-4 strategic insights that:
1. Explain the key championship implications of these changes
2. Identify which drivers were most affected and why
3. Highlight critical moments or pivot points revealed by this scenario
4. Suggest strategic implications for future races
{f"5. Focus on the target driver's strong points compared to top 5 competitors and what areas need improvement" if target_driver_number else ""}

IMPORTANT: Write in plain text only. Use simple bullet points with dashes (-). Do NOT use any markdown formatting like asterisks, bold text, or special characters. Write naturally and analytically as if explaining to a team strategist. Each insight should be 1-2 sentences.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a championship strategist providing strategic insights about race scenarios. Always respond in plain text without any markdown formatting, bold text, or special characters. Write naturally and professionally."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        if response.choices and response.choices[0].message.content:
            cleaned = clean_markdown(response.choices[0].message.content.strip())
            return cleaned
        return "AI response empty."
    except Exception as exc:
        return f"AI recommendation unavailable: {str(exc)}"


def generate_combined_recommendations(
    coach_insights: Optional[Dict[str, Any]] = None,
    scenario_changes: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate AI recommendations that combine insights from both Coach Lab and Scenario Lab.
    """
    if not openai_client:
        return "AI recommendations unavailable (missing OPENAI_API_KEY). Configure OPENAI_API_KEY in your .env file to enable AI-powered insights."

    coach_text = ""
    if coach_insights:
        coach_text = f"""
COACH LAB INSIGHTS:
Track: {coach_insights.get('track_id', 'N/A')}
Driver: #{coach_insights.get('driver_number', '?')}
Key Focus Areas: {', '.join(coach_insights.get('focus', []))}
Strengths: {', '.join(coach_insights.get('strengths', []))}
"""

    scenario_text = ""
    if scenario_changes:
        scenario_text = f"""
SCENARIO LAB CHANGES:
Scenario: {scenario_changes.get('name', 'N/A')}
Key Changes: {scenario_changes.get('summary', 'N/A')}
Championship Impact: {scenario_changes.get('impact', 'N/A')}
"""

    if not coach_text and not scenario_text:
        return "No insights available. Use Coach Lab or Scenario Lab to generate recommendations."

    prompt = f"""You are a racing strategist providing comprehensive recommendations that connect coaching insights with championship strategy.

{coach_text}
{scenario_text}

Provide 4-5 strategic recommendations that:
1. Connect coaching insights to championship implications (if both available)
2. Reference specific changes from Scenario Lab and their strategic meaning
3. Incorporate Coach Lab sector/weather advice into race strategy
4. Provide actionable next steps based on these insights

IMPORTANT: Write in plain text only. Use simple bullet points with dashes (-). Do NOT use any markdown formatting like asterisks, bold text, or special characters. Write naturally and strategically as if briefing a racing team. Each recommendation should be 1-2 sentences.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a racing strategist connecting coaching insights with championship strategy. Always respond in plain text without any markdown formatting, bold text, or special characters. Write naturally and strategically."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600,
        )
        if response.choices and response.choices[0].message.content:
            cleaned = clean_markdown(response.choices[0].message.content.strip())
            return cleaned
        return "AI response empty."
    except Exception as exc:
        return f"AI recommendation unavailable: {str(exc)}"

