"""
Example: Phase 5 Butterfly Effect & Scenario Analysis
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from src.championship import ChampionshipSimulator, ButterflyEffectAnalyzer, ScenarioGenerator

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: OpenAI client initialization failed: {exc}")
        openai_client = None


def _summarize_with_llm(section_text: str) -> str:
    if not openai_client:
        return "LLM summary unavailable (missing OPENAI_API_KEY or openai package)."
    prompt = (
        "You are a championship strategist. Summarize the following butterfly-effect analysis "
        "into 4 concise bullet points that highlight pivotal races, key drivers affected, and the "
        "overall storyline in plain English:\n\n"
        f"{section_text}"
    )
    try:
        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
            max_output_tokens=300,
        )
        if response.output and response.output[0].content:
            return response.output[0].content[0].text.strip()
        return "LLM summary unavailable (empty response)."
    except Exception as exc:  # pragma: no cover
        return f"LLM summary unavailable ({exc})"


def _format_top_standings(df, title: str, limit: int = 5) -> List[str]:
    lines = [title]
    for _, row in df.head(limit).iterrows():
        lines.append(
            f"  P{int(row['rank'])}: #{int(row['driver_number']):02d} "
            f"{row['driver_name']} - {row['season_points']:.1f} pts"
        )
    return lines


def _build_comparison(baseline: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    comparison = baseline.merge(
        scenario,
        on="driver_number",
        suffixes=("_base", "_scenario"),
    )
    comparison["points_delta"] = comparison["season_points_scenario"] - comparison["season_points_base"]
    comparison["rank_delta"] = comparison["rank_scenario"] - comparison["rank_base"]
    return comparison.sort_values("rank_scenario")


def _append_scenario_section(
    lines: List[str],
    title: str,
    description: str,
    adjustments: List[dict],
    scenario_result: Dict,
    baseline_standings: pd.DataFrame,
) -> None:
    scenario_standings = scenario_result["final_standings"]
    lines.append(f"\n=== Scenario: {title} ===")
    lines.append(f"Description: {description}")
    lines.append("Adjustments applied:")
    for adjustment in adjustments:
        lines.append(f"  Event {adjustment['event_order']}: {adjustment['changes']}")
    lines.extend(_format_top_standings(scenario_standings, "\nScenario Final Standings (Top 5):", limit=5))

    lines.append("\nTimeline Leaderboard:")
    for entry in scenario_result["timeline"]:
        lines.append(
            f"  After Event {entry['event_order']} ({entry['track_id']}): "
            f"Leader #{int(entry['leader_driver_number']):02d} at {entry['leader_points']:.1f} pts"
        )

    comparison = _build_comparison(baseline_standings, scenario_standings)
    lines.append("\nBaseline vs Scenario Comparison (Top 10 by scenario rank):")
    for _, row in comparison.head(10).iterrows():
        lines.append(
            f"#{int(row['driver_number']):02d} {row['driver_name_base']} | "
            f"Base P{int(row['rank_base'])} ({row['season_points_base']:.1f} pts) "
            f"-> Scenario P{int(row['rank_scenario'])} ({row['season_points_scenario']:.1f} pts) | "
            f"ΔPts {row['points_delta']:+.1f}, ΔRank {row['rank_delta']:+}"
        )


def main() -> None:
    simulator = ChampionshipSimulator()
    analyzer = ButterflyEffectAnalyzer(simulator, include_advanced=True, random_seed=21)
    baseline = analyzer.run_baseline()
    baseline_standings = baseline["final_standings"].copy()
    baseline_champion = baseline_standings.sort_values("season_points", ascending=False).iloc[0]

    impact_reports = analyzer.rank_event_impacts(max_events=5)

    scenario_generator = ScenarioGenerator(analyzer)

    lines: List[str] = []
    lines.append("=== Baseline Championship (Top 5) ===")
    lines.extend(_format_top_standings(baseline_standings, "", limit=5))
    lines.append(
        f"\nBaseline Champion: #{int(baseline_champion['driver_number']):02d} {baseline_champion['driver_name']} "
        f"({baseline_champion['season_points']:.1f} pts)"
    )

    lines.append("\n=== Butterfly Effect: Top Event Impacts ===")
    for report in impact_reports:
        lines.append(
            f"Event {report.event_order} ({report.track_id}) | Score={report.impact_score:.2f} | "
            f"Champion change: {report.champion_before} -> "
            f"{report.champion_after if report.champion_changed else report.champion_before}"
        )
        for mover in report.key_movers:
            lines.append(
                f"    ΔPts {mover['points_delta_abs']:+.1f} | ΔRank {mover['rank_delta']:+} "
                f"for #{int(mover['driver_number']):02d} {mover['driver_name']}"
            )

    scenario_adjustments_driver13 = [
        {
            "event_order": impact_reports[0].event_order if impact_reports else 1,
            "changes": [{"driver_number": 13, "new_position": 1}],
        },
        {
            "event_order": impact_reports[1].event_order if len(impact_reports) > 1 else 2,
            "changes": [{"driver_number": 72, "position_delta": -1}],
        },
    ]
    scenario_driver13 = scenario_generator.run_scenario(
        name="Driver 13 comeback",
        description="Force driver #13 to win one high-impact race and promote driver #72 in another.",
        adjustments=scenario_adjustments_driver13,
    )
    _append_scenario_section(
        lines,
        title="Driver 13 comeback",
        description=scenario_driver13["description"],
        adjustments=scenario_adjustments_driver13,
        scenario_result=scenario_driver13,
        baseline_standings=baseline_standings,
    )

    driver16_events = [report.event_order for report in impact_reports[:3]] or [1, 2, 3]
    scenario_adjustments_driver16 = [
        {
            "event_order": event_order,
            "changes": [{"driver_number": 16, "new_position": 1}],
        }
        for event_order in driver16_events
    ]
    scenario_driver16 = scenario_generator.run_scenario(
        name="Driver 16 title push",
        description="Grant driver #16 victories in the three highest-impact events to evaluate title potential.",
        adjustments=scenario_adjustments_driver16,
    )
    _append_scenario_section(
        lines,
        title="Driver 16 title push",
        description=scenario_driver16["description"],
        adjustments=scenario_adjustments_driver16,
        scenario_result=scenario_driver16,
        baseline_standings=baseline_standings,
    )

    llm_section = "\n".join(lines[-40:])  # summarize recent context
    lines.append("\n=== AI Narrative Summary ===")
    lines.append(_summarize_with_llm(llm_section))

    output_path = Path("butterfly_effect_results.txt")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Butterfly effect analysis written to {output_path.resolve()}")


if __name__ == "__main__":
    main()


