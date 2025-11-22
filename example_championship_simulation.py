Mechanical Karma Detector
Error: Failed to load components

Make sure telemetry data is available and processed."""
Example: Championship Simulation Engine

Runs a single-season simulation plus a Monte Carlo sweep, then writes the outputs
to championship_simulation_results.txt for reference.
"""

from __future__ import annotations

from pathlib import Path

from src.championship import ChampionshipSimulator


def main() -> None:
    simulator = ChampionshipSimulator()

    single_run = simulator.simulate_season(include_advanced=True)
    final_table = single_run["final_standings"]
    race_results = single_run.get("race_results")

    monte_carlo_summary = simulator.run_monte_carlo(iterations=200, include_advanced=True)

    output_lines = []
    output_lines.append("=== Championship Simulation: Single Run ===")
    for _, row in final_table.head(10).iterrows():
        output_lines.append(
            f"#{int(row['driver_number']):02d} {row['driver_name'] or 'Driver'} "
            f"- Points: {row['season_points']:.1f} (Rank {int(row['rank'])})"
        )

    if race_results is not None and not race_results.empty:
        output_lines.append("\n=== Race-by-Race Summary ===")
        race_results = race_results.sort_values(["event_order", "final_position"])
        for event_order, group in race_results.groupby("event_order"):
            track_id = group["track_id"].iloc[0] if "track_id" in group.columns else f"event_{event_order}"
            weather = (
                group["weather_condition"].iloc[0]
                if "weather_condition" in group.columns and not group["weather_condition"].isna().all()
                else "default"
            )
            winner = group.iloc[0]
            winner_name = winner.get("driver_name")
            if winner_name is None or (isinstance(winner_name, float) and pd.isna(winner_name)):
                winner_name = f"Driver #{int(winner['driver_number'])}"
            output_lines.append(
                f"\nEvent {int(event_order)} - {track_id} [{weather}] | "
                f"Winner: #{int(winner['driver_number']):02d} {winner_name}"
            )
            top3 = group.head(3)
            for _, driver in top3.iterrows():
                driver_name = driver.get("driver_name")
                if driver_name is None or (isinstance(driver_name, float) and pd.isna(driver_name)):
                    driver_name = f"Driver #{int(driver['driver_number'])}"
                predicted_rank = driver.get("predicted_rank", driver.get("predicted_position", 0))
                output_lines.append(
                    f"  - P{int(driver['final_position'])}: "
                    f"#{int(driver['driver_number']):02d} "
                    f"{driver_name} | "
                    f"PredRank≈{predicted_rank:.1f}, "
                    f"PredLap={driver.get('predicted_lap_time', 0):.2f}s, "
                    f"FinishProb={driver.get('finish_probability', 0)*100:.1f}%, "
                    f"Status={'FIN' if driver.get('did_finish', True) else 'DNF'}"
                )

        output_lines.append("\n=== Prediction Snapshot (First Event) ===")
        first_event_order = race_results["event_order"].min()
        first_event = race_results[race_results["event_order"] == first_event_order]
        sample_rows = first_event.head(5)
        for _, driver in sample_rows.iterrows():
            driver_name = driver.get("driver_name")
            if driver_name is None or (isinstance(driver_name, float) and pd.isna(driver_name)):
                driver_name = f"Driver #{int(driver['driver_number'])}"
            predicted_rank = driver.get("predicted_rank", driver.get("predicted_position", 0))
            output_lines.append(
                f"  Driver #{int(driver['driver_number']):02d} "
                f"{driver_name} | "
                f"PredRank {predicted_rank:.2f} → Final P{int(driver['final_position'])} | "
                f"PredSpeed {driver.get('predicted_speed', 0):.1f} km/h | "
                f"FinishProb {driver.get('finish_probability', 0)*100:.1f}%"
            )

        total_events = race_results["event_order"].nunique()
        if "status" in race_results.columns:
            total_dnfs = int((race_results["status"] == "DNF").sum())
        elif "did_finish" in race_results.columns:
            total_dnfs = int((race_results["did_finish"] == False).sum())
        else:
            total_dnfs = 0
        if "laps" in race_results.columns:
            avg_laps = race_results["laps"].dropna().mean()
            output_lines.append(
                f"\nSummary: {total_events} events simulated | Total DNFs: {total_dnfs} | "
                f"Average laps recorded: {avg_laps:.1f}"
            )
        else:
            output_lines.append(
                f"\nSummary: {total_events} events simulated | Total DNFs: {total_dnfs}"
            )

    output_lines.append("\n=== Monte Carlo Summary (Top 10) ===")
    for _, row in monte_carlo_summary.head(10).iterrows():
        output_lines.append(
            f"#{int(row['driver_number']):02d} {row['driver_name']} "
            f"- AvgPts: {row['average_points']:.1f} | "
            f"Win%: {row['win_probability']*100:.1f}% | "
            f"Podium%: {row['podium_probability']*100:.1f}%"
        )
    output_lines.append(
        "\nNote: Monte Carlo averages reflect 200 simulated seasons and will differ from the single-run standings above."
    )

    output_path = Path("championship_simulation_results.txt")
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Simulation summaries written to {output_path.resolve()}")


if __name__ == "__main__":
    main()


