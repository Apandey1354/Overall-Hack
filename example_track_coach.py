"""
Example: Track Coach usage
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.track_coach import create_track_coach, TrackCoachConfig
from src.models.track_coaches.barber_coach import create_barber_coach
from src.models.track_coaches.cota_coach import create_cota_coach
from src.models.track_coaches.indy_coach import create_indianapolis_coach
from src.models.track_coaches.vir_coach import create_vir_coach
from src.data_processing.driver_embedder import create_driver_embeddings
from src.data_processing.track_dna_extractor import extract_all_tracks_dna
from src.data_processing.data_loader import DataLoader
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI

        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        print(f"Warning: OpenAI client initialization failed: {exc}")
        openai_client = None


def _generate_llm_recommendations(track_id, overview, sectors, advice):
    if not openai_client:
        return "LLM recommendations unavailable (missing OPENAI_API_KEY)."

    prompt = f"""
You are a professional racing coach.
Track: {track_id}
Overview: {overview}
Sectors: {sectors}
Driver Advice: {advice}

Provide three concise bullet-point recommendations that combine these insights into actionable coaching guidance.
"""
    try:
        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
            max_output_tokens=400,
        )
        if response.output and response.output[0].content:
            return response.output[0].content[0].text.strip()
        return "LLM response empty."
    except Exception as exc:
        return f"LLM recommendation unavailable: {exc}"


def main(driver_number: int = 13):
    loader = DataLoader()
    driver_embeddings_df = create_driver_embeddings(loader)
    track_dna_df = extract_all_tracks_dna(loader)

    track_ids = track_dna_df["track_id"].unique().tolist() if "track_id" in track_dna_df.columns else [
        f"{venue}_Race {race}" for venue in loader.venues for race in [1, 2]
    ]
    weather_conditions = ["default", "rain", "hot", "cold"]

    results = []

    def log(line=""):
        print(line)
        results.append(line)

    coach_factories = {
        "barber": create_barber_coach,
        "cota": create_cota_coach,
        "circuit of the americas": create_cota_coach,
        "indianapolis": create_indianapolis_coach,
        "virginia-international-raceway": create_vir_coach,
        "vir": create_vir_coach,
    }

    for track_id in track_ids:
        venue_key = track_id.split("_")[0].lower()
        factory = coach_factories.get(venue_key)
        config = TrackCoachConfig(
            track_id=track_id,
            driver_embeddings_df=driver_embeddings_df,
            track_dna_df=track_dna_df,
        )
        if factory:
            coach = factory(config=config, data_loader=loader)
        else:
            coach = create_track_coach(
                track_id=track_id,
                driver_embeddings_df=driver_embeddings_df,
                track_dna_df=track_dna_df,
                data_loader=loader,
            )
        log("=" * 80)
        log(f"TRACK: {track_id}")
        log("=" * 80)

        overview = coach.get_track_overview()
        log("Overview:")
        for k, v in overview.items():
            log(f"  - {k}: {v}")

        log("\nSector Recommendations:")
        sectors = coach.get_sector_recommendations(driver_number=driver_number)
        for tip in sectors:
            log(f"  - {tip['sector']}: focus={tip['focus']} variance={tip['variance']:.4f}")
            if "driver_tip" in tip:
                log(f"      Driver tip: {tip['driver_tip']}")

        for weather in weather_conditions:
            strategy = coach.get_weather_strategy(weather)
            log(f"\nWeather Strategy [{weather}]:")
            for note in strategy["notes"]:
                log(f"  - {note}")

        advice = coach.get_driver_advice(driver_number=driver_number)
        log("\nDriver Advice:")
        log(f"  Strengths: {advice['strengths']}")
        log(f"  Weaknesses: {advice['weaknesses']}")
        log("  Focus:")
        for item in advice["focus"]:
            log(f"    - {item}")

        if openai_client:
            llm_reco = _generate_llm_recommendations(
                track_id=track_id,
                overview=overview,
                sectors=sectors,
                advice=advice,
            )
            log("\nAI-Augmented Recommendations:")
            log(llm_reco)
        else:
            log("\nAI-Augmented Recommendations: Skipped (missing OPENAI_API_KEY)")

        log("\n")

    output_path = Path("track_coach_results.txt")
    output_path.write_text("\n".join(results), encoding="utf-8")
    log(f"Results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()

