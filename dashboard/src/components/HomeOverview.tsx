import { ChampionshipState, EDAData, TrackCoachEntry, TrackSummary } from "../types";
import { SectionCard } from "./SectionCard";

interface Props {
  trackSummary: TrackSummary[];
  championshipState: ChampionshipState | null;
  coachData: TrackCoachEntry[];
  edaData: EDAData | null;
}

export function HomeOverview({ trackSummary, championshipState, coachData, edaData }: Props) {
  const completeness = edaData?.completeness.summary;
  const telemetry = edaData?.telemetry;
  const champion = championshipState?.final_standings?.[0];

  const quickFacts = [
    { label: "Tracks in this build", value: trackSummary.length || "Not ready yet" },
    {
      label: "Drivers with points",
      value: championshipState?.final_standings?.length ?? "Not ready yet",
    },
    {
      label: "Telemetry rows per track",
      value: telemetry ? telemetry.samples.toLocaleString() : "Awaiting data",
    },
  ];

  const simpleNotes = [
    completeness
      ? `We checked ${completeness.total_checks} files and ${completeness.files_found} were present.`
      : "Run the data check to see which files might be missing.",
    champion
      ? `Right now #${champion.driver_number} ${champion.driver_name ?? ""} sits on top of the standings.`
      : "Standings will appear after you run the cache builder.",
    coachData.length
      ? "Track Coach already has playbooks for these venues, so you can pick any track/driver combo."
      : "Coach data loads once the cache builder finishes.",
  ];

  return (
    <>
      <SectionCard
        title="Quick facts"
        description="Simple numbers so you know what’s inside this build."
      >
        <div className="grid gap-4 md:grid-cols-3">
          {quickFacts.map((fact) => (
            <div key={fact.label} className="rounded-2xl border border-white/10 bg-carbon-800/60 p-4">
              <p className="text-xs uppercase tracking-[0.25em] text-white/60">{fact.label}</p>
              <div className="mt-2 text-2xl font-semibold text-white">{fact.value}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="What it means" description="Plain-language takeaways.">
        <ul className="space-y-3 text-white/85">
          {simpleNotes.map((note) => (
            <li key={note}>• {note}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Where to click next" description="Pick an area based on what you want to explore.">
        <div className="grid-2">
          <div>
            <h4>Tracks & driving tips</h4>
            <p className="text-white/75">
              Track DNA compares circuits in plain English. The Track Coach Lab gives bite-sized advice for any driver you
              select.
            </p>
          </div>
          <div>
            <h4>Season stories</h4>
            <p className="text-white/75">
              Use the Fate Matrix to follow the standings, then open Scenario Lab to change a finishing order and see how the
              title race changes.
            </p>
          </div>
        </div>
        <p className="mt-4 text-sm text-white/70">
          Need to reload everything? From the project root run{" "}
          <code>python scripts/build_dashboard_cache.py</code>, then restart the API and dashboard (<code>uvicorn ...</code>{" "}
          and <code>npm run dev</code>).
        </p>
      </SectionCard>
    </>
  );
}


