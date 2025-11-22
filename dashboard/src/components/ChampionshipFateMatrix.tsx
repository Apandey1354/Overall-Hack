import { useMemo } from "react";
import { ChampionshipState } from "../types";
import { SectionCard } from "./SectionCard";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  state: ChampionshipState | null;
}

export function ChampionshipFateMatrix({ state }: Props) {
  const topStandings = useMemo(
    () => state?.final_standings?.slice(0, 8) ?? [],
    [state],
  );
  const topImpacts = state?.impact_reports ?? [];
  const probabilityData = useMemo(() => {
    const summary = state?.monte_carlo_summary ?? [];
    return summary
      .map((row: any) => ({
        driver: `#${row.driver_number}`,
        winProb: (row.win_probability ?? 0) * 100,
        podiumProb: (row.podium_probability ?? 0) * 100,
      }))
      .slice(0, 8);
  }, [state]);

  if (!state) {
    return (
      <SectionCard title="Championship Fate Matrix">
        <p>Loading championship state ...</p>
      </SectionCard>
    );
  }

  return (
    <SectionCard
      title="Championship Fate Matrix"
      description="Live standings, Monte Carlo odds, and butterfly effect alerts."
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
        {/* Current Standings Section */}
        <div>
          <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
            🏆 Current Championship Standings
          </h4>
          <div style={{ 
            background: "rgba(255, 255, 255, 0.03)", 
            borderRadius: "12px", 
            padding: "1rem",
            border: "1px solid rgba(255, 255, 255, 0.1)"
          }}>
            <table className="data-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>Pos</th>
                  <th>Driver</th>
                  <th>Points</th>
                </tr>
              </thead>
              <tbody>
                {topStandings.map((row: any, idx: number) => (
                  <tr key={row.driver_number ?? idx}>
                    <td style={{ 
                      fontWeight: idx < 3 ? 700 : 500,
                      color: idx === 0 ? "#ffd700" : idx === 1 ? "#c0c0c0" : idx === 2 ? "#cd7f32" : "#fff"
                    }}>
                      {row.rank ?? idx + 1}
                    </td>
                    <td>
                      #{row.driver_number} {row.driver_name}
                    </td>
                    <td style={{ fontWeight: 600 }}>{row.season_points?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Monte Carlo Probabilities Section */}
        <div>
          <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
            🎲 Monte Carlo Championship Odds
          </h4>
          <div style={{ 
            background: "rgba(255, 255, 255, 0.03)", 
            borderRadius: "12px", 
            padding: "1rem",
            border: "1px solid rgba(255, 255, 255, 0.1)"
          }}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={probabilityData} margin={{ left: 5, right: 5, top: 10, bottom: 5 }}>
                <XAxis 
                  dataKey="driver" 
                  stroke="#fff" 
                  fontSize={12}
                  tick={{ fill: "#fff" }}
                />
                <YAxis 
                  stroke="#fff" 
                  unit="%" 
                  fontSize={12}
                  tick={{ fill: "#fff" }}
                  domain={[0, 100]}
                />
                <Tooltip 
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                  contentStyle={{ 
                    background: "rgba(15, 17, 23, 0.95)", 
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                    borderRadius: "8px"
                  }}
                />
                <Bar dataKey="winProb" fill="#ff3358" name="Win Probability" radius={[6, 6, 0, 0]}>
                  {probabilityData.map((_, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={index === 0 ? "#ff7043" : index === 1 ? "#ff8a65" : "#ff3358"} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* High-Impact Races Section */}
        <div>
          <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
            🦋 High-Impact Races (Butterfly Effect Analysis)
          </h4>
          <div style={{ 
            background: "rgba(255, 255, 255, 0.03)", 
            borderRadius: "12px", 
            padding: "1rem",
            border: "1px solid rgba(255, 255, 255, 0.1)"
          }}>
            <table className="data-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>Event</th>
                  <th>Track</th>
                  <th>Impact Score</th>
                  <th>Champion Change</th>
                </tr>
              </thead>
              <tbody>
                {topImpacts.slice(0, 8).map((report) => (
                  <tr key={report.event_order}>
                    <td style={{ fontWeight: 600 }}>#{report.event_order}</td>
                    <td>{report.track_id}</td>
                    <td style={{ 
                      fontWeight: 600,
                      color: report.impact_score > 25 ? "#ff3358" : report.impact_score > 15 ? "#ff7043" : "#fff"
                    }}>
                      {report.impact_score.toFixed(2)}
                    </td>
                    <td>
                      {report.champion_changed ? (
                        <span className="champion-change" style={{ 
                          color: "#00ffc6",
                          fontWeight: 600
                        }}>
                          {report.champion_before} → {report.champion_after}
                        </span>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Technical Details Section */}
        <div style={{ 
          marginTop: "1.5rem", 
          padding: "1rem", 
          background: "rgba(0, 255, 198, 0.05)",
          borderRadius: "12px",
          border: "1px solid rgba(0, 255, 198, 0.2)"
        }}>
          <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#00ffc6" }}>
            🔧 Technical Implementation Details
          </h4>
          <div style={{ color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
            <p style={{ margin: "0 0 0.75rem 0" }}>
              The Championship Fate Matrix combines multiple simulation techniques to predict championship outcomes:
            </p>
            <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
              <li style={{ marginBottom: "0.5rem" }}>
                <strong>Race Outcome Prediction:</strong> Uses <code style={{ 
                  background: "rgba(255, 255, 255, 0.1)", 
                  padding: "0.2rem 0.4rem", 
                  borderRadius: "4px",
                  fontSize: "0.85rem"
                }}>RaceOutcomePredictor</code> which blends driver embeddings (skill vectors), track DNA features, and transfer 
                learning model outputs. Each driver's predicted performance is adjusted by contextual factors: weather penalties, 
                mechanical wear, momentum, pressure, and fatigue.
              </li>
              <li style={{ marginBottom: "0.5rem" }}>
                <strong>Monte Carlo Simulation:</strong> Runs hundreds of independent season simulations (default: 500 iterations) 
                with randomized weather, mechanical events, and performance variations. Each iteration starts from a fresh driver 
                state. Win and podium probabilities are calculated as the percentage of iterations where a driver finishes in 
                those positions.
              </li>
              <li style={{ marginBottom: "0.5rem" }}>
                <strong>Points Calculation:</strong> Uses <code style={{ 
                  background: "rgba(255, 255, 255, 0.1)", 
                  padding: "0.2rem 0.4rem", 
                  borderRadius: "4px",
                  fontSize: "0.85rem"
                }}>PointsCalculator</code> with FIA points system (1st: 25, 2nd: 18, 3rd: 15, etc.) plus bonus points for pole 
                position (+1) and fastest lap (+1). Season totals are summed across all races.
              </li>
              <li style={{ marginBottom: "0.5rem" }}>
                <strong>Butterfly Effect Analysis:</strong> <code style={{ 
                  background: "rgba(255, 255, 255, 0.1)", 
                  padding: "0.2rem 0.4rem", 
                  borderRadius: "4px",
                  fontSize: "0.85rem"
                }}>ButterflyEffectAnalyzer</code> reruns isolated events with modified finishing orders to measure championship 
                impact. Impact scores quantify how much a single race result change affects final standings. Events that cause 
                champion changes receive higher impact scores.
              </li>
              <li style={{ marginBottom: "0.5rem" }}>
                <strong>Advanced Contextual Adjustments:</strong> Each race applies weather variation (±15%), mechanical wear 
                (3% degradation per race with 8% recovery), momentum effects (8% gain for wins, 15% decay), pressure penalties 
                (5% for championship leaders), and fatigue penalties (2% cumulative). These factors create realistic performance 
                variations across the season.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}

