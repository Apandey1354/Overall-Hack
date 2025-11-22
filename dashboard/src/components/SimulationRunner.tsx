import { useState } from "react";
import { API_BASE_URL } from "../config";
import { SectionCard } from "./SectionCard";
import { BackendSetupGuide } from "./BackendSetupGuide";
import { isBackendConnectionError } from "../utils/errorDetection";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  Legend,
} from "recharts";

interface ImpactReport {
  event_order: number;
  track_id: string;
  impact_score: number;
  champion_changed: boolean;
  champion_before: string;
  champion_after: string;
  max_points_delta: number;
  key_movers: Array<{
    driver_number: number;
    driver_name: string;
    points_delta_abs: number;
    rank_delta: number;
  }>;
}

interface SimulationResult {
  single_run: {
    final_standings: Array<{
      driver_number: number;
      driver_name: string;
      season_points: number;
      rank: number;
    }>;
    race_results: Array<{
      event_order: number;
      track_id: string;
      driver_number: number;
      driver_name: string;
      final_position: number;
      total_points: number;
      weather_condition: string;
    }>;
  };
  monte_carlo?: Array<{
    driver_number: number;
    driver_name: string;
    average_points: number;
    win_probability: number;
    podium_probability: number;
  }>;
  impact_reports?: ImpactReport[];
}

export function SimulationRunner() {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [iterations, setIterations] = useState(200);
  const [includeAdvanced, setIncludeAdvanced] = useState(true);
  const [useCachedSeed, setUseCachedSeed] = useState(true);
  const [customSeed, setCustomSeed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isBackendError, setIsBackendError] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isQuickRun, setIsQuickRun] = useState(false);

  const runSimulation = async (quick: boolean = false) => {
    setRunning(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setIsQuickRun(quick);

    // Simulate progress updates (since backend doesn't provide real-time progress)
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        // Quick run: faster progress (30 seconds estimated)
        // Full run: slower progress (based on iterations)
        const increment = quick ? 3 : Math.max(1, Math.floor(200 / iterations));
        return Math.min(prev + increment, 95); // Cap at 95% until complete
      });
    }, 500);

    try {
      let response;
      if (quick) {
        response = await fetch(`${API_BASE_URL}/simulation/quick`);
      } else {
        response = await fetch(`${API_BASE_URL}/simulation/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            iterations,
            include_advanced: includeAdvanced,
            run_monte_carlo: true,
            random_seed: useCachedSeed ? null : (customSeed || null),
          }),
        });
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Simulation failed: ${response.statusText}`);
      }

      const data = await response.json();
      setProgress(100); // Complete
      setTimeout(() => {
        setResult(data);
      }, 300); // Small delay to show 100% completion
    } catch (err) {
      const isConnectionError = isBackendConnectionError(err);
      setIsBackendError(isConnectionError);
      if (isConnectionError) {
        setError(null); // Don't show generic error, show setup guide instead
      } else {
        setError(err instanceof Error ? err.message : "Simulation failed");
      }
      console.error("Simulation error:", err);
    } finally {
      clearInterval(progressInterval);
      setRunning(false);
      setProgress(0);
    }
  };

  const standingsData = result?.single_run.final_standings
    .slice(0, 10)
    .map((driver) => ({
      name: `#${driver.driver_number}`,
      points: driver.season_points,
      rank: driver.rank,
    })) || [];

  const monteCarloData = result?.monte_carlo
    ?.slice(0, 10)
    .map((driver) => ({
      name: `#${driver.driver_number}`,
      winProb: (driver.win_probability || 0) * 100,
      podiumProb: (driver.podium_probability || 0) * 100,
    })) || [];

  const raceResultsByEvent = result?.single_run.race_results.reduce((acc, race) => {
    if (!acc[race.event_order]) {
      acc[race.event_order] = [];
    }
    acc[race.event_order].push(race);
    return acc;
  }, {} as Record<number, typeof result.single_run.race_results>) || {};

  return (
    <div className="space-y-6">
      <SectionCard
        title="Simulate Championship"
        description="Run interactive simulations to predict championship outcomes. Adjust parameters and see results in real-time."
      >
        <div className="space-y-4">
          {/* Controls */}
          <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">Simulation Controls</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Monte Carlo Iterations:
                </label>
                <input
                  type="number"
                  min="10"
                  max="1000"
                  value={iterations}
                  onChange={(e) => setIterations(parseInt(e.target.value) || 200)}
                  disabled={running}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-[#00ffc6]"
                />
                <p className="text-xs text-white/50 mt-1">
                  More iterations = more accurate but slower (10-1000)
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  Random Seed:
                </label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="seed"
                      checked={useCachedSeed}
                      onChange={() => {
                        setUseCachedSeed(true);
                        setCustomSeed(null);
                      }}
                      disabled={running}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">Use Cached Seed (matches Championship Fate Matrix)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="seed"
                      checked={!useCachedSeed}
                      onChange={() => setUseCachedSeed(false)}
                      disabled={running}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">Custom Seed:</span>
                    <input
                      type="number"
                      value={customSeed || ""}
                      onChange={(e) => setCustomSeed(e.target.value ? parseInt(e.target.value) : null)}
                      disabled={running || useCachedSeed}
                      placeholder="Random"
                      className="w-24 px-2 py-1 bg-white/5 border border-white/10 rounded text-white text-sm focus:outline-none focus:border-[#00ffc6]"
                    />
                  </label>
                </div>
              </div>
            </div>
            <div className="mb-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeAdvanced}
                  onChange={(e) => setIncludeAdvanced(e.target.checked)}
                  disabled={running}
                  className="w-4 h-4"
                />
                <span className="text-sm">Include Advanced Factors</span>
              </label>
              <p className="text-xs text-white/50 ml-6">
                (Weather, momentum, pressure, fatigue)
              </p>
            </div>
            <div className="flex gap-4">
              <button
                onClick={() => runSimulation(true)}
                disabled={running}
                className="px-6 py-2 bg-[#00ffc6] text-[#0f1117] rounded-lg font-semibold hover:bg-[#00e6b8] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {running ? "Running..." : "Quick Run (100 iterations)"}
              </button>
              <button
                onClick={() => runSimulation(false)}
                disabled={running}
                className="px-6 py-2 bg-[#ff3358] text-white rounded-lg font-semibold hover:bg-[#ff1a42] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {running ? "Running..." : `Run Full Simulation (${iterations} iterations)`}
              </button>
            </div>
          </div>

          {isBackendError && <BackendSetupGuide />}
          {error && !isBackendError && (
            <div className="p-4 bg-[#ff3358]/20 text-[#ff3358] border border-[#ff3358]/30 rounded-lg">
              Error: {error}
            </div>
          )}

          {!result && !running && !error && (
            <div className="p-6 bg-white/5 border border-white/10 rounded-lg text-center">
              <p className="text-lg font-semibold mb-2">Ready to Run Simulation</p>
              <p className="text-sm text-white/70 mb-4">
                Click "Quick Run" or "Run Full Simulation" above to start. Results will appear here.
              </p>
              <p className="text-xs text-white/50">
                Quick Run: ~10-30 seconds | Full Simulation: ~1-3 minutes (depending on iterations)
              </p>
            </div>
          )}

          {running && (
            <div className="p-6 bg-white/5 border border-white/10 rounded-lg">
              <div className="text-center mb-4">
                <p className="text-lg font-semibold mb-2">Running Simulation...</p>
                <p className="text-sm text-white/70 mb-4">
                  {isQuickRun 
                    ? "Quick simulation in progress (~10-30 seconds)"
                    : `Full simulation with ${iterations} iterations (~1-3 minutes)`
                  }
                </p>
              </div>
              <div className="w-full bg-white/10 rounded-full h-3 mb-2">
                <div
                  className="h-3 rounded-full transition-all duration-500 ease-out bg-gradient-to-r from-[#00ffc6] to-[#00e6b8]"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-white/50">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6">
              {/* Final Standings */}
              <div>
                <h3 className="text-lg font-semibold mb-4">🏆 Final Championship Standings</h3>
                <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={standingsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="name" stroke="rgba(255,255,255,0.7)" />
                      <YAxis stroke="rgba(255,255,255,0.7)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(15, 17, 23, 0.95)",
                          border: "1px solid rgba(255,255,255,0.2)",
                        }}
                      />
                      <Bar dataKey="points" fill="#00ffc6" />
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="mt-4">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="text-left py-2">Rank</th>
                          <th className="text-left py-2">Driver</th>
                          <th className="text-right py-2">Points</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.single_run.final_standings.slice(0, 10).map((driver) => (
                          <tr key={driver.driver_number} className="border-b border-white/5">
                            <td className="py-2">
                              <span
                                className={
                                  driver.rank === 1
                                    ? "text-[#ffd700] font-bold"
                                    : driver.rank === 2
                                    ? "text-[#c0c0c0] font-bold"
                                    : driver.rank === 3
                                    ? "text-[#cd7f32] font-bold"
                                    : ""
                                }
                              >
                                {driver.rank}
                              </span>
                            </td>
                            <td className="py-2">
                              #{driver.driver_number} {driver.driver_name}
                            </td>
                            <td className="text-right py-2 font-semibold">
                              {driver.season_points.toFixed(1)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Monte Carlo Probabilities */}
              {result.monte_carlo && result.monte_carlo.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">🎲 Monte Carlo Championship Probabilities</h3>
                  <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={monteCarloData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="name" stroke="rgba(255,255,255,0.7)" />
                        <YAxis stroke="rgba(255,255,255,0.7)" domain={[0, 100]} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "rgba(15, 17, 23, 0.95)",
                            border: "1px solid rgba(255,255,255,0.2)",
                          }}
                        />
                        <Legend />
                        <Bar dataKey="winProb" fill="#ffd700" name="Win %" />
                        <Bar dataKey="podiumProb" fill="#00ffc6" name="Podium %" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Race-by-Race Results */}
              <div>
                <h3 className="text-lg font-semibold mb-4">🏁 Race-by-Race Results</h3>
                <div className="space-y-4">
                  {Object.entries(raceResultsByEvent)
                    .sort(([a], [b]) => parseInt(a) - parseInt(b))
                    .map(([eventOrder, races]) => {
                      const sortedRaces = [...races].sort((a, b) => a.final_position - b.final_position);
                      const winner = sortedRaces[0];
                      return (
                        <div
                          key={eventOrder}
                          className="p-4 bg-white/5 border border-white/10 rounded-lg"
                        >
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="font-semibold">
                              Event {eventOrder} - {winner.track_id}
                            </h4>
                            <span className="text-sm text-white/70">
                              Weather: {winner.weather_condition || "Default"}
                            </span>
                          </div>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            {sortedRaces.slice(0, 3).map((race) => (
                              <div key={race.driver_number} className="p-2 bg-white/5 rounded">
                                <span className="font-bold text-[#00ffc6]">P{race.final_position}</span>{" "}
                                #{race.driver_number} {race.driver_name} ({race.total_points} pts)
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* Butterfly Effect Analysis */}
              {result.impact_reports && result.impact_reports.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">🦋 High-Impact Races (Butterfly Effect Analysis)</h3>
                  <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <p className="text-sm text-white/70 mb-4">
                      These races have the highest impact on championship outcomes. Changing results in these events 
                      would most significantly affect the final standings.
                    </p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-white/10">
                            <th className="text-left py-2">Event</th>
                            <th className="text-left py-2">Track</th>
                            <th className="text-right py-2">Impact Score</th>
                            <th className="text-left py-2">Champion Change</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.impact_reports.map((report) => (
                            <tr key={report.event_order} className="border-b border-white/5">
                              <td className="py-2 font-semibold">#{report.event_order}</td>
                              <td className="py-2">{report.track_id}</td>
                              <td className="text-right py-2 font-semibold" style={{
                                color: report.impact_score > 25 ? "#ff3358" : report.impact_score > 15 ? "#ff7043" : "#fff"
                              }}>
                                {report.impact_score.toFixed(2)}
                              </td>
                              <td className="py-2">
                                {report.champion_changed ? (
                                  <span className="text-[#00ffc6] font-semibold">
                                    {report.champion_before} → {report.champion_after}
                                  </span>
                                ) : (
                                  <span className="text-white/50">—</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {result.impact_reports.some(r => r.key_movers && r.key_movers.length > 0) && (
                      <div className="mt-4 space-y-3">
                        <h4 className="text-sm font-semibold mb-2">Key Movers (Top Impact Events):</h4>
                        {result.impact_reports
                          .filter(r => r.key_movers && r.key_movers.length > 0)
                          .slice(0, 3)
                          .map((report) => (
                            <div key={report.event_order} className="p-3 bg-white/5 rounded-lg">
                              <div className="text-sm font-semibold mb-2">
                                Event #{report.event_order} - {report.track_id}
                              </div>
                              <div className="text-xs text-white/70 space-y-1">
                                {report.key_movers.slice(0, 3).map((mover, idx) => (
                                  <div key={idx}>
                                    #{mover.driver_number} {mover.driver_name}: 
                                    {mover.points_delta_abs > 0 ? " +" : " "}
                                    {mover.points_delta_abs.toFixed(1)} pts, 
                                    Rank {mover.rank_delta > 0 ? "+" : ""}{mover.rank_delta}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </SectionCard>
    </div>
  );
}

