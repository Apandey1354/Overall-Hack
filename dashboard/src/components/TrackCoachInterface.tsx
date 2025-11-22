import { useMemo, useState } from "react";
import { TrackCoachEntry, DriverEmbedding } from "../types";
import { SectionCard } from "./SectionCard";
import { API_BASE_URL } from "../config";
import { Modal } from "./Modal";
import { BackendSetupGuide } from "./BackendSetupGuide";
import { isBackendConnectionError } from "../utils/errorDetection";

interface Props {
  entries: TrackCoachEntry[];
  driverEmbeddings: DriverEmbedding[];
}

export function TrackCoachInterface({ entries, driverEmbeddings }: Props) {
  const [trackId, setTrackId] = useState(() => entries[0]?.track_id ?? "");
  const [driverNumber, setDriverNumber] = useState<number>(() => driverEmbeddings[0]?.driver_number ?? 13);
  const [driverAdvice, setDriverAdvice] = useState<any>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [isBackendError, setIsBackendError] = useState(false);
  const [showTechnicalInfo, setShowTechnicalInfo] = useState(false);
  const [aiRecommendation, setAiRecommendation] = useState<string>();
  const [loadingAI, setLoadingAI] = useState(false);

  const selected = useMemo(
    () => entries.find((entry) => entry.track_id === trackId) ?? entries[0],
    [entries, trackId],
  );

  // Extract available drivers from embeddings
  const availableDrivers = useMemo(() => {
    return driverEmbeddings
      .map((emb) => ({
        number: emb.driver_number,
        name: emb.driver_name || `Driver #${emb.driver_number}`,
      }))
      .sort((a, b) => a.number - b.number);
  }, [driverEmbeddings]);

  if (!entries.length) {
    return (
      <SectionCard title="Track Coach Interface">
        <p>No coach data yet. Run `python scripts/build_dashboard_cache.py`.</p>
      </SectionCard>
    );
  }

  const fetchAdvice = async () => {
    setLoading(true);
    setError(undefined);
    try {
      const response = await fetch(`${API_BASE_URL}/coach/advice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: trackId, driver_number: driverNumber }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const json = await response.json();
      setDriverAdvice(json.advice);
    } catch (err) {
      const isConnectionError = isBackendConnectionError(err);
      setIsBackendError(isConnectionError);
      if (!isConnectionError) {
        setError(String(err));
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchAIRecommendation = async () => {
    setLoadingAI(true);
    setAiRecommendation(undefined);
    try {
      const response = await fetch(`${API_BASE_URL}/ai/coach-recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: trackId, driver_number: driverNumber }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const json = await response.json();
      setAiRecommendation(json.recommendation);
    } catch (err) {
      const isConnectionError = isBackendConnectionError(err);
      setIsBackendError(isConnectionError);
      if (!isConnectionError) {
        setError(String(err));
      }
    } finally {
      setLoadingAI(false);
    }
  };

  return (
    <SectionCard
      title="Track Coach Interface"
      description="Surface tailored coaching intel per venue."
      action={
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <label style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", fontWeight: 500 }}>
            Select Track here
          </label>
          <select 
            value={selected?.track_id || ""} 
            onChange={(event) => {
              if (event.target.value) {
                setTrackId(event.target.value);
              }
            }}
          >
            <option value="" disabled style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
              🔽 Select to change track...
            </option>
            {entries.map((entry) => (
              <option key={entry.track_id} value={entry.track_id}>
                {entry.track_id}
              </option>
            ))}
          </select>
        </div>
      }
    >
      {selected && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Track Overview Section */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              📍 Track Overview
            </h4>
            <div className="grid-2" style={{ gap: "1rem" }}>
              <div style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Track</div>
                <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>{selected.track_id}</div>
              </div>
              <div style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Recommended Focus</div>
                <div style={{ fontSize: "0.95rem", fontWeight: 500, color: "#00ffc6" }}>
                  {selected.recommended_focus || "N/A"}
                </div>
              </div>
              <div style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Complexity</div>
                <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>
                  {selected.complexity_score ? (selected.complexity_score * 100).toFixed(0) + "%" : "N/A"}
                </div>
              </div>
              <div style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Sectors</div>
                <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>
                  {selected.num_sectors ?? "N/A"}
                </div>
              </div>
            </div>
          </div>

          {/* Sector Focus Section */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              🎯 Sector Focus Recommendations
            </h4>
            <div style={{ 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "12px", 
              padding: "1rem",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem" }}>
                {selected.sector_recommendations.slice(0, 4).map((tip: any, idx: number) => (
                  <div 
                    key={tip.sector ?? idx}
                    style={{
                      padding: "0.75rem",
                      background: "rgba(255, 255, 255, 0.05)",
                      borderRadius: "8px",
                      border: "1px solid rgba(255, 255, 255, 0.1)"
                    }}
                  >
                    <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#00ffc6", marginBottom: "0.5rem" }}>
                      {tip.sector || `Sector ${idx + 1}`}
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.9)", marginBottom: "0.25rem" }}>
                      <strong>Focus:</strong> {tip.focus}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "rgba(255, 255, 255, 0.6)" }}>
                      Variance: {tip.variance?.toFixed?.(3) ?? tip.variance}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Driver Advice Section */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              👤 Get Personalized Driver Advice
            </h4>
            <div style={{ 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "12px", 
              padding: "1rem",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "flex-end" }}>
                <div className="form-field" style={{ flex: "1", minWidth: "200px" }}>
                  <label style={{ display: "block", fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.5rem", fontWeight: 500 }}>
                    Select Driver here
                  </label>
                  <select
                    value={driverNumber || ""}
                    onChange={(event) => {
                      if (event.target.value) {
                        setDriverNumber(Number(event.target.value));
                      }
                    }}
                  >
                    <option value="" disabled style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
                      🔽 Select to change driver...
                    </option>
                    {availableDrivers.map((driver) => (
                      <option key={driver.number} value={driver.number}>
                        #{driver.number} - {driver.name}
                      </option>
                    ))}
                  </select>
                </div>
                <button 
                  className="tab-button active" 
                  onClick={fetchAdvice} 
                  disabled={loading}
                  style={{ minWidth: "150px" }}
                >
                  {loading ? "Fetching..." : "Get Driver Advice"}
                </button>
                <button 
                  className="ghost-button" 
                  onClick={fetchAIRecommendation} 
                  disabled={loadingAI}
                  style={{ minWidth: "180px" }}
                >
                  {loadingAI ? "Generating..." : "🤖 Get AI Recommendations"}
                </button>
              </div>
            </div>
          </div>

          {/* Technical Information Section (Collapsible) */}
          <div>
            <button
              onClick={() => setShowTechnicalInfo(!showTechnicalInfo)}
              className="ghost-button"
              style={{ width: "100%", marginBottom: "0.5rem" }}
            >
              {showTechnicalInfo ? "🔽 Hide" : "▶️ Show"} Detailed Technical Information: How Track Coach Works
            </button>
            
            {showTechnicalInfo && (
              <div
                style={{
                  background: "rgba(0, 255, 198, 0.1)",
                  border: "1px solid rgba(0, 255, 198, 0.3)",
                  borderRadius: "16px",
                  padding: "1.25rem",
                  marginTop: "0.5rem",
                }}
              >
                <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", fontWeight: 600, color: "#00ffc6" }}>
                  🔬 Track Coach Algorithms & Data Sources
                </h3>
                
                <div style={{ marginBottom: "1.5rem" }}>
                  <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                    Core Data Sources:
                  </h4>
                  <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                    <li><strong>Track DNA:</strong> Extracted from telemetry and analysis files, includes technical complexity score, speed profile (straight/corner ratio), sector time variance, braking zones, and physical characteristics</li>
                    <li><strong>Driver Embeddings:</strong> 8-dimensional skill vectors encoding technical proficiency, high-speed ability, consistency score, weather adaptability, and track-specific strengths/weaknesses</li>
                    <li><strong>Sector Analysis Data:</strong> Computed from "Analysis Endurance with Sections" CSV files, calculates variance, standard deviation, and coefficient of variation per sector</li>
                    <li><strong>Historical Performance:</strong> Past race results, lap times, and finishing positions used for pattern recognition</li>
                  </ul>
                </div>

                <div style={{ marginBottom: "1.5rem" }}>
                  <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                    Sector Recommendation Algorithm:
                  </h4>
                  <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                    <li><strong>Variance Analysis:</strong> Calculates sector time variance from lap-by-lap telemetry data</li>
                    <li><strong>Focus Classification:</strong> Categorizes sectors based on variance thresholds:
                      <ul style={{ marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                        <li>Variance &gt; 50.0: "stability" focus (critical inconsistency detected)</li>
                        <li>Variance 5.0-50.0: "precision" focus (moderate spread, needs refinement)</li>
                        <li>Variance &lt; 5.0: "attack speed" focus (consistent, can push harder)</li>
                      </ul>
                    </li>
                    <li><strong>Driver-Specific Tips:</strong> When driver_number provided, compares driver's consistency_score against sector variance:
                      <ul style={{ marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                        <li>High variance + low consistency: Recommends braking point review and track limit awareness</li>
                        <li>Moderate variance: Suggests trimming entry speed and smoothing brake release</li>
                        <li>Low variance + high consistency: Encourages carrying more apex speed and earlier throttle</li>
                      </ul>
                    </li>
                    <li><strong>Standard Deviation Calculation:</strong> Uses std_seconds = √variance when available, or computes from coefficient of variation (CV)</li>
                  </ol>
                </div>

                <div style={{ marginBottom: "1.5rem" }}>
                  <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                    Weather Strategy Generation:
                  </h4>
                  <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                    <li><strong>Rain Conditions:</strong> Focus = "traction", recommends smooth throttle application, early upshifts, and defensive lines through high-variance sectors</li>
                    <li><strong>Hot Conditions:</strong> Focus = "tire management", suggests reducing sliding in critical sectors to control tire temperatures</li>
                    <li><strong>Cold Conditions:</strong> Focus = "warm-up", emphasizes building temperature on straights before pushing on timed laps</li>
                    <li><strong>Default Conditions:</strong> Focus = "balanced", optimizes balance between corner entry stability and rotation</li>
                    <li><strong>Track Length Adjustment:</strong> For tracks with &gt;3 sectors, adds note about monitoring delta per sector to catch time losses early</li>
                  </ul>
                </div>

                <div style={{ marginBottom: "1.5rem" }}>
                  <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                    Driver-Specific Advice Generation:
                  </h4>
                  <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                    <li><strong>Profile Extraction:</strong> Retrieves driver's skill scores (technical, speed, consistency) and strengths/weaknesses from embeddings</li>
                    <li><strong>Track-Driver Matching:</strong> Compares driver profile against track DNA characteristics:
                      <ul style={{ marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                        <li>If complexity_score &gt; 0.6 and driver lacks "technical" strength: Recommends trail braking and rotation work</li>
                        <li>If speed_ratio &gt; 1.2 and driver lacks "speed-focused" strength: Suggests maximizing exit speed using transfer model predictions</li>
                        <li>If no specific gaps: Recommends maintaining current approach with telemetry refinement</li>
                      </ul>
                    </li>
                    <li><strong>High-Variance Sector Identification:</strong> Flags sectors with std_seconds ≥ 1.0 as focus areas, recommending data review to reduce lap time swings</li>
                    <li><strong>Strength/Weakness Analysis:</strong> Extracts from driver embedding's track_specific_strengths and weaknesses arrays</li>
                  </ol>
                </div>

                <div>
                  <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                    Knowledge Base Construction:
                  </h4>
                  <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                    <li><strong>Complexity Classification:</strong> If complexity_score ≥ 0.55: "technical" track; if ≤ 0.35: "low-complexity" track</li>
                    <li><strong>Speed Profile Classification:</strong> If speed_ratio ≥ 1.15: "high-speed" track; if ≤ 0.9: "corner-heavy" track</li>
                    <li><strong>Sector Count Analysis:</strong> If num_sectors ≥ 5: "flowing" track requiring rhythm maintenance</li>
                    <li><strong>Recommendation Generation:</strong> Combines dominant features to generate actionable focus areas (e.g., "Link apexes and manage rotation through tighter corners" for technical tracks)</li>
                    <li><strong>Track Overview:</strong> Aggregates complexity_score, track_length_km, num_sectors, dominant_features, and recommended_focus into unified summary</li>
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Technical Details Section (Permanent) */}
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
                Track Coach intelligence comes from the <code style={{ 
                  background: "rgba(255, 255, 255, 0.1)", 
                  padding: "0.2rem 0.4rem", 
                  borderRadius: "4px",
                  fontSize: "0.85rem"
                }}>TrackCoach</code> class, which pairs driver embeddings with each venue's DNA:
              </p>
              <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Sector Recommendations:</strong> Calculates variance from sector timing data and classifies sectors as "stability", 
                  "precision", or "attack speed" based on variance thresholds. Driver-specific tips compare driver consistency against sector variance.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Weather Strategies:</strong> Generates condition-specific playbooks (rain, hot, cold, default) with focus areas and actionable 
                  notes. Adjusts throttle/braking guidance based on track characteristics and weather conditions.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Driver Advice:</strong> Matches driver skill profile (from embeddings) against track DNA characteristics to identify 
                  strengths/weaknesses and recommend focus areas. Uses track-type analysis to suggest improvements.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Data Integration:</strong> Combines Track DNA (complexity, speed profile, sector variance), Driver Embeddings (8-dimensional 
                  skill vectors), and historical performance data to generate personalized coaching recommendations.
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
      {isBackendError && <BackendSetupGuide />}
      {error && !isBackendError && <p className="error">{error}</p>}

      {/* AI Recommendations Section */}
      {aiRecommendation && (
        <div
          style={{
            marginTop: "1.5rem",
            background: "rgba(255, 51, 88, 0.1)",
            border: "1px solid rgba(255, 51, 88, 0.3)",
            borderRadius: "16px",
            padding: "1.25rem",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
            <h3 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 600, color: "#ff3358" }}>
              🤖 AI-Powered Recommendations
            </h3>
            <button
              onClick={() => setAiRecommendation(undefined)}
              style={{
                background: "rgba(255, 255, 255, 0.1)",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                borderRadius: "8px",
                padding: "0.25rem 0.5rem",
                color: "rgba(255, 255, 255, 0.8)",
                cursor: "pointer",
                fontSize: "0.85rem",
              }}
            >
              ✕
            </button>
          </div>
          <div
            style={{
              color: "rgba(255, 255, 255, 0.9)",
              lineHeight: "1.8",
              fontSize: "0.95rem",
              whiteSpace: "pre-line",
            }}
          >
            {aiRecommendation}
          </div>
          <p style={{ marginTop: "0.75rem", marginBottom: 0, fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", fontStyle: "italic" }}>
            💡 These recommendations are generated by AI based on the Coach Lab insights above, including sector recommendations, weather strategies, and driver-specific advice.
          </p>
        </div>
      )}
      
      <Modal
        isOpen={!!driverAdvice}
        onClose={() => setDriverAdvice(undefined)}
        title={`Driver Advice for #${driverAdvice?.driver_number ?? driverNumber} on ${trackId}`}
      >
        <div className="scenario-results">
          <h3>Focus Areas</h3>
          <ul>
            {(driverAdvice?.focus ?? []).map((item: string, idx: number) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
          {driverAdvice?.strengths && (
            <div style={{ marginTop: "1.5rem" }}>
              <h3>Strengths</h3>
              <p>
                {Array.isArray(driverAdvice.strengths) ? driverAdvice.strengths.join(", ") : driverAdvice.strengths}
              </p>
            </div>
          )}
          {driverAdvice?.weaknesses && (
            <div style={{ marginTop: "1.5rem" }}>
              <h3>Weaknesses</h3>
              <p>
                {Array.isArray(driverAdvice.weaknesses) ? driverAdvice.weaknesses.join(", ") : driverAdvice.weaknesses}
              </p>
            </div>
          )}
        </div>
      </Modal>
    </SectionCard>
  );
}

