import { useMemo, useState } from "react";
import { DriverEmbedding } from "../types";
import { SectionCard } from "./SectionCard";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  LabelList,
} from "recharts";

interface Props {
  embeddings: DriverEmbedding[];
}

const SKILL_LABELS = [
  "Technical",
  "High-Speed",
  "Consistency",
  "Weather",
  "Tech-Track",
  "Speed-Track",
  "Balanced-Track",
  "Finish-Rate",
];

export function DriverEmbeddingsView({ embeddings }: Props) {
  const [selectedDriver, setSelectedDriver] = useState<number>(() => 
    embeddings[0]?.driver_number ?? 0
  );
  const [showTechnicalInfo, setShowTechnicalInfo] = useState(false);

  const selected = useMemo(
    () => embeddings.find((emb) => emb.driver_number === selectedDriver) ?? embeddings[0],
    [embeddings, selectedDriver]
  );

  // Prepare radar chart data
  const radarData = useMemo(() => {
    if (!selected || !selected.skill_vector || selected.skill_vector.length < 8) {
      return [];
    }
    return SKILL_LABELS.map((label, idx) => ({
      skill: label,
      value: selected.skill_vector[idx] ?? 0,
    }));
  }, [selected]);

  // Prepare comparison bar chart data (top drivers by each skill)
  const comparisonData = useMemo(() => {
    if (embeddings.length === 0) return [];
    
    const topBySkill = SKILL_LABELS.map((label, skillIdx) => {
      // Calculate statistics for this skill
      const allValues = embeddings
        .map(emb => emb.skill_vector?.[skillIdx] ?? 0)
        .filter(val => val > 0);
      
      const avgValue = allValues.length > 0 
        ? allValues.reduce((sum, val) => sum + val, 0) / allValues.length 
        : 0;
      const maxValue = allValues.length > 0 ? Math.max(...allValues) : 0;
      const minValue = allValues.length > 0 ? Math.min(...allValues) : 0;
      
      const sorted = [...embeddings]
        .sort((a, b) => {
          const aVal = a.skill_vector?.[skillIdx] ?? 0;
          const bVal = b.skill_vector?.[skillIdx] ?? 0;
          return bVal - aVal;
        })
        .slice(0, 10); // Show top 10 instead of 5
      
      return {
        skill: label,
        topDrivers: sorted.map((emb, idx) => {
          const value = emb.skill_vector?.[skillIdx] ?? 0;
          const percentile = allValues.length > 0
            ? ((allValues.filter(v => v < value).length / allValues.length) * 100).toFixed(1)
            : "0.0";
          
          return {
            driver: `#${emb.driver_number}`,
            driverName: emb.driver_name || `Driver #${emb.driver_number}`,
            value: value,
            rank: idx + 1,
            percentile: percentile,
            vsAverage: ((value - avgValue) * 100).toFixed(1),
          };
        }),
        stats: {
          average: avgValue,
          max: maxValue,
          min: minValue,
          totalDrivers: allValues.length,
        },
      };
    });
    
    return topBySkill;
  }, [embeddings]);

  if (!embeddings.length) {
    return (
      <SectionCard title="Driver Embeddings">
        <p>No driver embeddings found. Run `python scripts/build_dashboard_cache.py` first.</p>
      </SectionCard>
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title="Driver Skill Embeddings"
        description="8-dimensional skill vectors for each driver"
        action={
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <label style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", fontWeight: 500 }}>
              Select Driver here
            </label>
            <select 
              value={selectedDriver || ""} 
              onChange={(event) => {
                if (event.target.value) {
                  setSelectedDriver(Number(event.target.value));
                }
              }}
            >
              <option value="" disabled style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
                🔽 Select to change driver...
              </option>
              {embeddings.map((emb) => (
                <option key={emb.driver_number} value={emb.driver_number}>
                  #{emb.driver_number} - {emb.driver_name}
                </option>
              ))}
            </select>
          </div>
        }
      >
        {selected && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            {/* Driver Overview Section */}
            <div>
              <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
                👤 Driver Overview
              </h4>
              <div className="grid-2" style={{ gap: "1rem" }}>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px",
                  border: "1px solid rgba(255, 255, 255, 0.1)"
                }}>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Driver</div>
                  <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>
                    #{selected.driver_number} {selected.driver_name}
                  </div>
                </div>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px",
                  border: "1px solid rgba(255, 255, 255, 0.1)"
                }}>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Best Track Type</div>
                  <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#00ffc6" }}>
                    {selected.best_track_type || "N/A"}
                  </div>
                </div>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px",
                  border: "1px solid rgba(255, 255, 255, 0.1)"
                }}>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Strengths</div>
                  <div style={{ fontSize: "0.95rem", fontWeight: 500, color: "#fff" }}>
                    {selected.strengths || "N/A"}
                  </div>
                </div>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px",
                  border: "1px solid rgba(255, 255, 255, 0.1)"
                }}>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Weather Adaptability</div>
                  <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>
                    {(selected.weather_adaptability * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Core Skill Metrics Section */}
            <div>
              <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
                📊 Core Skill Metrics
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
                      <th>Skill</th>
                      <th>Score</th>
                      <th>Skill</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>Technical Proficiency</strong></td>
                      <td style={{ fontWeight: 600, color: "#00ffc6" }}>
                        {(selected.technical_proficiency * 100).toFixed(1)}%
                      </td>
                      <td><strong>High-Speed Proficiency</strong></td>
                      <td style={{ fontWeight: 600, color: "#00ffc6" }}>
                        {(selected.high_speed_proficiency * 100).toFixed(1)}%
                      </td>
                    </tr>
                    <tr>
                      <td><strong>Consistency Score</strong></td>
                      <td style={{ fontWeight: 600, color: "#00ffc6" }}>
                        {(selected.consistency_score * 100).toFixed(1)}%
                      </td>
                      <td><strong>Finish Rate</strong></td>
                      <td style={{ fontWeight: 600, color: "#00ffc6" }}>
                        {((selected.skill_vector?.[7] ?? 0) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Skill Vector Visualization Section */}
            <div>
              <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
                📈 Skill Vector Radar Chart
              </h4>
              <div style={{ 
                background: "rgba(255, 255, 255, 0.03)", 
                borderRadius: "12px", 
                padding: "1rem",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <ResponsiveContainer width="100%" height={320}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.2)" />
                    <PolarAngleAxis dataKey="skill" stroke="#fff" fontSize={12} />
                    <PolarRadiusAxis 
                      angle={90} 
                      domain={[0, 1]} 
                      stroke="rgba(255,255,255,0.4)"
                      tickFormatter={(value) => (value * 100).toFixed(0) + "%"}
                    />
                    <Tooltip 
                      formatter={(value: number) => [(value * 100).toFixed(1) + "%", "Score"]}
                      contentStyle={{ 
                        background: "rgba(15, 17, 23, 0.95)", 
                        border: "1px solid rgba(255, 255, 255, 0.2)",
                        borderRadius: "8px",
                        color: "#fff"
                      }}
                    />
                    <Radar 
                      name="Skill Score" 
                      dataKey="value" 
                      stroke="#00ffc6" 
                      fill="rgba(0, 255, 198, 0.4)"
                      strokeWidth={2}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Technical Information Section (Collapsible) */}
            <div>
              <button
                onClick={() => setShowTechnicalInfo(!showTechnicalInfo)}
                className="ghost-button"
                style={{ width: "100%", marginBottom: "0.5rem" }}
              >
                {showTechnicalInfo ? "🔽 Hide" : "▶️ Show"} Detailed Technical Information: How Driver Embeddings Are Calculated
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
                    🔬 Driver Embedding Calculation Algorithms
                  </h3>
                  
                  <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                      8-Dimensional Skill Vector Components:
                    </h4>
                    <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                      <li><strong>Technical Proficiency:</strong> Calculated from performance at technical tracks (high complexity, many corners). Uses track-type analysis from TrackPerformanceAnalyzer, combining average position (normalized: 1.0 - (pos-1)/19) and performance scores. Fallback uses position-based scoring if track-type data unavailable.</li>
                      <li><strong>High-Speed Proficiency:</strong> Derived from performance at speed-focused tracks (high straight/corner ratio). Primary method: track-type analysis for speed-focused tracks. Fallback: 70% fastest-lap speed percentile (vs. all drivers) + 30% position component.</li>
                      <li><strong>Consistency Score:</strong> Weighted combination of:
                        <ul style={{ marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                          <li>Lap time coefficient of variation (CV): 1/(1 + CV×100) - lower CV = higher score</li>
                          <li>Position variance: 1/(1 + variance/10) - lower variance = higher score</li>
                          <li>Finish rate: Direct percentage of races finished (DNF/DSQ excluded)</li>
                        </ul>
                        Final score: 50% CV component + 30% position component + 20% finish rate
                      </li>
                      <li><strong>Weather Adaptability:</strong> Measures performance consistency across weather conditions. Compares average position in rain vs. dry conditions. Adaptability = 1.0 - |rain_performance - dry_performance|, clamped to [0,1]. If only one condition available, uses that condition's performance score.</li>
                      <li><strong>Tech-Track Score:</strong> Normalized performance score on technical tracks (from track-specific strengths analysis). Uses performance_score or position-based scoring, normalized to sum with other track types.</li>
                      <li><strong>Speed-Track Score:</strong> Normalized performance score on speed-focused tracks. If zero, computed via fastest-lap speed percentile or high_speed_proficiency score.</li>
                      <li><strong>Balanced-Track Score:</strong> Normalized performance score on balanced tracks (neither highly technical nor speed-focused).</li>
                      <li><strong>Finish Rate:</strong> Direct from consistency metrics - percentage of races completed (excludes DNF, DSQ, NC, DNQ).</li>
                    </ol>
                  </div>

                  <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                      Calculation Methods:
                    </h4>
                    <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                      <li><strong>Track-Type Analysis:</strong> Uses TrackPerformanceAnalyzer to classify tracks (technical, speed-focused, balanced) and compute driver performance per type using average positions and performance scores</li>
                      <li><strong>Speed Percentile Calculation:</strong> Builds cache of all fastest-lap speeds (FL_KPH) across all venues/races, then computes percentile rank: (count of speeds &lt; driver_speed) / total_speeds</li>
                      <li><strong>Position Normalization:</strong> Converts finishing positions to 0-1 scale: pos_score = 1.0 - (position - 1) / 19.0 (1st = 1.0, 20th = 0.0)</li>
                      <li><strong>Min-Max Scaling:</strong> All intermediate values normalized using min-max scaling: (value - min) / (max - min), with special handling for constant arrays</li>
                      <li><strong>Robust Clipping:</strong> Final vector components clipped to [0,1] using _clip01() which handles NaN, infinity, and out-of-range values</li>
                    </ul>
                  </div>

                  <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                      Track-Specific Strengths Algorithm:
                    </h4>
                    <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                      <li><strong>Raw Score Extraction:</strong> Gets performance scores for each track type (technical, speed-focused, balanced) from TrackPerformanceAnalyzer</li>
                      <li><strong>Normalization:</strong> Converts raw scores to probability distribution: track_scores = raw_scores / sum(raw_scores), ensuring they sum to 1.0</li>
                      <li><strong>Strength/Weakness Identification:</strong>
                        <ul style={{ marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                          <li>Strengths: Track types where score &gt; average_score × 1.1</li>
                          <li>Weaknesses: Track types where score &lt; average_score × 0.9</li>
                          <li>Best Track Type: Track type with highest normalized score</li>
                        </ul>
                      </li>
                      <li><strong>Fallback Handling:</strong> If all track scores are zero, applies tiny deterministic variation based on driver number to avoid identical vectors</li>
                    </ol>
                  </div>

                  <div>
                    <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                      Data Sources & Processing:
                    </h4>
                    <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                      <li><strong>Race Results:</strong> Provisional and official results files (POSITION, FL_TIME, FL_KPH, STATUS columns)</li>
                      <li><strong>Weather Data:</strong> Weather CSV files (AIR_TEMP, RAIN columns) for condition-based analysis</li>
                      <li><strong>Track DNA:</strong> Track classification (technical/speed-focused/balanced) from TrackDNAExtractor</li>
                      <li><strong>Performance Analyzer:</strong> TrackPerformanceAnalyzer provides track-type-specific performance metrics</li>
                      <li><strong>Lap Time Parsing:</strong> Handles multiple formats (M:SS.mmm, SS.mmm, mm:ss) and converts to seconds for CV calculation</li>
                      <li><strong>Finish Status Detection:</strong> Identifies DNF/DSQ/NC/DNQ from STATUS column to compute finish rates</li>
                    </ul>
                  </div>
                </div>
              )}
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
                  Driver embeddings are 8-dimensional skill vectors extracted from historical performance data using the <code style={{ 
                    background: "rgba(255, 255, 255, 0.1)", 
                    padding: "0.2rem 0.4rem", 
                    borderRadius: "4px",
                    fontSize: "0.85rem"
                  }}>DriverEmbedder</code> class:
                </p>
                <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
                  <li style={{ marginBottom: "0.5rem" }}>
                    <strong>Technical Proficiency:</strong> Calculated from performance at technical tracks (high complexity, many corners). 
                    Uses track-type analysis from <code style={{ 
                      background: "rgba(255, 255, 255, 0.1)", 
                      padding: "0.2rem 0.4rem", 
                      borderRadius: "4px",
                      fontSize: "0.85rem"
                    }}>TrackPerformanceAnalyzer</code>, combining average position (normalized: 1.0 - (pos-1)/19) and performance scores.
                  </li>
                  <li style={{ marginBottom: "0.5rem" }}>
                    <strong>High-Speed Proficiency:</strong> Derived from performance at speed-focused tracks (high straight/corner ratio). 
                    Primary method uses track-type analysis. Fallback: 70% fastest-lap speed percentile (vs. all drivers) + 30% position component.
                  </li>
                  <li style={{ marginBottom: "0.5rem" }}>
                    <strong>Consistency Score:</strong> Weighted combination of lap time coefficient of variation (CV), position variance, 
                    and finish rate. Formula: 50% CV component + 30% position component + 20% finish rate. Lower variance = higher score.
                  </li>
                  <li style={{ marginBottom: "0.5rem" }}>
                    <strong>Weather Adaptability:</strong> Measures performance consistency across weather conditions by comparing average 
                    position in rain vs. dry conditions. Adaptability = 1.0 - |rain_performance - dry_performance|, clamped to [0,1].
                  </li>
                  <li style={{ marginBottom: "0.5rem" }}>
                    <strong>Track-Type Scores:</strong> Tech-Track, Speed-Track, and Balanced-Track scores are normalized performance scores 
                    on each track type. Scores are converted to probability distribution (sum to 1.0) to identify driver strengths and weaknesses.
                  </li>
                  <li style={{ marginBottom: "0.5rem" }}>
                    <strong>Data Processing:</strong> Uses race results (POSITION, FL_TIME, FL_KPH, STATUS), weather data (AIR_TEMP, RAIN), 
                    and track DNA classification. Position normalization converts finishing positions to 0-1 scale. All values are min-max scaled 
                    and clipped to [0,1] using robust clipping that handles NaN and infinity values.
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Skill Comparison"
        description="Top 5 drivers by each skill dimension"
      >
        <div className="grid-2">
          {comparisonData.slice(0, 4).map((skillData) => (
            <div key={skillData.skill} className="chart-wrapper">
              <div style={{ marginBottom: "0.75rem" }}>
                <h4 style={{ marginBottom: "0.25rem", color: "#fff", fontSize: "0.9rem" }}>
                  {skillData.skill}
                </h4>
                <div style={{ fontSize: "0.75rem", color: "rgba(255, 255, 255, 0.6)", display: "flex", gap: "1rem" }}>
                  <span>Avg: {(skillData.stats.average * 100).toFixed(1)}%</span>
                  <span>Max: {(skillData.stats.max * 100).toFixed(1)}%</span>
                  <span>Drivers: {skillData.stats.totalDrivers}</span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={skillData.topDrivers} margin={{ left: 5, right: 5, top: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="driver" 
                    stroke="#fff" 
                    fontSize={10}
                    tick={{ fill: "#fff" }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis 
                    domain={[0, 1]}
                    stroke="#fff"
                    fontSize={11}
                    tick={{ fill: "#fff" }}
                    tickFormatter={(value) => (value * 100).toFixed(0) + "%"}
                    label={{ value: "Skill Score (%)", angle: -90, position: "insideLeft", style: { fill: "#fff", fontSize: "0.75rem" } }}
                  />
                  <Tooltip 
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div style={{ 
                            background: "rgba(15, 17, 23, 0.98)", 
                            border: "1px solid rgba(255, 255, 255, 0.3)",
                            borderRadius: "8px",
                            color: "#fff",
                            padding: "0.75rem",
                            lineHeight: "1.8"
                          }}>
                            <div style={{ fontWeight: 600, marginBottom: "0.5rem", fontSize: "0.95rem" }}>
                              {data.driverName}
                            </div>
                            <div>Score: <strong>{(data.value * 100).toFixed(1)}%</strong></div>
                            <div>Rank: <strong>#{data.rank}</strong></div>
                            <div>Percentile: <strong>{data.percentile}%</strong></div>
                            <div>vs Average: <strong style={{ color: data.vsAverage > 0 ? "#00ffc6" : "#ff3358" }}>
                              {data.vsAverage > 0 ? "+" : ""}{data.vsAverage}%
                            </strong></div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar 
                    dataKey="value" 
                    fill="rgba(255, 51, 88, 0.7)"
                    stroke="rgba(255, 51, 88, 1)"
                    strokeWidth={1}
                  >
                    <LabelList 
                      dataKey="value" 
                      position="top" 
                      formatter={(value: number) => `${(value * 100).toFixed(0)}%`}
                      style={{ fill: "#fff", fontSize: "9px", fontWeight: 500 }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
        <div className="grid-2" style={{ marginTop: "1rem" }}>
          {comparisonData.slice(4).map((skillData) => (
            <div key={skillData.skill} className="chart-wrapper">
              <div style={{ marginBottom: "0.75rem" }}>
                <h4 style={{ marginBottom: "0.25rem", color: "#fff", fontSize: "0.9rem" }}>
                  {skillData.skill}
                </h4>
                <div style={{ fontSize: "0.75rem", color: "rgba(255, 255, 255, 0.6)", display: "flex", gap: "1rem" }}>
                  <span>Avg: {(skillData.stats.average * 100).toFixed(1)}%</span>
                  <span>Max: {(skillData.stats.max * 100).toFixed(1)}%</span>
                  <span>Drivers: {skillData.stats.totalDrivers}</span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={skillData.topDrivers} margin={{ left: 5, right: 5, top: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="driver" 
                    stroke="#fff" 
                    fontSize={10}
                    tick={{ fill: "#fff" }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis 
                    domain={[0, 1]}
                    stroke="#fff"
                    fontSize={11}
                    tick={{ fill: "#fff" }}
                    tickFormatter={(value) => (value * 100).toFixed(0) + "%"}
                    label={{ value: "Skill Score (%)", angle: -90, position: "insideLeft", style: { fill: "#fff", fontSize: "0.75rem" } }}
                  />
                  <Tooltip 
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div style={{ 
                            background: "rgba(15, 17, 23, 0.98)", 
                            border: "1px solid rgba(255, 255, 255, 0.3)",
                            borderRadius: "8px",
                            color: "#fff",
                            padding: "0.75rem",
                            lineHeight: "1.8"
                          }}>
                            <div style={{ fontWeight: 600, marginBottom: "0.5rem", fontSize: "0.95rem" }}>
                              {data.driverName}
                            </div>
                            <div>Score: <strong>{(data.value * 100).toFixed(1)}%</strong></div>
                            <div>Rank: <strong>#{data.rank}</strong></div>
                            <div>Percentile: <strong>{data.percentile}%</strong></div>
                            <div>vs Average: <strong style={{ color: data.vsAverage > 0 ? "#00ffc6" : "#ff3358" }}>
                              {data.vsAverage > 0 ? "+" : ""}{data.vsAverage}%
                            </strong></div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar 
                    dataKey="value" 
                    fill="rgba(255, 51, 88, 0.7)"
                    stroke="rgba(255, 51, 88, 1)"
                    strokeWidth={1}
                  >
                    <LabelList 
                      dataKey="value" 
                      position="top" 
                      formatter={(value: number) => `${(value * 100).toFixed(0)}%`}
                      style={{ fill: "#fff", fontSize: "9px", fontWeight: 500 }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

