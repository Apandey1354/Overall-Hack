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
} from "recharts";

interface Props {
  embeddings: DriverEmbedding[];
  selectedDriverNumber: number;
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

export function DriverSkillsEmbedding({ embeddings, selectedDriverNumber }: Props) {
  const [showTechnicalInfo, setShowTechnicalInfo] = useState(false);

  const selected = useMemo(
    () => embeddings.find((emb) => emb.driver_number === selectedDriverNumber) ?? embeddings[0],
    [embeddings, selectedDriverNumber]
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
      const sorted = [...embeddings]
        .sort((a, b) => {
          const aVal = a.skill_vector?.[skillIdx] ?? 0;
          const bVal = b.skill_vector?.[skillIdx] ?? 0;
          return bVal - aVal;
        })
        .slice(0, 5);
      
      return {
        skill: label,
        topDrivers: sorted.map((emb) => ({
          driver: `#${emb.driver_number}`,
          value: emb.skill_vector?.[skillIdx] ?? 0,
        })),
      };
    });
    
    return topBySkill;
  }, [embeddings]);

  if (!embeddings.length || !selected) {
    return null;
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title="Driver Skill Embeddings"
        description="8-dimensional skill vectors for the selected driver"
      >
        <p className="tab-description">
          Driver embeddings are 8-dimensional skill vectors created from historical performance data.
          Each dimension represents a different aspect of driver capability, normalized to 0-1 scale.
        </p>

        {/* Technical Information Section */}
        <div style={{ marginBottom: "1.5rem" }}>
          <button
            onClick={() => setShowTechnicalInfo(!showTechnicalInfo)}
            className="ghost-button"
            style={{ width: "100%", marginBottom: "0.5rem" }}
          >
            {showTechnicalInfo ? "🔽 Hide" : "▶️ Show"} Technical Details: How Driver Embeddings Are Calculated
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
        
        <div style={{ marginBottom: "1.5rem" }}>
          <table className="data-table" style={{ width: "100%" }}>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Driver</strong></td>
                <td>#{selected.driver_number} {selected.driver_name}</td>
                <td><strong>Weather Adaptability</strong></td>
                <td>{(selected.weather_adaptability * 100).toFixed(1)}%</td>
              </tr>
              <tr>
                <td><strong>Technical Proficiency</strong></td>
                <td>{(selected.technical_proficiency * 100).toFixed(1)}%</td>
                <td><strong>Best Track Type</strong></td>
                <td>{selected.best_track_type || "N/A"}</td>
              </tr>
              <tr>
                <td><strong>High-Speed Proficiency</strong></td>
                <td>{(selected.high_speed_proficiency * 100).toFixed(1)}%</td>
                <td><strong>Strengths</strong></td>
                <td>{selected.strengths || "N/A"}</td>
              </tr>
              <tr>
                <td><strong>Consistency Score</strong></td>
                <td>{(selected.consistency_score * 100).toFixed(1)}%</td>
                <td></td>
                <td></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="chart-wrapper">
          <h4 style={{ marginBottom: "1rem", color: "#fff", fontSize: "1rem" }}>
            Skill Vector Radar Chart
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.2)" />
              <PolarAngleAxis dataKey="skill" stroke="#fff" fontSize={12} />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 1]} 
                stroke="rgba(255,255,255,0.4)"
                tickFormatter={(value) => value.toFixed(1)}
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
      </SectionCard>

      <SectionCard
        title="Skill Comparison"
        description="Top 5 drivers by each skill dimension"
      >
        {/* Technical Information Section for Skill Comparison */}
        <div style={{ marginBottom: "1.5rem" }}>
          <button
            onClick={() => setShowTechnicalInfo(!showTechnicalInfo)}
            className="ghost-button"
            style={{ width: "100%", marginBottom: "0.5rem" }}
          >
            {showTechnicalInfo ? "🔽 Hide" : "▶️ Show"} Technical Details: How Skill Comparison Works
          </button>
          
          {showTechnicalInfo && (
            <div
              style={{
                background: "rgba(255, 51, 88, 0.1)",
                border: "1px solid rgba(255, 51, 88, 0.3)",
                borderRadius: "16px",
                padding: "1.25rem",
                marginTop: "0.5rem",
              }}
            >
              <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", fontWeight: 600, color: "#ff3358" }}>
                🔬 Skill Comparison Calculation Algorithm
              </h3>
              
              <div style={{ marginBottom: "1.5rem" }}>
                <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                  Comparison Methodology:
                </h4>
                <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                  <li><strong>Skill Dimension Extraction:</strong> For each of the 8 skill dimensions (Technical, High-Speed, Consistency, Weather, Tech-Track, Speed-Track, Balanced-Track, Finish-Rate), extracts the corresponding value from each driver's skill_vector array</li>
                  <li><strong>Sorting Algorithm:</strong> Sorts all drivers by their score in each skill dimension in descending order (highest to lowest)</li>
                  <li><strong>Top 5 Selection:</strong> Selects the top 5 drivers for each skill dimension based on their skill_vector value at that dimension's index</li>
                  <li><strong>Data Structure:</strong> Creates a comparison data structure with:
                    <ul style={{ marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                      <li>Skill name (e.g., "Technical", "High-Speed")</li>
                      <li>Array of top 5 drivers with their driver number and skill value</li>
                    </ul>
                  </li>
                  <li><strong>Visualization:</strong> Displays each skill dimension as a separate bar chart showing the top 5 drivers ranked by that specific skill</li>
                </ol>
              </div>

              <div style={{ marginBottom: "1.5rem" }}>
                <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                  Skill Vector Index Mapping:
                </h4>
                <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                  <li><strong>Index 0:</strong> Technical Proficiency</li>
                  <li><strong>Index 1:</strong> High-Speed Proficiency</li>
                  <li><strong>Index 2:</strong> Consistency Score</li>
                  <li><strong>Index 3:</strong> Weather Adaptability</li>
                  <li><strong>Index 4:</strong> Tech-Track Performance</li>
                  <li><strong>Index 5:</strong> Speed-Track Performance</li>
                  <li><strong>Index 6:</strong> Balanced-Track Performance</li>
                  <li><strong>Index 7:</strong> Finish Rate</li>
                </ul>
              </div>

              <div>
                <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                  Chart Interpretation:
                </h4>
                <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                  <li><strong>Bar Height:</strong> Represents the normalized skill score (0-1 scale, displayed as 0-100%)</li>
                  <li><strong>X-Axis:</strong> Shows driver numbers (#1, #2, etc.) of the top 5 performers</li>
                  <li><strong>Y-Axis:</strong> Skill score from 0 to 1.0 (0% to 100%)</li>
                  <li><strong>Color Coding:</strong> Pink gradient bars (rgba(255, 51, 88)) for visual consistency</li>
                  <li><strong>Use Case:</strong> Helps identify which drivers excel in specific skill areas, useful for team strategy and driver development</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        <div className="grid-2">
          {comparisonData.slice(0, 4).map((skillData) => (
            <div key={skillData.skill} className="chart-wrapper">
              <h4 style={{ marginBottom: "0.75rem", color: "#fff", fontSize: "0.9rem" }}>
                {skillData.skill}
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={skillData.topDrivers} margin={{ left: 5, right: 5, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="driver" 
                    stroke="#fff" 
                    fontSize={11}
                    tick={{ fill: "#fff" }}
                  />
                  <YAxis 
                    domain={[0, 1]}
                    stroke="#fff"
                    fontSize={11}
                    tick={{ fill: "#fff" }}
                    tickFormatter={(value) => value.toFixed(1)}
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
                  <Bar 
                    dataKey="value" 
                    fill="rgba(255, 51, 88, 0.7)"
                    stroke="rgba(255, 51, 88, 1)"
                    strokeWidth={1}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
        <div className="grid-2" style={{ marginTop: "1rem" }}>
          {comparisonData.slice(4).map((skillData) => (
            <div key={skillData.skill} className="chart-wrapper">
              <h4 style={{ marginBottom: "0.75rem", color: "#fff", fontSize: "0.9rem" }}>
                {skillData.skill}
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={skillData.topDrivers} margin={{ left: 5, right: 5, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="driver" 
                    stroke="#fff" 
                    fontSize={11}
                    tick={{ fill: "#fff" }}
                  />
                  <YAxis 
                    domain={[0, 1]}
                    stroke="#fff"
                    fontSize={11}
                    tick={{ fill: "#fff" }}
                    tickFormatter={(value) => value.toFixed(1)}
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
                  <Bar 
                    dataKey="value" 
                    fill="rgba(255, 51, 88, 0.7)"
                    stroke="rgba(255, 51, 88, 1)"
                    strokeWidth={1}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

