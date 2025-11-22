import { useMemo, useState } from "react";
import { TrackSummary } from "../types";
import { SectionCard } from "./SectionCard";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

interface Props {
  tracks: TrackSummary[];
}

export function TrackDNAView({ tracks }: Props) {
  const [selectedId, setSelectedId] = useState<string>(() => (tracks[0]?.track_id ?? ""));
  const selected = useMemo(
    () => tracks.find((track) => track.track_id === selectedId) ?? tracks[0],
    [tracks, selectedId],
  );

  if (!tracks.length) {
    return (
      <SectionCard title="Track DNA Profiler">
        <p>No track DNA data found. Run `python scripts/build_dashboard_cache.py` first.</p>
      </SectionCard>
    );
  }

  return (
    <SectionCard
      title="Track DNA Profiler"
      description="Compare technical DNA across venues."
      action={
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <label style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", fontWeight: 500 }}>
            Select Track here
          </label>
          <select 
            value={selected?.track_id || ""} 
            onChange={(event) => {
              if (event.target.value) {
                setSelectedId(event.target.value);
              }
            }}
            style={{ position: "relative" }}
          >
            <option value="" disabled style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
              🔽 Select to change track...
            </option>
            {tracks.map((track) => (
              <option key={track.track_id} value={track.track_id}>
                {track.track_id}
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
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Venue</div>
                <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>{selected.venue}</div>
              </div>
              <div style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Race</div>
                <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>{selected.race}</div>
              </div>
              <div style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Track Type</div>
                <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#00ffc6" }}>{selected.cluster_label ?? "N/A"}</div>
              </div>
              <div style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Length</div>
                <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>{selected.track_length_km?.toFixed(2) ?? "?"} km</div>
              </div>
            </div>
          </div>

          {/* Track Characteristics Section */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              📊 Track Characteristics
            </h4>
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
                  <td><strong>Complexity Score</strong></td>
                  <td>{selected.complexity_score?.toFixed(3) ?? "?"}</td>
                  <td><strong>Sector Std (s)</strong></td>
                  <td>{selected.overall_sector_std?.toFixed(3) ?? "?"}</td>
                </tr>
                <tr>
                  <td><strong>Straight / Corner Ratio</strong></td>
                  <td>{selected.straight_corner_ratio?.toFixed(2) ?? "?"}</td>
                  <td><strong>Braking Zones</strong></td>
                  <td>{selected.braking_zones ?? "?"}</td>
                </tr>
                <tr>
                  <td><strong>Number of Sectors</strong></td>
                  <td>{selected.num_sectors ?? "?"}</td>
                  <td><strong>Top Speed (km/h)</strong></td>
                  <td>{selected.top_speed?.toFixed(1) ?? "?"}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Radar Chart Section */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              📈 Track Profile Visualization
            </h4>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart
                  data={[
                    { metric: "Complexity", value: selected.complexity_score ?? 0 },
                    { metric: "Sector Std", value: selected.overall_sector_std ?? 0 },
                    { metric: "Straight Ratio", value: selected.straight_corner_ratio ?? 0 },
                    { metric: "Num Sectors", value: Math.min((selected.num_sectors ?? 0) / 12, 1) },
                    { metric: "Length", value: Math.min((selected.track_length_km ?? 0) / 10, 1) },
                  ]}
                >
                  <PolarGrid stroke="rgba(255,255,255,0.2)" />
                  <PolarAngleAxis dataKey="metric" stroke="#fff" fontSize={12} />
                  <PolarRadiusAxis stroke="rgba(255,255,255,0.4)" />
                  <Tooltip 
                    formatter={(value: number) => value.toFixed(3)}
                    contentStyle={{ 
                      background: "rgba(15, 17, 23, 0.95)", 
                      border: "1px solid rgba(255, 255, 255, 0.2)",
                      borderRadius: "8px"
                    }}
                  />
                  <Radar dataKey="value" stroke="#ff3358" fill="rgba(255, 51, 88, 0.4)" strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
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
                Track DNA metrics are extracted using the <code style={{ 
                  background: "rgba(255, 255, 255, 0.1)", 
                  padding: "0.2rem 0.4rem", 
                  borderRadius: "4px",
                  fontSize: "0.85rem"
                }}>TrackDNAExtractor</code> class, which processes raw telemetry and timing data:
              </p>
              <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Technical Complexity:</strong> Calculated from sector time variance (standard deviation across sectors), 
                  braking zone density (estimated from brake pressure telemetry), and normalized corner speeds. Higher variance 
                  indicates more technical complexity.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Speed Profile:</strong> Extracted from speed distribution histograms. Top speeds, mean/median speeds, 
                  and percentiles are computed. Straight/corner ratio is derived by analyzing speed distribution patterns to 
                  identify fast (straight) vs. slow (corner) sectors.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Physical Characteristics:</strong> Track length is estimated from average speed and lap time data, 
                  or uses known track lengths when available. Number of sectors and intermediate timing points are extracted 
                  directly from timing data.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Track Clustering:</strong> Uses <code style={{ 
                    background: "rgba(255, 255, 255, 0.1)", 
                    padding: "0.2rem 0.4rem", 
                    borderRadius: "4px",
                    fontSize: "0.85rem"
                  }}>TrackClusterer</code> with KMeans clustering (scaled feature vectors via StandardScaler) to classify tracks 
                  into categories (e.g., technical, speed-focused, balanced). Features are normalized before clustering to ensure 
                  equal weighting across different metric scales.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Data Processing:</strong> Telemetry data is cached per venue/race to avoid redundant reads. Multiple 
                  feature extraction methods (sector analysis, speed histograms, braking zone detection) are combined into a 
                  unified track DNA profile.
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </SectionCard>
  );
}

