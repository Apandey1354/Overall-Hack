import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from "recharts";
import { BoxPlotStat, EDAData, HistogramBin } from "../types";
import { SectionCard } from "./SectionCard";

const COLORS = ["#ff7043", "#4fc3f7", "#9575cd", "#26a69a", "#ffca28", "#ef5350"];

interface Props {
  data: EDAData | null;
}

export function EDAExplorer({ data }: Props) {
  if (!data) {
    return (
      <SectionCard title="Data Health Check">
        <p>Loading exploratory analytics …</p>
      </SectionCard>
    );
  }

  const completeness = data.completeness;
  const results = data.results;
  const weatherMetrics = Object.values(data.weather.metrics ?? {});

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Data Completeness Section */}
      <div>
        <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
          📊 Data Completeness Summary
        </h4>
        <div style={{ 
          background: "rgba(255, 255, 255, 0.03)", 
          borderRadius: "12px", 
          padding: "1rem",
          border: "1px solid rgba(255, 255, 255, 0.1)"
        }}>
          <div className="grid-2" style={{ gap: "1.5rem", marginBottom: "1rem" }}>
            <div style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.05)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Total Checks</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#fff" }}>{completeness.summary.total_checks}</div>
            </div>
            <div style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.05)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Files Found</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#00ffc6" }}>{completeness.summary.files_found}</div>
            </div>
            <div style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.05)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Files Missing</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#ff3358" }}>{completeness.summary.files_missing}</div>
            </div>
            <div style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.05)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", marginBottom: "0.25rem" }}>Overall Coverage</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#fff" }}>
                {completeness.summary.overall_pct.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Venue Breakdown Table */}
      <div>
        <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
          🏁 Venue Breakdown
        </h4>
        <div style={{ 
          background: "rgba(255, 255, 255, 0.03)", 
          borderRadius: "12px", 
          padding: "1rem",
          border: "1px solid rgba(255, 255, 255, 0.1)"
        }}>
          <table className="data-table" style={{ width: "100%", fontSize: "0.95rem" }}>
            <thead>
              <tr>
                <th style={{ padding: "0.75rem", fontSize: "0.9rem" }}>Venue</th>
                <th style={{ padding: "0.75rem", fontSize: "0.9rem" }}>Files Found</th>
                <th style={{ padding: "0.75rem", fontSize: "0.9rem" }}>Total Files</th>
                <th style={{ padding: "0.75rem", fontSize: "0.9rem" }}>Coverage %</th>
              </tr>
            </thead>
            <tbody>
              {completeness.breakdown.map((row) => (
                <tr key={row.venue}>
                  <td style={{ padding: "0.75rem", fontWeight: 600 }}><strong>{row.venue}</strong></td>
                  <td style={{ padding: "0.75rem", color: "#00ffc6", fontWeight: 600 }}>{row.found}</td>
                  <td style={{ padding: "0.75rem" }}>{row.total}</td>
                  <td style={{ padding: "0.75rem", fontWeight: 600 }}>
                    <span style={{ color: row.pct >= 90 ? "#00ffc6" : row.pct >= 70 ? "#ffca28" : "#ff3358" }}>
                      {row.pct.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Completeness Heatmap */}
      <div>
        <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
          🔍 File Completeness Heatmap
        </h4>
        <div style={{ 
          background: "rgba(255, 255, 255, 0.03)", 
          borderRadius: "12px", 
          padding: "1rem",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          overflowX: "auto"
        }}>
          <HeatmapGrid heatmap={completeness.heatmap} />
        </div>
      </div>

      {/* Race Result Distributions */}
      <div>
        <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
          ⏱️ Race Result Distributions
        </h4>
        <div style={{ 
          background: "rgba(255, 255, 255, 0.03)", 
          borderRadius: "12px", 
          padding: "1rem",
          border: "1px solid rgba(255, 255, 255, 0.1)"
        }}>
          <div style={{ marginBottom: "1.5rem" }}>
            <h5 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#00ffc6" }}>
              Lap Count Distribution by Venue
            </h5>
            <BoxPlotList data={results.lap_boxplot} />
          </div>
          <div style={{ marginBottom: "1.5rem" }}>
            <HistogramChart
              title="Fastest Lap Histogram (FL_TIME_seconds)"
              data={results.fastest_lap_histogram.bins}
              categories={results.fastest_lap_histogram.categories}
            />
          </div>
          <div>
            <HistogramChart
              title="Best Lap Histogram (BESTLAP_1_seconds)"
              data={results.best_lap_histogram.bins}
              categories={results.best_lap_histogram.categories}
            />
          </div>
        </div>
      </div>

      {/* Weather Variability */}
      <div>
        <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
          🌦️ Weather Variability Analysis
        </h4>
        <div style={{ 
          background: "rgba(255, 255, 255, 0.03)", 
          borderRadius: "12px", 
          padding: "1rem",
          border: "1px solid rgba(255, 255, 255, 0.1)"
        }}>
          <div className="grid-2" style={{ gap: "1.5rem" }}>
            {weatherMetrics.map((metric) => (
              <div key={metric.title} style={{
                padding: "1rem",
                background: "rgba(255, 255, 255, 0.05)",
                borderRadius: "8px",
                border: "1px solid rgba(255, 255, 255, 0.1)"
              }}>
                <h5 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                  {metric.title}
                </h5>
                <BoxPlotList data={metric.boxplot} compact />
              </div>
            ))}
          </div>
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
            The Data Health Check uses the <code style={{ 
              background: "rgba(255, 255, 255, 0.1)", 
              padding: "0.2rem 0.4rem", 
              borderRadius: "4px",
              fontSize: "0.85rem"
            }}>EDAExplorer</code> component to validate and visualize data completeness:
          </p>
          <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
            <li style={{ marginBottom: "0.5rem" }}>
              <strong>Data Completeness Validation:</strong> Uses <code style={{ 
                background: "rgba(255, 255, 255, 0.1)", 
                padding: "0.2rem 0.4rem", 
                borderRadius: "4px",
                fontSize: "0.85rem"
              }}>validate_data_completeness()</code> to check for required files (results, weather, telemetry, analysis) 
              per venue. The heatmap shows green cells for existing files and gray for missing ones.
            </li>
            <li style={{ marginBottom: "0.5rem" }}>
              <strong>Race Result Analysis:</strong> Computes quartiles (Q1, median, Q3) per venue against the `LAPS` column 
              using pandas describe(). Histograms use 15 bins between global min/max lap times, matching seaborn/Plotly 
              visualizations from the notebook.
            </li>
            <li style={{ marginBottom: "0.5rem" }}>
              <strong>Weather Metrics:</strong> Extracts metrics directly from <code style={{ 
                background: "rgba(255, 255, 255, 0.1)", 
                padding: "0.2rem 0.4rem", 
                borderRadius: "4px",
                fontSize: "0.85rem"
              }}>load_weather_data</code>. Box plots show min/max whiskers and interquartile range per venue, 
              matching original Plotly subplots.
            </li>
            <li style={{ marginBottom: "0.5rem" }}>
              <strong>Data Processing:</strong> All calculations are performed server-side using pandas DataFrames. 
              The frontend receives pre-computed statistics and visualizations are rendered using Recharts. 
              Box plots use scaled positioning based on global min/max values for consistent visualization.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function HeatmapGrid({ heatmap }: { heatmap: EDAData["completeness"]["heatmap"] }) {
  if (!heatmap.cells.length) {
    return <p style={{ color: "rgba(255, 255, 255, 0.7)" }}>No completeness heatmap data available.</p>;
  }

  const cellLookup = new Map<string, boolean>();
  heatmap.cells.forEach((cell) => {
    cellLookup.set(`${cell.venue}:${cell.file_type}`, cell.found);
  });

  return (
    <div className="heatmap-container" style={{ overflowX: "auto" }}>
      <table className="heatmap-table" style={{ 
        width: "100%", 
        fontSize: "0.9rem",
        borderCollapse: "separate",
        borderSpacing: "2px"
      }}>
        <thead>
          <tr>
            <th style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.1)",
              fontWeight: 600,
              textAlign: "left",
              position: "sticky",
              left: 0,
              zIndex: 10
            }}>
              Venue \\ File Type
            </th>
            {heatmap.file_types.map((file) => (
              <th key={file} style={{ 
                padding: "0.75rem", 
                background: "rgba(255, 255, 255, 0.1)",
                fontWeight: 600,
                minWidth: "100px"
              }}>
                {file}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {heatmap.venues.map((venue) => (
            <tr key={venue}>
              <td style={{ 
                padding: "0.75rem", 
                fontWeight: 600,
                background: "rgba(255, 255, 255, 0.05)",
                position: "sticky",
                left: 0,
                zIndex: 5
              }}>
                {venue}
              </td>
              {heatmap.file_types.map((file) => {
                const found = cellLookup.get(`${venue}:${file}`) ?? false;
                return (
                  <td key={`${venue}-${file}`} style={{ 
                    padding: "0.5rem",
                    textAlign: "center"
                  }}>
                    <span 
                      className={["heatmap-cell", found ? "found" : "missing"].join(" ")} 
                      style={{
                        display: "inline-block",
                        width: "32px",
                        height: "32px",
                        borderRadius: "4px",
                        border: found ? "2px solid #00ffc6" : "2px solid rgba(255, 255, 255, 0.2)"
                      }}
                      title={found ? `✓ ${file} found for ${venue}` : `✗ ${file} missing for ${venue}`}
                    />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface BoxPlotListProps {
  data: BoxPlotStat[];
  compact?: boolean;
}

function BoxPlotList({ data, compact }: BoxPlotListProps) {
  const [minValue, maxValue] = useMemo(() => {
    if (!data.length) {
      return [0, 1];
    }
    const mins = data.map((row) => row.min);
    const maxs = data.map((row) => row.max);
    const globalMin = Math.min(...mins);
    const globalMax = Math.max(...maxs);
    return [globalMin, globalMax === globalMin ? globalMin + 1 : globalMax];
  }, [data]);

  const scale = (value: number) => ((value - minValue) / (maxValue - minValue)) * 100;

  if (!data.length) {
    return <p>No numerical data found for this chart.</p>;
  }

  return (
    <div className="boxplot-list">
      {data.map((row) => (
        <div key={row.label} className="boxplot-row">
          <span className="boxplot-label">{row.label}</span>
          <div className={["boxplot-track", compact ? "compact" : ""].join(" ")}>
            <span className="whisker" style={{ left: `${scale(row.min)}%` }} />
            <span className="whisker" style={{ left: `${scale(row.max)}%` }} />
            <span
              className="boxplot-box"
              style={{
                left: `${scale(row.q1)}%`,
                width: `${scale(row.q3) - scale(row.q1)}%`,
              }}
            />
            <span className="median" style={{ left: `${scale(row.median)}%` }} />
          </div>
          <span className="boxplot-count">{row.count} laps</span>
        </div>
      ))}
    </div>
  );
}

interface HistogramChartProps {
  title: string;
  data: HistogramBin[];
  categories: string[];
}

function HistogramChart({ title, data, categories }: HistogramChartProps) {
  if (!data.length) {
    return (
      <div className="chart-wrapper">
        <h5 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>{title}</h5>
        <p>No data available.</p>
      </div>
    );
  }
  return (
    <div className="chart-wrapper">
      <h5 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>{title}</h5>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ left: 20, right: 20, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis 
            dataKey="bin" 
            stroke="#fff" 
            fontSize={12}
            tick={{ fill: "#fff" }}
          />
          <YAxis 
            stroke="#fff"
            fontSize={12}
            tick={{ fill: "#fff" }}
          />
          <Tooltip 
            contentStyle={{ 
              background: "rgba(15, 17, 23, 0.95)", 
              border: "1px solid rgba(255, 255, 255, 0.2)",
              borderRadius: "8px",
              color: "#fff"
            }}
          />
          <Legend 
            wrapperStyle={{ color: "#fff" }}
          />
          {categories.map((category, index) => (
            <Bar
              key={category}
              dataKey={category}
              stackId="hist"
              fill={COLORS[index % COLORS.length]}
              maxBarSize={40}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}


