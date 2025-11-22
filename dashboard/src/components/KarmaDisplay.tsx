import { useEffect, useState, useRef } from "react";
import { API_BASE_URL } from "../config";
import { SectionCard } from "./SectionCard";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface KarmaDataPoint {
  lap: number;
  component: string;
  instant_score: number;
  karma_score: number;
}

interface KarmaResponse {
  vehicle_id: string;
  latest_scores: Record<string, { score: number; lap: number }>;
  time_series: KarmaDataPoint[];
}

interface ComponentInfo {
  name: string;
  description: string;
  features: string[];
}

export function KarmaDisplay() {
  const [vehicles, setVehicles] = useState<string[]>([]);
  const [selectedVehicle, setSelectedVehicle] = useState<string>("");
  const [karmaData, setKarmaData] = useState<KarmaResponse | null>(null);
  const [components, setComponents] = useState<ComponentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Default components (fallback if API fails)
      const defaultComponents = [
        { name: "engine", description: "RPM + sustained speed stress", features: ["speed_mean", "nmot_mean"] },
        { name: "gearbox", description: "Gear usage and longitudinal jolts", features: ["gear_mean", "accx_can_std"] },
        { name: "brakes", description: "Brake pressure spikes front/rear", features: ["pbrake_f_max", "pbrake_r_max"] },
        { name: "tires", description: "Cornering + abrasion load", features: ["speed_mean", "Steering_Angle_std"] },
      ];
      
      try {
        const [vehiclesRes, componentsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/karma/vehicles`),
          fetch(`${API_BASE_URL}/karma/components`),
        ]);

        // Handle vehicles
        let vehiclesData = { vehicles: [] };
        if (vehiclesRes.ok) {
          vehiclesData = await vehiclesRes.json();
          setVehicles(vehiclesData.vehicles || []);
          
          // Select first vehicle if available
          if (vehiclesData.vehicles && vehiclesData.vehicles.length > 0) {
            setSelectedVehicle(vehiclesData.vehicles[0]);
          }
        } else {
          console.warn(`Failed to load vehicles: ${vehiclesRes.status} ${vehiclesRes.statusText}`);
          setVehicles([]);
        }

        // Handle components
        if (componentsRes.ok) {
          const componentsData = await componentsRes.json();
          setComponents(componentsData.components || defaultComponents);
        } else {
          console.warn(`Failed to load components: ${componentsRes.status} ${componentsRes.statusText}`);
          setComponents(defaultComponents);
        }
      } catch (fetchError) {
        // Network error or API server not running
        console.error("API fetch error:", fetchError);
        setVehicles([]);
        setComponents(defaultComponents);
        
        if (fetchError instanceof TypeError && fetchError.message.includes("fetch")) {
          setError("Cannot connect to API server. Make sure the backend is running on http://localhost:8000");
        } else {
          setError(`Failed to load data: ${fetchError instanceof Error ? fetchError.message : "Unknown error"}`);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      console.error("Error loading karma data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      setUploadMessage(null);
      setError(null);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/karma/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Upload result:", result);
      
      setUploadMessage({
        type: "success",
        text: `File processed successfully! Found ${result.vehicles?.length || 0} vehicle(s) with ${result.rows_processed || 0} laps.`,
      });

      // Reload vehicles list
      await loadData();

      // Select first vehicle if available
      if (result.vehicles && result.vehicles.length > 0) {
        console.log("Selecting vehicle:", result.vehicles[0]);
        setSelectedVehicle(result.vehicles[0]);
      } else {
        console.warn("No vehicles found in upload result");
        setError("File processed but no vehicles found. Check file format.");
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Upload failed";
      setUploadMessage({ type: "error", text: errorMsg });
      setError(errorMsg);
    } finally {
      setUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  useEffect(() => {
    if (!selectedVehicle) return;

    setLoading(true);
    fetch(`${API_BASE_URL}/karma/${selectedVehicle}`)
      .then((res) => {
        if (!res.ok) {
          return res.json().then(errData => {
            throw new Error(errData.detail || `Failed to load karma: ${res.statusText}`);
          });
        }
        return res.json();
      })
      .then((data) => {
        console.log("Karma data received:", data);
        if (!data.time_series || data.time_series.length === 0) {
          console.warn("No karma time series data received", data);
          if (data.debug_info) {
            console.warn("Debug info:", data.debug_info);
          }
        }
        setKarmaData(data);
        setError(null);
      })
      .catch((err) => {
        const errorMsg = err instanceof Error ? err.message : "Failed to load karma data";
        setError(errorMsg);
        console.error("Error loading karma:", err);
      })
      .finally(() => setLoading(false));
  }, [selectedVehicle]);

  // Prepare chart data
  const chartData = karmaData?.time_series?.reduce((acc, point) => {
    if (!point || typeof point.lap === 'undefined' || !point.component) {
      return acc;
    }
    const lapKey = point.lap;
    if (!acc[lapKey]) {
      acc[lapKey] = { lap: lapKey };
    }
    acc[lapKey][point.component] = point.karma_score;
    return acc;
  }, {} as Record<number, Record<string, number>>) || {};

  const chartDataArray = Object.values(chartData).sort(
    (a, b) => (a.lap as number) - (b.lap as number)
  );

  const getTrend = (component: string): "up" | "down" | "stable" => {
    if (!karmaData?.time_series) return "stable";
    const componentData = karmaData.time_series
      .filter((item) => item.component === component)
      .sort((a, b) => a.lap - b.lap);
    if (componentData.length < 2) return "stable";
    const recent = componentData.slice(-3);
    if (recent.length < 2) return "stable";
    const trend = recent[recent.length - 1].karma_score - recent[0].karma_score;
    return trend > 0.01 ? "up" : trend < -0.01 ? "down" : "stable";
  };

  const getColor = (component: string): string => {
    const colors: Record<string, string> = {
      engine: "#ff3358",
      gearbox: "#00ffc6",
      brakes: "#ff8038",
      tires: "#ffd700",
    };
    return colors[component] || "#ffffff";
  };

  if (loading && !karmaData) {
    return (
      <SectionCard title="Mechanical Karma Detector">
        <p>Loading karma data...</p>
      </SectionCard>
    );
  }

  if (error) {
    return (
      <SectionCard title="Mechanical Karma Detector">
        <div className="space-y-4">
          <p style={{ color: "#ff3358" }}>Error: {error}</p>
          <div style={{ fontSize: "0.9em", color: "rgba(255,255,255,0.7)" }} className="space-y-2">
            <p><strong>Troubleshooting:</strong></p>
            <ul style={{ listStyle: "disc", paddingLeft: "1.5rem", marginTop: "0.5rem" }}>
              <li>Make sure the API server is running: <code style={{ background: "rgba(255,255,255,0.1)", padding: "0.2rem 0.4rem", borderRadius: "4px" }}>python -m uvicorn src.api.dashboard_api:app --reload --port 8000</code></li>
              <li>Check that the API is accessible at: <code style={{ background: "rgba(255,255,255,0.1)", padding: "0.2rem 0.4rem", borderRadius: "4px" }}>http://localhost:8000/api/status</code></li>
              <li>Upload telemetry data using the file upload below</li>
            </ul>
          </div>
        </div>
      </SectionCard>
    );
  }

  // Always show the main interface, even if no vehicles yet
  // The file upload will allow users to add data

  return (
    <div className="space-y-6">
      <SectionCard
        title="Mechanical Karma Detector"
        description="Track component health and wear over time. Higher scores indicate increased stress and degradation risk."
      >
        <div className="space-y-4">
          {/* File Upload Section */}
          <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
            <h3 className="text-lg font-semibold mb-3">Upload Telemetry Data</h3>
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.parquet"
                onChange={handleFileUpload}
                disabled={uploading}
                className="block w-full sm:w-auto text-sm text-white/70 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-[#00ffc6] file:text-[#0f1117] hover:file:bg-[#00e6b8] file:cursor-pointer disabled:opacity-50"
              />
              {uploading && (
                <span className="text-sm text-white/70">Processing...</span>
              )}
            </div>
            {uploadMessage && (
              <div
                className={`mt-3 p-3 rounded-lg text-sm ${
                  uploadMessage.type === "success"
                    ? "bg-[#00ffc6]/20 text-[#00ffc6] border border-[#00ffc6]/30"
                    : "bg-[#ff3358]/20 text-[#ff3358] border border-[#ff3358]/30"
                }`}
              >
                {uploadMessage.text}
              </div>
            )}
            <p className="mt-2 text-xs text-white/50">
              Upload a CSV file with telemetry data. The file should contain vehicle_id, lap, and telemetry parameters.
            </p>
          </div>

          {vehicles.length > 0 ? (
            <div>
              <label htmlFor="vehicle-select" className="block text-sm font-medium mb-2">
                Select Vehicle:
              </label>
              <select
                id="vehicle-select"
                value={selectedVehicle}
                onChange={(e) => setSelectedVehicle(e.target.value)}
                className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-[#00ffc6]"
              >
                {vehicles.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
              <p className="text-sm text-white/70 mb-2">
                No vehicles with karma data available yet.
              </p>
              <p className="text-xs text-white/50">
                Upload a telemetry CSV file above to process and visualize component health data.
              </p>
            </div>
          )}

          {loading && selectedVehicle && (
            <div className="p-4 bg-white/5 border border-white/10 rounded-lg text-center">
              <p className="text-sm text-white/70">Loading karma data for {selectedVehicle}...</p>
            </div>
          )}

          {!loading && karmaData && (!karmaData.latest_scores || Object.keys(karmaData.latest_scores).length === 0) && (
            <div className="p-4 bg-[#ff3358]/20 text-[#ff3358] border border-[#ff3358]/30 rounded-lg">
              <p className="font-semibold mb-2">No Karma Data Available</p>
              <p className="text-sm mb-2">
                The file was processed but no karma scores could be computed. This usually means:
              </p>
              <ul className="text-sm list-disc list-inside space-y-1 mb-2">
                <li>The file format doesn't match expected telemetry structure</li>
                <li>Required features (speed, RPM, gear, brakes, steering) are missing</li>
                <li>The data columns don't match the expected names</li>
              </ul>
              {karmaData.debug_info && (
                <div className="mt-3 p-3 bg-white/5 rounded text-xs">
                  <p className="font-semibold mb-1">Debug Info:</p>
                  <p>Rows: {karmaData.debug_info.vehicle_rows}</p>
                  <p>Available columns: {karmaData.debug_info.available_columns?.join(", ") || "none"}</p>
                  <p>Required features: {karmaData.debug_info.required_features?.join(", ") || "none"}</p>
                </div>
              )}
            </div>
          )}

          {!loading && karmaData && karmaData.latest_scores && Object.keys(karmaData.latest_scores).length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(karmaData.latest_scores).map(([component, data]) => {
                const trend = getTrend(component);
                const scorePercent = (data.score * 100).toFixed(1);
                const color = getColor(component);
                return (
                  <div
                    key={component}
                    className="p-4 bg-white/5 border border-white/10 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold capitalize">
                        {component}
                      </span>
                      {trend === "up" && <span className="text-[#ff3358]">↗</span>}
                      {trend === "down" && <span className="text-[#00ffc6]">↘</span>}
                      {trend === "stable" && <span className="text-white/50">→</span>}
                    </div>
                    <div className="text-2xl font-bold mb-2" style={{ color }}>
                      {scorePercent}%
                    </div>
                    <div className="w-full bg-white/10 rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{
                          width: `${scorePercent}%`,
                          backgroundColor: color,
                        }}
                      />
                    </div>
                    <div className="text-xs text-white/50 mt-1">
                      Lap {data.lap}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {chartDataArray.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-4">Karma Score Over Time</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={chartDataArray}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis
                    dataKey="lap"
                    stroke="rgba(255,255,255,0.7)"
                    label={{ value: "Lap", position: "insideBottom", offset: -5 }}
                  />
                  <YAxis
                    stroke="rgba(255,255,255,0.7)"
                    domain={[0, 1]}
                    label={{ value: "Karma Score", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(15, 17, 23, 0.95)",
                      border: "1px solid rgba(255,255,255,0.2)",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  {components.map((comp) => (
                    <Line
                      key={comp.name}
                      type="monotone"
                      dataKey={comp.name}
                      stroke={getColor(comp.name)}
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      name={comp.name.charAt(0).toUpperCase() + comp.name.slice(1)}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {components.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-4">Component Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {components.map((comp) => (
                  <div
                    key={comp.name}
                    className="p-4 bg-white/5 border border-white/10 rounded-lg"
                  >
                    <h4 className="font-semibold capitalize mb-2" style={{ color: getColor(comp.name) }}>
                      {comp.name}
                    </h4>
                    <p className="text-sm text-white/70 mb-2">{comp.description}</p>
                    <div className="text-xs text-white/50">
                      Features: {comp.features.join(", ")}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </SectionCard>
    </div>
  );
}

