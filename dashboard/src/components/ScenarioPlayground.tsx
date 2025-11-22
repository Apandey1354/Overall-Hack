import { useMemo, useState } from "react";
import { ChampionshipState, ImpactReport, DriverEmbedding } from "../types";
import { SectionCard } from "./SectionCard";
import { API_BASE_URL } from "../config";
import { Modal } from "./Modal";
import { BackendSetupGuide } from "./BackendSetupGuide";
import { isBackendConnectionError } from "../utils/errorDetection";

interface Props {
  impactReports: ImpactReport[];
  championshipState: ChampionshipState | null;
  driverEmbeddings?: DriverEmbedding[];
}

interface Adjustment {
  event_order: number;
  driver_number: number;
  new_position?: number;
  position_delta?: number;
}

export function ScenarioPlayground({ impactReports, championshipState, driverEmbeddings = [] }: Props) {
  const [adjustments, setAdjustments] = useState<Adjustment[]>([]);
  const [name, setName] = useState("Custom Scenario");
  const [description, setDescription] = useState("User-defined tweaks to explore butterfly effects.");
  const [result, setResult] = useState<any>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [isBackendError, setIsBackendError] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);
  const [showTechnicalInfo, setShowTechnicalInfo] = useState(false);
  const [aiRecommendation, setAiRecommendation] = useState<string>();
  const [loadingAI, setLoadingAI] = useState(false);

  // Extract available event orders with track names from race results (all events, not just top 5 from impact reports)
  const availableEvents = useMemo(() => {
    const eventMap = new Map<number, { track_id: string; venue?: string }>();
    
    // Always get ALL events from race_results (not just top 5 from impact reports)
    if (championshipState?.race_results) {
      championshipState.race_results.forEach((result: any) => {
        if (result.event_order && result.track_id && !eventMap.has(result.event_order)) {
          eventMap.set(result.event_order, { track_id: result.track_id });
        }
      });
    }
    
    // Fallback to impact reports if race_results not available
    if (eventMap.size === 0 && impactReports) {
      impactReports.forEach((report) => {
        if (report.event_order && report.track_id) {
          eventMap.set(report.event_order, { track_id: report.track_id });
        }
      });
    }
    
    // Convert to array and sort by event_order
    return Array.from(eventMap.entries())
      .map(([order, data]) => ({ event_order: order, track_id: data.track_id }))
      .sort((a, b) => a.event_order - b.event_order);
  }, [impactReports, championshipState]);
  
  // Keep availableEventOrders for backward compatibility
  const availableEventOrders = useMemo(() => {
    return availableEvents.map(e => e.event_order);
  }, [availableEvents]);

  // Extract top 6 drivers from championship standings, ranked by skill
  const availableDrivers = useMemo(() => {
    if (!championshipState?.final_standings) return [];
    
    // Get top 6 drivers by championship position
    const top6ByStandings = championshipState.final_standings
      .slice(0, 6)
      .map((driver: any) => ({
        number: driver.driver_number,
        name: driver.driver_name || `Driver #${driver.driver_number}`,
        position: driver.position || driver.rank || 0,
        points: driver.season_points || 0,
      }));
    
    // If we have driver embeddings, rank by overall skill score
    if (driverEmbeddings && driverEmbeddings.length > 0) {
      // Calculate overall skill score for each driver (average of all 8 skill dimensions)
      const driversWithSkills = top6ByStandings.map(driver => {
        const embedding = driverEmbeddings.find(emb => emb.driver_number === driver.number);
        if (embedding && embedding.skill_vector && embedding.skill_vector.length >= 8) {
          // Calculate average skill score
          const avgSkill = embedding.skill_vector.reduce((sum, val) => sum + val, 0) / embedding.skill_vector.length;
          return {
            ...driver,
            skillScore: avgSkill,
          };
        }
        return {
          ...driver,
          skillScore: 0, // Default if no embedding found
        };
      });
      
      // Sort by skill score descending, then by championship position
      driversWithSkills.sort((a, b) => {
        if (Math.abs(a.skillScore - b.skillScore) > 0.001) {
          return b.skillScore - a.skillScore;
        }
        return a.position - b.position;
      });
      
      return driversWithSkills;
    }
    
    // If no embeddings, just return top 6 by championship position
    return top6ByStandings;
  }, [championshipState, driverEmbeddings]);

  // Generate position options (1-20)
  const positionOptions = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => i + 1);
  }, []);

  // Generate position delta options (-19 to +19)
  const positionDeltaOptions = useMemo(() => {
    return Array.from({ length: 39 }, (_, i) => i - 19);
  }, []);

  const topEvents = useMemo(() => impactReports?.slice(0, 5) ?? [], [impactReports]);

  // Helper function to get original position for a driver in an event
  const getOriginalPosition = (eventOrder: number, driverNumber: number): number | null => {
    if (!championshipState?.race_results) return null;
    const result = championshipState.race_results.find(
      (r: any) => r.event_order === eventOrder && r.driver_number === driverNumber
    );
    return result?.final_position ?? null;
  };

  // Helper function to get driver number at a specific position in an event
  const getDriverAtPosition = (eventOrder: number, position: number): number | null => {
    if (!championshipState?.race_results) return null;
    const result = championshipState.race_results.find(
      (r: any) => r.event_order === eventOrder && r.final_position === position
    );
    return result?.driver_number ?? null;
  };

  // Helper function to get track name from event order
  const getTrackName = (eventOrder: number): string => {
    const event = availableEvents.find(e => e.event_order === eventOrder);
    if (event) {
      return event.track_id;
    }
    // Fallback: try to get from race results
    if (championshipState?.race_results) {
      const result = championshipState.race_results.find((r: any) => r.event_order === eventOrder);
      if (result?.track_id) {
        return result.track_id;
      }
    }
    // Final fallback
    return `Event ${eventOrder}`;
  };

  const addAdjustment = () => {
    const fallbackEvent = availableEventOrders[0] ?? 1;
    const fallbackDriver = availableDrivers[0]?.number ?? 13;
    setAdjustments((prev) => [
      ...prev,
      { event_order: fallbackEvent, driver_number: fallbackDriver, new_position: 1 },
    ]);
  };

  const updateAdjustment = (index: number, patch: Partial<Adjustment>) => {
    setAdjustments((prev) => {
      const updated = prev.map((item, idx) => {
        if (idx !== index) return item;
        
        const updatedItem = { ...item, ...patch };
        const { event_order, driver_number, new_position, position_delta } = updatedItem;
        
        // Only auto-calculate if we have both event_order and driver_number
        if (event_order && driver_number) {
          const originalPosition = getOriginalPosition(event_order, driver_number);
          
          if (originalPosition !== null) {
            // If new_position was set, calculate position_delta
            if (patch.new_position !== undefined && patch.new_position !== null && patch.new_position !== "") {
              const targetPosition = Number(patch.new_position);
              const delta = targetPosition - originalPosition;
              updatedItem.position_delta = delta;
              updatedItem.new_position = targetPosition;
            }
            // If position_delta was set, calculate new_position
            else if (patch.position_delta !== undefined && patch.position_delta !== null && patch.position_delta !== "") {
              const newPos = originalPosition + Number(patch.position_delta);
              // Clamp between 1 and 20
              const clampedPos = Math.max(1, Math.min(20, newPos));
              updatedItem.new_position = clampedPos;
              updatedItem.position_delta = Number(patch.position_delta);
            }
            // If clearing one field, clear the other
            else if (patch.new_position === "" || patch.new_position === null || patch.new_position === undefined) {
              updatedItem.position_delta = undefined;
            }
            else if (patch.position_delta === "" || patch.position_delta === null || patch.position_delta === undefined) {
              updatedItem.new_position = undefined;
            }
          }
        }
        
        return updatedItem;
      });
      
      // After updating, automatically add position swaps for any new_position changes
      // Only do this if new_position was actually set in this update (not just selecting a driver)
      const finalUpdated: Adjustment[] = [...updated];
      const swapsToAdd: Adjustment[] = [];
      
      // Only process swaps if new_position was explicitly set in the patch
      if (patch.new_position !== undefined && patch.new_position !== null && patch.new_position !== "") {
        updated.forEach((item, idx) => {
          // Only process the item that was just updated (at the index)
          if (idx === index && item.new_position && item.event_order && item.driver_number) {
            const originalPos = getOriginalPosition(item.event_order, item.driver_number);
            if (originalPos !== null && originalPos !== item.new_position) {
              const driverAtTargetPos = getDriverAtPosition(item.event_order, item.new_position);
              if (driverAtTargetPos && driverAtTargetPos !== item.driver_number) {
                // Check if swap adjustment already exists
                const swapExists = finalUpdated.some(
                  (adj, adjIdx) =>
                    adjIdx !== index &&
                    adj.event_order === item.event_order &&
                    adj.driver_number === driverAtTargetPos &&
                    adj.new_position === originalPos
                );
                
                // Also check if the driver at target position already has any adjustment in this event
                const hasExistingAdjustment = finalUpdated.some(
                  (adj, adjIdx) =>
                    adjIdx !== index &&
                    adj.event_order === item.event_order &&
                    adj.driver_number === driverAtTargetPos
                );
                
                if (!swapExists && !hasExistingAdjustment) {
                  // Add automatic swap adjustment
                  swapsToAdd.push({
                    event_order: item.event_order,
                    driver_number: driverAtTargetPos,
                    new_position: originalPos,
                  });
                } else if (hasExistingAdjustment && !swapExists) {
                  // Update existing adjustment to swap position
                  const existingIndex = finalUpdated.findIndex(
                    (adj, adjIdx) =>
                      adjIdx !== index &&
                      adj.event_order === item.event_order &&
                      adj.driver_number === driverAtTargetPos
                  );
                  if (existingIndex !== -1) {
                    const existingOriginalPos = getOriginalPosition(item.event_order, driverAtTargetPos);
                    if (existingOriginalPos !== null) {
                      finalUpdated[existingIndex] = {
                        ...finalUpdated[existingIndex],
                        new_position: originalPos,
                        position_delta: originalPos - existingOriginalPos,
                      };
                    }
                  }
                }
              }
            }
          }
        });
      }
      
      // Add all new swap adjustments
      return [...finalUpdated, ...swapsToAdd];
    });
  };


  const fetchAIRecommendation = async () => {
    if (adjustments.length === 0) {
      setError("Please add scenario adjustments first to generate AI recommendations.");
      return;
    }

    setLoadingAI(true);
    setAiRecommendation(undefined);
    try {
      const body = {
        name,
        description,
        adjustments: adjustments.map((adj) => ({
          event_order: adj.event_order,
          changes: [
            {
              driver_number: adj.driver_number,
              new_position: adj.new_position,
              position_delta: adj.position_delta,
            },
          ],
        })),
      };
      const response = await fetch(`${API_BASE_URL}/ai/scenario-recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const json = await response.json();
      setAiRecommendation(json.recommendation);
      // Update result if scenario_result is included
      if (json.scenario_result) {
        setResult(json.scenario_result);
      }
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

  const runScenario = async () => {
    setLoading(true);
    setError(undefined);
    setAiRecommendation(undefined);
    try {
      const body = {
        name,
        description,
        adjustments: adjustments.map((adj) => ({
          event_order: adj.event_order,
          changes: [
            {
              driver_number: adj.driver_number,
              new_position: adj.new_position,
              position_delta: adj.position_delta,
            },
          ],
        })),
      };
      const response = await fetch(`${API_BASE_URL}/scenario/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const json = await response.json();
      setResult(json);
      
      // Automatically fetch AI recommendations after scenario runs
      if (json && adjustments.length > 0) {
        // Use setTimeout to avoid calling before state is updated
        setTimeout(() => {
          fetchAIRecommendation();
        }, 100);
      }
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

  return (
    <SectionCard
      title="Championship Scenario Playground"
      description="Pick a race, change a finishing spot, and press run."
      action={
        <button onClick={addAdjustment} className="ghost-button">
          + Add race change
        </button>
      }
    >
      {/* Instruction Box */}
      {showInstructions && (
        <div
          style={{
            background: "rgba(255, 51, 88, 0.1)",
            border: "1px solid rgba(255, 51, 88, 0.3)",
            borderRadius: "16px",
            padding: "1.25rem",
            marginBottom: "1.5rem",
            position: "relative",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
            <h3 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 600, color: "#fff" }}>
              📖 How to Use Scenario Lab
            </h3>
            <button
              onClick={() => setShowInstructions(false)}
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
          <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8" }}>
            <li>
              <strong>Click "+ Add race change"</strong> to create a new scenario adjustment
            </li>
            <li>
              <strong>Select an Event #</strong> from the dropdown (these are the races in the championship)
            </li>
            <li>
              <strong>Select a Driver #</strong> from the dropdown (choose which driver's result to change)
            </li>
            <li>
              <strong>Choose New Position</strong> (1-20) to set exactly where the driver should finish, OR
            </li>
            <li>
              <strong>Use Position Delta</strong> (+/-) to adjust their position relative to their original finish
            </li>
            <li>
              <strong>Add multiple adjustments</strong> to explore complex "what if" scenarios
            </li>
            <li>
              <strong>Click "Run scenario"</strong> to see how the championship would change
            </li>
          </ol>
          <p style={{ marginTop: "0.75rem", marginBottom: 0, fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.7)", fontStyle: "italic" }}>
            💡 Tip: Use "New Position" for exact finishes (e.g., "What if Driver #13 finished 1st?") or "Position Delta" for relative changes (e.g., "+3" means 3 positions better).
          </p>
        </div>
      )}

      {!showInstructions && (
        <button
          onClick={() => setShowInstructions(true)}
          className="ghost-button"
          style={{ marginBottom: "1rem" }}
        >
          📖 Show Instructions
        </button>
      )}


      {/* Technical Information Section */}
      <div style={{ marginBottom: "1.5rem" }}>
        <button
          onClick={() => setShowTechnicalInfo(!showTechnicalInfo)}
          className="ghost-button"
          style={{ width: "100%", marginBottom: "0.5rem" }}
        >
          {showTechnicalInfo ? "🔽 Hide" : "▶️ Show"} Technical Details: How Scenario Calculation Works
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
              🦋 How the Butterfly Effect Works
            </h3>
            
            <div style={{ marginBottom: "1rem" }}>
              <p style={{ margin: "0 0 1rem 0", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.95rem" }}>
                The butterfly effect means that changing one race result can ripple through the entire championship standings. 
                Here's how it's implemented:
              </p>
              
              <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                <li style={{ marginBottom: "0.75rem" }}>
                  <strong>You change a position:</strong> When you set a driver to a new finishing position (e.g., Driver #72 → 1st place), 
                  the system automatically swaps positions with the driver who was originally in that spot.
                </li>
                <li style={{ marginBottom: "0.75rem" }}>
                  <strong>Points are recalculated:</strong> Based on the new position, points are updated using the FIA points system 
                  (1st: 25pts, 2nd: 18pts, 3rd: 15pts, etc.). Bonus points for pole position and fastest lap are preserved.
                </li>
                <li style={{ marginBottom: "0.75rem" }}>
                  <strong>Championship standings update:</strong> All drivers' season totals are recalculated by summing their points 
                  across all races. The standings are then re-sorted by total points.
                </li>
                <li style={{ marginBottom: "0.75rem" }}>
                  <strong>The ripple effect:</strong> Even though you only changed one race, the championship order can completely shift 
                  because points accumulate across the entire season. A driver who wins one extra race might jump multiple positions in the final standings.
                </li>
              </ol>
            </div>

            <div style={{ marginTop: "1.5rem", marginBottom: "1rem" }}>
              <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#fff" }}>
                Main Technical Implementation:
              </h4>
              <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8", fontSize: "0.9rem" }}>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Position Conflict Resolution:</strong> When multiple drivers are assigned the same position, the system uses deterministic conflict resolution. 
                  Manual position changes are prioritized, and remaining drivers are placed in the next available slots while preserving their relative order.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Points Mapping:</strong> Uses a unified points table (1st: 25, 2nd: 18, 3rd: 15, 4th: 12, 5th: 10, 6th: 8, 7th: 6, 8th: 4, 9th: 2, 10th: 1). 
                  Points are recalculated from scratch based on final positions, ensuring no cached or stale values are used.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Standings Calculation:</strong> Groups all race results by driver_number, sums total_points (base_points + bonus_points) across all events, 
                  then sorts by season_points descending. Ties are broken by driver_number (ascending).
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Data Integrity:</strong> Handles NaN/missing positions gracefully (assigns 0 points), ensures all positions are valid integers (1-20), 
                  and maintains consistency between position and points columns.
                </li>
                <li style={{ marginBottom: "0.5rem" }}>
                  <strong>Automatic Position Swaps:</strong> When you change a driver's position, the system automatically moves the driver who was in that position 
                  to the original driver's position, creating a proper swap without manual intervention.
                </li>
              </ul>
            </div>

            <div style={{ 
              marginTop: "1rem", 
              padding: "0.75rem", 
              background: "rgba(0, 255, 198, 0.1)",
              borderRadius: "8px",
              border: "1px solid rgba(0, 255, 198, 0.3)"
            }}>
              <p style={{ margin: 0, fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", fontStyle: "italic" }}>
                💡 <strong>Example:</strong> If Driver #72 wins all 8 races (25 points each), they get 200 total points. 
                This single change affects every other driver's relative position in the championship, even though their individual race results didn't change.
              </p>
            </div>
          </div>
        )}
      </div>

      {topEvents.length === 0 && availableEventOrders.length === 0 && (
        <p style={{ color: "rgba(255, 255, 255, 0.7)", padding: "1rem", background: "rgba(255, 255, 255, 0.05)", borderRadius: "12px" }}>
          ⚠️ No impact reports available. Run cache builder first.
        </p>
      )}

      {adjustments.map((adj, index) => (
        <div key={index} className="scenario-grid" style={{ marginBottom: "1rem", padding: "1rem", background: "rgba(255, 255, 255, 0.03)", borderRadius: "12px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
          <div className="form-field">
            <span>Track / Event</span>
            <select
              value={adj.event_order || ""}
              onChange={(event) => {
                if (event.target.value) {
                  updateAdjustment(index, { event_order: Number(event.target.value) });
                }
              }}
            >
              <option value="" disabled style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
                🔽 Select to change track...
              </option>
              {availableEvents.length > 0 ? (
                availableEvents.map((event) => (
                  <option key={event.event_order} value={event.event_order}>
                    {event.track_id} (Event {event.event_order})
                  </option>
                ))
              ) : (
                <option value={1}>Event 1</option>
              )}
            </select>
          </div>
          <div className="form-field">
            <span>Driver #</span>
            <select
              value={adj.driver_number ?? ""}
              onChange={(event) =>
                updateAdjustment(index, {
                  driver_number: event.target.value ? Number(event.target.value) : undefined,
                })
              }
            >
              <option value="" style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
                🔽 Select to change driver...
              </option>
              {availableDrivers.map((driver) => (
                <option key={driver.number} value={driver.number}>
                  #{driver.number} - {driver.name}
                </option>
              ))}
            </select>
          </div>
          <div className="form-field">
            <span>New position (1-20)</span>
            <select
              value={adj.new_position ?? ""}
              onChange={(event) =>
                updateAdjustment(index, {
                  new_position: event.target.value ? Number(event.target.value) : undefined,
                  position_delta: undefined, // Clear delta when setting new position
                })
              }
            >
              <option value="" style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
                🔽 Select to change position...
              </option>
              {positionOptions.map((pos) => (
                <option key={pos} value={pos}>
                  Position {pos}
                </option>
              ))}
            </select>
          </div>
          <div className="form-field">
            <span>Change by (+/-)</span>
            <select
              value={adj.position_delta ?? ""}
              onChange={(event) =>
                updateAdjustment(index, {
                  position_delta: event.target.value ? Number(event.target.value) : undefined,
                  new_position: undefined, // Clear new position when using delta
                })
              }
            >
              <option value="" style={{ fontStyle: "italic", color: "rgba(0, 255, 198, 0.7)" }}>
                🔽 Select to change by...
              </option>
              {positionDeltaOptions.map((delta) => (
                <option key={delta} value={delta}>
                  {delta > 0 ? `+${delta}` : delta.toString()} positions
                </option>
              ))}
            </select>
          </div>
          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button
              onClick={() => setAdjustments((prev) => prev.filter((_, idx) => idx !== index))}
              className="ghost-button"
              style={{ padding: "0.6rem 1rem", fontSize: "0.85rem" }}
            >
              Remove
            </button>
          </div>
        </div>
      ))}

      <div className="form-grid">
        <div className="form-field">
          <span>Scenario name</span>
          <input value={name} onChange={(event) => setName(event.target.value)} />
        </div>
        <div className="form-field">
          <span>Description</span>
          <textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={2} />
        </div>
      </div>

      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
        <button className="primary-button" onClick={runScenario} disabled={loading || adjustments.length === 0}>
          {loading ? "Running…" : "Run scenario"}
        </button>
        {result && (
          <button 
            className="ghost-button" 
            onClick={fetchAIRecommendation} 
            disabled={loadingAI}
          >
            {loadingAI ? "Generating..." : "🤖 Get AI Analysis"}
          </button>
        )}
      </div>
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
              🤖 AI Strategic Analysis
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
            💡 This AI analysis references the scenario changes you made above and explains their championship implications, identifying key pivot points and strategic insights.
          </p>
        </div>
      )}

      <Modal
        isOpen={!!result}
        onClose={() => setResult(undefined)}
        title={`Scenario Results: ${result?.scenario_name ?? "Custom Scenario"}`}
      >
        <div className="scenario-results">
          <p style={{ marginBottom: "1.5rem", color: "rgba(255, 255, 255, 0.8)" }}>
            {result?.description}
          </p>
          
          {/* Show adjustments made */}
          {result?.adjustments && result.adjustments.length > 0 && (
            <div style={{ 
              marginBottom: "1.5rem", 
              padding: "1rem", 
              background: "rgba(0, 255, 198, 0.1)",
              borderRadius: "12px",
              border: "1px solid rgba(0, 255, 198, 0.3)"
            }}>
              <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#00ffc6" }}>
                📝 Adjustments Applied:
              </h4>
              <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.8" }}>
                {result.adjustments.map((adj: any, idx: number) => (
                  <li key={idx}>
                    <strong>{getTrackName(adj.event_order)} (Event {adj.event_order}):</strong>{" "}
                    {adj.changes?.map((change: any, cIdx: number) => (
                      <span key={cIdx}>
                        Driver #{change.driver_number} → Position {change.new_position ?? 
                          (change.position_delta ? `${change.position_delta > 0 ? '+' : ''}${change.position_delta}` : 'N/A')}
                        {cIdx < adj.changes.length - 1 ? ", " : ""}
                      </span>
                    ))}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <h3 style={{ marginBottom: "1rem" }}>Final Standings (Top 10)</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Pos</th>
                <th>Driver</th>
                <th>Points</th>
                <th>Change</th>
              </tr>
            </thead>
            <tbody>
              {result?.final_standings?.slice?.(0, 10).map((row: any, idx: number) => {
                // Find baseline position for comparison
                const baselineDriver = championshipState?.final_standings?.find(
                  (d: any) => d.driver_number === row.driver_number
                );
                const baselinePos = baselineDriver?.position ?? baselineDriver?.rank ?? null;
                const positionChange = baselinePos ? baselinePos - (idx + 1) : null;
                
                return (
                  <tr key={row.driver_number}>
                    <td>
                      <span style={{ 
                        display: "inline-flex", 
                        alignItems: "center", 
                        justifyContent: "center",
                        width: "28px",
                        height: "28px",
                        borderRadius: "50%",
                        background: idx < 3 
                          ? "linear-gradient(135deg, rgba(255, 51, 88, 0.4), rgba(0, 255, 198, 0.4))"
                          : "rgba(255, 255, 255, 0.1)",
                        fontSize: "0.8rem",
                        fontWeight: "700",
                        color: "#fff"
                      }}>
                        {idx + 1}
                      </span>
                    </td>
                    <td>#{row.driver_number} {row.driver_name}</td>
                    <td><strong>{row.season_points?.toFixed?.(1)}</strong></td>
                    <td>
                      {positionChange !== null && (
                        <span style={{
                          color: positionChange > 0 ? "#00ffc6" : positionChange < 0 ? "#ff3358" : "rgba(255, 255, 255, 0.5)",
                          fontWeight: 600
                        }}>
                          {positionChange > 0 ? `+${positionChange}` : positionChange < 0 ? positionChange : "—"}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          
          {adjustments.length > 0 && (
            <div style={{ 
              marginTop: "1.5rem", 
              padding: "1rem", 
              background: "rgba(255, 51, 88, 0.1)",
              borderRadius: "12px",
              border: "1px solid rgba(255, 51, 88, 0.3)"
            }}>
              <p style={{ margin: 0, fontSize: "0.9rem", color: "rgba(255, 255, 255, 0.8)", fontStyle: "italic" }}>
                💡 <strong>Note:</strong> If the standings haven't changed as expected, the adjustments might not be significant enough, 
                or multiple drivers may need position changes. Try making the target driver win in multiple races, or adjust positions 
                of competing drivers to create a larger points gap.
              </p>
            </div>
          )}
        </div>
      </Modal>
    </SectionCard>
  );
}

