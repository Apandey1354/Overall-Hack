import { SectionCard } from "./SectionCard";
import { EDAExplorer } from "./EDAExplorer";
import { EDAData } from "../types";

interface Props {
  edaData: EDAData | null;
}

export function ProjectOverview({ edaData }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Project Introduction Section */}
      <SectionCard title="GR Cup Championship Simulator" description="ML-powered racing analytics platform">
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Project Overview */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              📋 What This Does
            </h4>
            <div style={{ 
              padding: "1rem", 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "12px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <p style={{ margin: 0, color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.6", fontSize: "0.95rem" }}>
                Uses <strong style={{ color: "#00ffc6" }}>ML algorithms</strong> (transfer learning, driver embeddings, Monte Carlo) to analyze tracks, 
                predict race outcomes, and simulate championship seasons. Transforms telemetry data into actionable racing insights.
              </p>
            </div>
          </div>

          {/* Core Components */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              🎯 Core Components
            </h4>
            <div className="grid-2" style={{ gap: "1rem" }}>
              <div style={{ 
                padding: "1rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "12px",
                border: "1px solid rgba(0, 255, 198, 0.3)"
              }}>
                <div style={{ fontSize: "1rem", fontWeight: 700, color: "#00ffc6", marginBottom: "0.5rem" }}>
                  🧬 Track DNA Profiler
                </div>
                <div style={{ fontSize: "0.9rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.6" }}>
                  Analyzes track complexity, speed profiles, and sector characteristics. Classifies tracks using ML clustering.
                </div>
              </div>
              <div style={{ 
                padding: "1rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "12px",
                border: "1px solid rgba(255, 51, 88, 0.3)"
              }}>
                <div style={{ fontSize: "1rem", fontWeight: 700, color: "#ff3358", marginBottom: "0.5rem" }}>
                  🏆 Championship Simulator
                </div>
                <div style={{ fontSize: "0.9rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.6" }}>
                  Runs Monte Carlo simulations to predict championship outcomes, win probabilities, and test "what-if" scenarios.
                </div>
              </div>
              <div style={{ 
                padding: "1rem", 
                background: "rgba(255, 255, 255, 0.05)", 
                borderRadius: "12px",
                border: "1px solid rgba(255, 128, 56, 0.3)"
              }}>
                <div style={{ fontSize: "1rem", fontWeight: 700, color: "#ff8038", marginBottom: "0.5rem" }}>
                  ⚙️ Realtime Mechanical Karma Testing
                </div>
                <div style={{ fontSize: "0.9rem", color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.6" }}>
                  Monitors component health (engine, gearbox, brakes, tires) in real-time from telemetry data to predict failure risks.
                </div>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              ⚡ Key Features
            </h4>
            <div style={{ 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "12px", 
              padding: "1rem",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem" }}>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px"
                }}>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#00ffc6", marginBottom: "0.25rem" }}>
                    👤 Driver Skills Analysis
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.5" }}>
                    8D skill vectors: technical ability, speed, consistency, weather adaptability
                  </div>
                </div>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px"
                }}>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#00ffc6", marginBottom: "0.25rem" }}>
                    🤖 AI Coaching
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.5" }}>
                    Personalized advice based on track DNA and driver profile
                  </div>
                </div>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px"
                }}>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#00ffc6", marginBottom: "0.25rem" }}>
                    🦋 Butterfly Effect
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.5" }}>
                    Test how single race changes affect championship outcomes
                  </div>
                </div>
                <div style={{ 
                  padding: "0.75rem", 
                  background: "rgba(255, 255, 255, 0.05)", 
                  borderRadius: "8px"
                }}>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#00ffc6", marginBottom: "0.25rem" }}>
                    📊 Visualizations
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.5" }}>
                    Interactive charts and dashboards for all data
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Problems Solved */}
          <div>
            <h4 style={{ margin: "0 0 0.75rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              🎯 Problems Solved
            </h4>
            <div style={{ 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "12px", 
              padding: "1rem",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem" }}>
                <div>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#ff3358", marginBottom: "0.5rem" }}>
                    📍 Track Understanding
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.6" }}>
                    Quantifies track difficulty, identifies similarities, explains behavior in simple terms
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#ff3358", marginBottom: "0.5rem" }}>
                    🎲 Outcome Prediction
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.6" }}>
                    Predicts race results and championship outcomes with probability estimates
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#ff3358", marginBottom: "0.5rem" }}>
                    💡 Strategic Insights
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.6" }}>
                    Provides coaching recommendations and highlights driver strengths/weaknesses
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "#ff3358", marginBottom: "0.5rem" }}>
                    ⚙️ Component Health
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)", lineHeight: "1.6" }}>
                    Real-time mechanical karma monitoring to predict component failure risks
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </SectionCard>

      {/* How It Works Section */}
      <SectionCard title="How It Works" description="Understanding the simulation">
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Roadmap to Champion */}
          <div>
            <h4 style={{ margin: "0 0 1rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              🗺️ Roadmap to Finding the Champion
            </h4>
            <div style={{ 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "12px", 
              padding: "1.5rem",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              {/* Tree/Flowchart Visualization */}
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", position: "relative" }}>
                {/* Level 1: Data Collection */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1rem 1.5rem",
                    background: "linear-gradient(135deg, rgba(255, 51, 88, 0.2), rgba(255, 51, 88, 0.1))",
                    borderRadius: "12px",
                    border: "2px solid rgba(255, 51, 88, 0.4)",
                    textAlign: "center",
                    minWidth: "250px"
                  }}>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#ff3358", marginBottom: "0.25rem" }}>
                      📊 Data Collection
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Race Results • Weather • Telemetry • Analysis
                    </div>
                  </div>
                </div>

                {/* Connector */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)" }} />
                </div>

                {/* Level 2: Feature Extraction */}
                <div style={{ display: "flex", justifyContent: "space-around", flexWrap: "wrap", gap: "1rem" }}>
                  <div style={{
                    padding: "0.9rem 1.2rem",
                    background: "linear-gradient(135deg, rgba(0, 255, 198, 0.2), rgba(0, 255, 198, 0.1))",
                    borderRadius: "10px",
                    border: "2px solid rgba(0, 255, 198, 0.4)",
                    textAlign: "center",
                    flex: "1",
                    minWidth: "200px"
                  }}>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#00ffc6", marginBottom: "0.25rem" }}>
                      🧬 Track DNA
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Complexity • Speed Profile • Characteristics
                    </div>
                  </div>
                  <div style={{
                    padding: "0.9rem 1.2rem",
                    background: "linear-gradient(135deg, rgba(0, 255, 198, 0.2), rgba(0, 255, 198, 0.1))",
                    borderRadius: "10px",
                    border: "2px solid rgba(0, 255, 198, 0.4)",
                    textAlign: "center",
                    flex: "1",
                    minWidth: "200px"
                  }}>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#00ffc6", marginBottom: "0.25rem" }}>
                      👤 Driver Embeddings
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      8D Skill Vectors • Strengths • Weaknesses
                    </div>
                  </div>
                </div>

                {/* Connectors */}
                <div style={{ display: "flex", justifyContent: "space-around", gap: "1rem" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)", marginLeft: "25%" }} />
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)", marginRight: "25%" }} />
                </div>

                {/* Level 3: ML Models */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1rem 1.5rem",
                    background: "linear-gradient(135deg, rgba(255, 128, 56, 0.2), rgba(255, 128, 56, 0.1))",
                    borderRadius: "12px",
                    border: "2px solid rgba(255, 128, 56, 0.4)",
                    textAlign: "center",
                    minWidth: "280px"
                  }}>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#ff8038", marginBottom: "0.25rem" }}>
                      🤖 Transfer Learning Model
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Predicts Performance from Track DNA + Driver Skills
                    </div>
                  </div>
                </div>

                {/* Connector */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)" }} />
                </div>

                {/* Level 4: Race Prediction */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1rem 1.5rem",
                    background: "linear-gradient(135deg, rgba(149, 117, 205, 0.2), rgba(149, 117, 205, 0.1))",
                    borderRadius: "12px",
                    border: "2px solid rgba(149, 117, 205, 0.4)",
                    textAlign: "center",
                    minWidth: "300px"
                  }}>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#9575cd", marginBottom: "0.25rem" }}>
                      🏁 Race Outcome Prediction
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Blends Embeddings + Track DNA + Transfer Model + Weather + Context
                    </div>
                  </div>
                </div>

                {/* Connector */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)" }} />
                </div>

                {/* Level 5: Points & Simulation */}
                <div style={{ display: "flex", justifyContent: "space-around", flexWrap: "wrap", gap: "1rem" }}>
                  <div style={{
                    padding: "0.9rem 1.2rem",
                    background: "linear-gradient(135deg, rgba(38, 166, 154, 0.2), rgba(38, 166, 154, 0.1))",
                    borderRadius: "10px",
                    border: "2px solid rgba(38, 166, 154, 0.4)",
                    textAlign: "center",
                    flex: "1",
                    minWidth: "200px"
                  }}>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#26a69a", marginBottom: "0.25rem" }}>
                      📈 Points Calculation
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      FIA Points System • Bonus Points
                    </div>
                  </div>
                  <div style={{
                    padding: "0.9rem 1.2rem",
                    background: "linear-gradient(135deg, rgba(38, 166, 154, 0.2), rgba(38, 166, 154, 0.1))",
                    borderRadius: "10px",
                    border: "2px solid rgba(38, 166, 154, 0.4)",
                    textAlign: "center",
                    flex: "1",
                    minWidth: "200px"
                  }}>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#26a69a", marginBottom: "0.25rem" }}>
                      🔄 Season Simulation
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      All Races • Cumulative Points • Driver States
                    </div>
                  </div>
                </div>

                {/* Connectors */}
                <div style={{ display: "flex", justifyContent: "space-around", gap: "1rem" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)", marginLeft: "25%" }} />
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)", marginRight: "25%" }} />
                </div>

                {/* Level 6: Monte Carlo */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1rem 1.5rem",
                    background: "linear-gradient(135deg, rgba(255, 202, 40, 0.2), rgba(255, 202, 40, 0.1))",
                    borderRadius: "12px",
                    border: "2px solid rgba(255, 202, 40, 0.4)",
                    textAlign: "center",
                    minWidth: "280px"
                  }}>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#ffca28", marginBottom: "0.25rem" }}>
                      🎲 Monte Carlo Simulation
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      500+ Iterations • Probability Estimates • Win Odds
                    </div>
                  </div>
                </div>

                {/* Connector */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(0, 255, 198, 0.5)" }} />
                </div>

                {/* Level 7: Champion */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1.25rem 2rem",
                    background: "linear-gradient(135deg, rgba(255, 215, 0, 0.3), rgba(255, 215, 0, 0.2))",
                    borderRadius: "16px",
                    border: "3px solid rgba(255, 215, 0, 0.6)",
                    textAlign: "center",
                    minWidth: "320px",
                    boxShadow: "0 8px 32px rgba(255, 215, 0, 0.3)"
                  }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#ffd700", marginBottom: "0.5rem" }}>
                      🏆 CHAMPION IDENTIFIED
                    </div>
                    <div style={{ fontSize: "0.9rem", color: "rgba(255, 255, 255, 0.9)", fontWeight: 500 }}>
                      Final Standings • Championship Winner • Probability Rankings
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Description Cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <h4 style={{ margin: "0 0 0.5rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#00ffc6" }}>
                🎯 The Simulation Process
              </h4>
              <p style={{ margin: 0, color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.6", fontSize: "0.9rem" }}>
                We imagine a season that visits each venue in the data set. For every race, the simulator mixes driver form, 
                weather conditions, and track difficulty to predict finishing positions. You can then tweak a race in Scenario Lab 
                to see how the title fight changes through the butterfly effect.
              </p>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* How Mechanical Karma Works Section */}
      <SectionCard title="How Mechanical Karma Works" description="Real-time component health monitoring">
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Mechanical Karma Flow */}
          <div>
            <h4 style={{ margin: "0 0 1rem 0", fontSize: "1rem", fontWeight: 600, color: "#fff" }}>
              ⚙️ Component Health Prediction Pipeline
            </h4>
            <div style={{ 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "12px", 
              padding: "1.5rem",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              {/* Flowchart Visualization */}
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", position: "relative" }}>
                {/* Level 1: Telemetry Data */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1rem 1.5rem",
                    background: "linear-gradient(135deg, rgba(255, 51, 88, 0.2), rgba(255, 51, 88, 0.1))",
                    borderRadius: "12px",
                    border: "2px solid rgba(255, 51, 88, 0.4)",
                    textAlign: "center",
                    minWidth: "250px"
                  }}>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#ff3358", marginBottom: "0.25rem" }}>
                      📊 Telemetry Data
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Speed • RPM • Brake Pressure • Gear • Steering
                    </div>
                  </div>
                </div>

                {/* Connector */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(255, 128, 56, 0.5)" }} />
                </div>

                {/* Level 2: Feature Extraction */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1rem 1.5rem",
                    background: "linear-gradient(135deg, rgba(255, 128, 56, 0.2), rgba(255, 128, 56, 0.1))",
                    borderRadius: "12px",
                    border: "2px solid rgba(255, 128, 56, 0.4)",
                    textAlign: "center",
                    minWidth: "280px"
                  }}>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#ff8038", marginBottom: "0.25rem" }}>
                      🔧 Per-Lap Feature Extraction
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Mean/Std Speed • RPM • Brake Pressure • Gear Usage • Steering Angle
                    </div>
                  </div>
                </div>

                {/* Connector */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(255, 128, 56, 0.5)" }} />
                </div>

                {/* Level 3: Random Forest Classifier */}
                <div style={{ display: "flex", justifyContent: "space-around", flexWrap: "wrap", gap: "1rem" }}>
                  <div style={{
                    padding: "0.9rem 1.2rem",
                    background: "linear-gradient(135deg, rgba(0, 255, 198, 0.2), rgba(0, 255, 198, 0.1))",
                    borderRadius: "10px",
                    border: "2px solid rgba(0, 255, 198, 0.4)",
                    textAlign: "center",
                    flex: "1",
                    minWidth: "200px"
                  }}>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#00ffc6", marginBottom: "0.25rem" }}>
                      🌲 Random Forest Classifier
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      400 Trees • Max Depth 12 • Trained on Failure Data
                    </div>
                  </div>
                  <div style={{
                    padding: "0.9rem 1.2rem",
                    background: "linear-gradient(135deg, rgba(0, 255, 198, 0.2), rgba(0, 255, 198, 0.1))",
                    borderRadius: "10px",
                    border: "2px solid rgba(0, 255, 198, 0.4)",
                    textAlign: "center",
                    flex: "1",
                    minWidth: "200px"
                  }}>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#00ffc6", marginBottom: "0.25rem" }}>
                      📈 Component Models
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Engine • Gearbox • Brakes • Tires
                    </div>
                  </div>
                </div>

                {/* Connectors */}
                <div style={{ display: "flex", justifyContent: "space-around", gap: "1rem" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(255, 128, 56, 0.5)", marginLeft: "25%" }} />
                  <div style={{ width: "2px", height: "30px", background: "rgba(255, 128, 56, 0.5)", marginRight: "25%" }} />
                </div>

                {/* Level 4: Karma Scores */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1rem 1.5rem",
                    background: "linear-gradient(135deg, rgba(149, 117, 205, 0.2), rgba(149, 117, 205, 0.1))",
                    borderRadius: "12px",
                    border: "2px solid rgba(149, 117, 205, 0.4)",
                    textAlign: "center",
                    minWidth: "300px"
                  }}>
                    <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#9575cd", marginBottom: "0.25rem" }}>
                      ⚡ Karma Score Calculation
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.8)" }}>
                      Failure Probability • Wear Rate • Smoothing • Component Health
                    </div>
                  </div>
                </div>

                {/* Connector */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{ width: "2px", height: "30px", background: "rgba(255, 128, 56, 0.5)" }} />
                </div>

                {/* Level 5: Real-time Monitoring */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div style={{
                    padding: "1.25rem 2rem",
                    background: "linear-gradient(135deg, rgba(255, 215, 0, 0.3), rgba(255, 215, 0, 0.2))",
                    borderRadius: "16px",
                    border: "3px solid rgba(255, 215, 0, 0.6)",
                    textAlign: "center",
                    minWidth: "320px",
                    boxShadow: "0 8px 32px rgba(255, 215, 0, 0.3)"
                  }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#ffd700", marginBottom: "0.5rem" }}>
                      📊 REAL-TIME HEALTH MONITORING
                    </div>
                    <div style={{ fontSize: "0.9rem", color: "rgba(255, 255, 255, 0.9)", fontWeight: 500 }}>
                      Component Risk Scores • Failure Predictions • Wear Tracking
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Description Cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <h4 style={{ margin: "0 0 0.5rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#00ffc6" }}>
                🌲 Random Forest Classifier
              </h4>
              <p style={{ margin: 0, color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.6", fontSize: "0.9rem" }}>
                We use a <strong>Random Forest Classifier</strong> with 400 decision trees and max depth of 12 to predict component failures. 
                The model is trained on historical telemetry data where component failures are known. For each component (engine, gearbox, 
                brakes, tires), we extract per-lap features (mean speed, RPM, brake pressure, gear usage, steering angle) and feed them 
                into the classifier to predict failure probability. Higher karma scores indicate increased stress and degradation risk.
              </p>
            </div>
            <div style={{ 
              padding: "0.75rem", 
              background: "rgba(255, 255, 255, 0.03)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.1)"
            }}>
              <h4 style={{ margin: "0 0 0.5rem 0", fontSize: "0.95rem", fontWeight: 600, color: "#00ffc6" }}>
                ⚡ Karma Score Calculation
              </h4>
              <p style={{ margin: 0, color: "rgba(255, 255, 255, 0.9)", lineHeight: "1.6", fontSize: "0.9rem" }}>
                The karma stream combines Random Forest predictions with wear rate modeling and exponential smoothing. Each component's 
                karma score accumulates over time based on telemetry stress indicators. The system tracks engine RPM stress, gearbox 
                jolts, brake pressure spikes, and tire cornering loads to provide real-time component health monitoring.
              </p>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Data Health Check Section */}
      <SectionCard title="Data Health Check" description="Verify your data is loaded correctly">
        <EDAExplorer data={edaData} />
      </SectionCard>

    </div>
  );
}

