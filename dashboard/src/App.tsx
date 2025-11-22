import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bot, Workflow, Sparkles, Github, User, ArrowRight, Play, Trophy, Car } from "lucide-react";
import { useDashboardData } from "./hooks/useDashboardData";
import { TrackDNAView } from "./components/TrackDNAView";
import { TrackCoachInterface } from "./components/TrackCoachInterface";
import { SectionCard } from "./components/SectionCard";
import { ScenarioPlayground } from "./components/ScenarioPlayground";
import { ProjectOverview } from "./components/ProjectOverview";
import { DriverEmbeddingsView } from "./components/DriverEmbeddingsView";
import { Chatbot } from "./components/Chatbot";
import { SimulationRunner } from "./components/SimulationRunner";
import { BackendSetupGuide } from "./components/BackendSetupGuide";
import { ChampionshipState } from "./types";
import { KARMA_DASHBOARD_URL } from "./config";
import { isBackendConnectionError } from "./utils/errorDetection";

const TABS = [
  { id: "Overview", label: "Project Overview", icon: Sparkles, description: "Season details" },
  { id: "Driver Skills", label: "Drivers and Tracks Analysis", icon: User, description: "All drivers" },
  { id: "Track Coach Lab", label: "Coaching and Realtime Telemetry Analysis", icon: Bot, description: "Coaching" },
  { id: "Simulation", label: "Simulate Championship", icon: Play, description: "Run simulations" },
  { id: "Scenario Lab", label: "Scenario Lab", icon: Workflow, description: "What if" },
];

const GITHUB_REPO_URL = "https://github.com/yourname/grcup-dashboard";

const HERO_HIGHLIGHTS = [
  { title: "Track Understanding", detail: "Quantifies track complexity using KMeans clustering and feature extraction from telemetry data." },
  { title: "Outcome Prediction", detail: "Monte Carlo simulations with transfer learning models predict race and championship outcomes." },
  { title: "Strategic Insights", detail: "8D driver embeddings matched against track DNA generate personalized coaching recommendations." },
  { title: "Component Health", detail: "Random Forest Classifier (400 trees) predicts mechanical failure risks from real-time telemetry." },
];

export default function App() {
  const [tab, setTab] = useState(TABS[0].id);
  const { trackSummary, championshipState, coachData, edaData, driverEmbeddings, loading, error } = useDashboardData();

  const renderTab = () => {
    switch (tab) {
      case "Overview":
        return <ProjectOverview edaData={edaData} />;
      case "Track Coach Lab":
        return (
          <div className="space-y-6">
            <TrackCoachInterface entries={coachData} driverEmbeddings={driverEmbeddings} />
            <SectionCard
              title="Test Mechanical Karma Realtime"
              description="Access the real-time mechanical karma dashboard for live telemetry analysis and component health monitoring."
            >
              <div style={{ 
                padding: "2rem", 
                background: "rgba(255, 255, 255, 0.03)", 
                borderRadius: "12px", 
                border: "1px solid rgba(255, 255, 255, 0.1)",
                textAlign: "center"
              }}>
                <p style={{ 
                  marginBottom: "1.5rem", 
                  color: "rgba(255, 255, 255, 0.9)",
                  fontSize: "1rem",
                  lineHeight: "1.6"
                }}>
                  Monitor real-time mechanical karma scores, component health metrics, and telemetry analysis in the dedicated dashboard.
                </p>
                <button
                  onClick={() => window.open(KARMA_DASHBOARD_URL, "_blank")}
                  className="tab-button active"
                  style={{
                    padding: "0.75rem 2rem",
                    fontSize: "1rem",
                    fontWeight: 600,
                    cursor: "pointer",
                    minWidth: "250px"
                  }}
                >
                  Take me to the dashboard
                </button>
              </div>
            </SectionCard>
          </div>
        );
      case "Driver Skills":
        return (
          <div className="space-y-6">
            <DriverEmbeddingsView embeddings={driverEmbeddings} />
            <SectionCard title="Track DNA Profiler" description="See how each venue behaves in simple terms.">
              <TrackDNAView tracks={trackSummary} />
            </SectionCard>
          </div>
        );
      case "Scenario Lab":
        return (
          <ScenarioPlayground
            impactReports={championshipState?.impact_reports ?? []}
            championshipState={championshipState as ChampionshipState}
            driverEmbeddings={driverEmbeddings}
          />
        );
      case "Simulation":
        return <SimulationRunner />;
      default:
        return null;
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden px-4 py-8 sm:px-6 lg:px-10">
      {/* Floating Particles Background */}
      <div className="floating-particles">
        {Array.from({ length: 9 }).map((_, i) => (
          <div key={i} className="particle" />
        ))}
      </div>

      {/* Left Side Decorative Elements */}
      <div className="fixed left-0 top-0 bottom-0 w-32 pointer-events-none z-0 hidden xl:block">
        <motion.div
          className="absolute top-20 left-4"
          animate={{
            y: [0, -20, 0],
            rotate: [0, 5, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <Trophy className="w-16 h-16 text-[#ffd700] opacity-30" strokeWidth={1.5} />
        </motion.div>
        <motion.div
          className="absolute top-60 left-8"
          animate={{
            y: [0, 15, 0],
            rotate: [0, -5, 0],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 0.5,
          }}
        >
          <Car className="w-12 h-12 text-[#00ffc6] opacity-25" strokeWidth={1.5} />
        </motion.div>
        <motion.div
          className="absolute top-96 left-6"
          animate={{
            y: [0, -15, 0],
            rotate: [0, 3, 0],
          }}
          transition={{
            duration: 4.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1,
          }}
        >
          <Trophy className="w-14 h-14 text-[#ff8038] opacity-20" strokeWidth={1.5} />
        </motion.div>
        <motion.div
          className="absolute bottom-40 left-4"
          animate={{
            y: [0, 20, 0],
            rotate: [0, -3, 0],
          }}
          transition={{
            duration: 5.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1.5,
          }}
        >
          <Car className="w-10 h-10 text-[#ff3358] opacity-25" strokeWidth={1.5} />
        </motion.div>
      </div>

      {/* Right Side Decorative Elements */}
      <div className="fixed right-0 top-0 bottom-0 w-32 pointer-events-none z-0 hidden xl:block">
        <motion.div
          className="absolute top-32 right-6"
          animate={{
            y: [0, 18, 0],
            rotate: [0, -5, 0],
          }}
          transition={{
            duration: 4.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 0.3,
          }}
        >
          <Car className="w-14 h-14 text-[#00ffc6] opacity-25" strokeWidth={1.5} />
        </motion.div>
        <motion.div
          className="absolute top-72 right-4"
          animate={{
            y: [0, -18, 0],
            rotate: [0, 5, 0],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 0.8,
          }}
        >
          <Trophy className="w-16 h-16 text-[#ffd700] opacity-30" strokeWidth={1.5} />
        </motion.div>
        <motion.div
          className="absolute top-[28rem] right-8"
          animate={{
            y: [0, 15, 0],
            rotate: [0, -3, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1.2,
          }}
        >
          <Car className="w-12 h-12 text-[#ff8038] opacity-20" strokeWidth={1.5} />
        </motion.div>
        <motion.div
          className="absolute bottom-32 right-6"
          animate={{
            y: [0, -20, 0],
            rotate: [0, 4, 0],
          }}
          transition={{
            duration: 5.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1.8,
          }}
        >
          <Trophy className="w-14 h-14 text-[#ff3358] opacity-25" strokeWidth={1.5} />
        </motion.div>
      </div>
      
      <div className="relative mx-auto flex max-w-6xl flex-col gap-8 z-10">
        <header>
          <div className="hero-shell flex flex-col gap-6 md:flex-row">
            <div className="hero-content">
              <p className="hero-eyebrow">Track DNA Profiler & Championship Simulation</p>
              <h1 className="relative">
                <span className="relative z-10">Let's start a GR championship!</span>
                <motion.span
                  className="absolute inset-0 bg-gradient-to-r from-[#ff3358] via-[#00ffc6] to-[#ff8038] opacity-20 blur-2xl"
                  animate={{
                    opacity: [0.2, 0.3, 0.2],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                />
              </h1>
              <p className="relative z-10">
                Using ML concepts for analysis and simulation of a whole championship with all 36 drivers, 
                8 races, 4 venues with AI coaching insights and what-if scenarios.
              </p>

              <div className="hero-actions">
                <a href={GITHUB_REPO_URL} target="_blank" rel="noreferrer" className="primary-button hero-button" style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
                  <Github className="h-4 w-4" />
                  <span>View GitHub Repo</span>
                </a>
              </div>
            </div>
            <div className="hero-visual">
              {HERO_HIGHLIGHTS.map((item) => (
                <div key={item.title} className="hero-card">
                  <p>{item.title}</p>
                  <span>{item.detail}</span>
                </div>
              ))}
            </div>
          </div>
        </header>

        <div>
          <nav className="flex flex-nowrap items-start gap-2 overflow-x-auto">
            {TABS.map(({ id, label, icon: Icon, description }, index) => (
              <div key={id} className="flex items-center gap-2">
                <div className="flex flex-col items-center gap-2">
                  <motion.button
                    onClick={() => setTab(id)}
                    className="group relative overflow-hidden rounded-full border border-white/10 px-4 py-2 text-sm font-semibold tracking-wide text-white/70 transition-all duration-300 backdrop-blur-md"
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                    animate={{
                      backgroundColor: tab === id ? "rgba(255, 51, 88, 0.25)" : "rgba(15, 17, 23, 0.6)",
                      borderColor: tab === id ? "rgba(255, 51, 88, 0.8)" : "rgba(255,255,255,0.15)",
                      color: tab === id ? "#ffffff" : "rgba(255,255,255,0.8)",
                      boxShadow: tab === id ? "0 8px 24px rgba(255, 51, 88, 0.3)" : "0 4px 12px rgba(0, 0, 0, 0.2)",
                    }}
                  >
                    <div className="flex items-center gap-2 relative z-10">
                      <Icon className="h-4 w-4" />
                      <span>{label}</span>
                    </div>
                    {tab === id && (
                      <motion.div 
                        layoutId="tab-glow" 
                        className="absolute inset-0 bg-gradient-to-r from-[rgba(255,51,88,0.4)] via-[rgba(0,255,198,0.3)] to-[rgba(255,51,88,0.4)] opacity-50 blur-sm" 
                      />
                    )}
                    {tab === id && (
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
                        animate={{
                          x: ["-100%", "100%"],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          repeatDelay: 1,
                        }}
                      />
                    )}
                  </motion.button>
                  {description && (
                    <span className="text-xs text-white/50 italic text-center whitespace-nowrap">
                      {description}
                    </span>
                  )}
                </div>
                {index < TABS.length - 1 && (
                  <ArrowRight className="h-4 w-4 text-white/40 mt-6" />
                )}
              </div>
            ))}
          </nav>
        </div>

        {loading && <SectionCard title="Loading data">Spooling up telemetry, clustering, and forecasts...</SectionCard>}
        {error && isBackendConnectionError(error) && <BackendSetupGuide />}
        {error && !isBackendConnectionError(error) && <SectionCard title="Error">{error}</SectionCard>}

        {!loading && !error && (
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.25 }}
              className="space-y-6"
            >
              {renderTab()}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
      
      {/* Chatbot Component */}
      <Chatbot />
    </div>
  );
}

