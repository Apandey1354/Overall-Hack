import { useEffect, useState } from "react";
import { DashboardData, ChampionshipState, EDAData, TrackCoachEntry, TrackSummary, DriverEmbedding } from "../types";
import { API_BASE_URL } from "../config";
import { isBackendConnectionError } from "../utils/errorDetection";

const SOURCES = {
  trackSummary: "/cache/track-dna",
  championship: "/cache/championship",
  coach: "/cache/track-coach",
  eda: "/eda/dashboard",
  driverEmbeddings: "/cache/driver-embeddings",
};

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export function useDashboardData(): DashboardData {
  const [trackSummary, setTrackSummary] = useState<TrackSummary[]>([]);
  const [championshipState, setChampionshipState] = useState<ChampionshipState | null>(null);
  const [coachData, setCoachData] = useState<TrackCoachEntry[]>([]);
  const [edaData, setEdaData] = useState<EDAData | null>(null);
  const [driverEmbeddings, setDriverEmbeddings] = useState<DriverEmbedding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  useEffect(() => {
    let isMounted = true;
    async function load() {
      try {
        const [tracks, champ, coach, eda, embeddings] = await Promise.all([
          fetchJson<TrackSummary[]>(SOURCES.trackSummary),
          fetchJson<ChampionshipState>(SOURCES.championship),
          fetchJson<TrackCoachEntry[]>(SOURCES.coach),
          fetchJson<EDAData>(SOURCES.eda),
          fetchJson<DriverEmbedding[]>(SOURCES.driverEmbeddings),
        ]);
        if (!isMounted) return;
        setTrackSummary(tracks);
        setChampionshipState(champ);
        setCoachData(coach);
        setEdaData(eda);
        setDriverEmbeddings(embeddings);
      } catch (err) {
        console.error(err);
        if (isMounted) {
          // Only set error if it's not a backend connection error
          // Backend connection errors will be handled by individual components
          if (!isBackendConnectionError(err)) {
            setError(String(err));
          } else {
            // Set a more helpful error message for backend connection issues
            setError("Backend API is not available. Please start the backend server.");
          }
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      isMounted = false;
    };
  }, []);

  return { trackSummary, championshipState, coachData, edaData, driverEmbeddings, loading, error };
}

