export interface TrackSummary {
  track_id: string;
  venue: string;
  race: string;
  complexity_score?: number;
  overall_sector_std?: number;
  braking_zones?: number;
  straight_corner_ratio?: number;
  top_speed?: number;
  track_length_km?: number;
  num_sectors?: number;
  cluster_label?: string;
}

export interface ChampionshipState {
  final_standings: Array<Record<string, unknown>>;
  race_results: Array<Record<string, unknown>>;
  monte_carlo_summary: Array<Record<string, unknown>>;
  impact_reports: ImpactReport[];
}

export interface ImpactReport {
  event_order: number;
  track_id: string;
  impact_score: number;
  champion_changed: boolean;
  champion_before: string;
  champion_after: string;
  max_points_delta: number;
  key_movers: Array<{
    driver_number: number;
    driver_name: string;
    points_delta_abs: number;
    rank_delta: number;
  }>;
}

export interface TrackCoachEntry {
  track_id: string;
  overview: Record<string, unknown>;
  sector_recommendations: Array<Record<string, unknown>>;
  weather_strategies: Record<string, unknown>;
  driver_advice_sample: Record<string, unknown>;
}

export interface BoxPlotStat {
  label: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  count: number;
}

export interface HistogramBin {
  bin: string;
  midpoint: number;
  [category: string]: number | string;
}

export interface EDAHeatmap {
  venues: string[];
  file_types: string[];
  cells: Array<{ venue: string; file_type: string; found: boolean }>;
}

export interface EDAResultsSection {
  summary: Record<string, number>;
  lap_boxplot: BoxPlotStat[];
  fastest_lap_histogram: { bins: HistogramBin[]; categories: string[] };
  best_lap_histogram: { bins: HistogramBin[]; categories: string[] };
}

export interface EDAWeatherSection {
  metrics: Record<string, { title: string; summary: Record<string, number>; boxplot: BoxPlotStat[] }>;
}

export interface EDATelemetrySection {
  venues_covered: number;
  samples: number;
  unique_parameters: number;
  top_parameters: Array<{ parameter: string; count: number }>;
}

export interface EDAData {
  generated_at: string;
  completeness: {
    summary: {
      total_checks: number;
      files_found: number;
      files_missing: number;
      overall_pct: number;
    };
    breakdown: Array<{ venue: string; found: number; total: number; pct: number }>;
    heatmap: EDAHeatmap;
  };
  results: EDAResultsSection;
  weather: EDAWeatherSection;
  telemetry: EDATelemetrySection;
}

export interface DriverEmbedding {
  driver_number: number;
  driver_name: string;
  skill_vector: number[];
  technical_proficiency: number;
  high_speed_proficiency: number;
  consistency_score: number;
  weather_adaptability: number;
  best_track_type: string;
  strengths: string;
}

export interface DashboardData {
  trackSummary: TrackSummary[];
  championshipState: ChampionshipState | null;
  coachData: TrackCoachEntry[];
  edaData: EDAData | null;
  driverEmbeddings: DriverEmbedding[];
  loading: boolean;
  error?: string;
}

