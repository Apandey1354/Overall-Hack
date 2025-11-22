export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000/api";

export const KARMA_DASHBOARD_URL =
  import.meta.env.VITE_KARMA_DASHBOARD_URL ?? "http://localhost:3000";

