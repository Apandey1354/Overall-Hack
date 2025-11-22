/**
 * Detects if an error is a backend connection error
 */
export function isBackendConnectionError(error: unknown): boolean {
  if (!error) return false;

  const errorMessage = error instanceof Error ? error.message : String(error);
  const errorString = errorMessage.toLowerCase();

  // Check for common connection error patterns
  const connectionErrorPatterns = [
    'failed to fetch',
    'network error',
    'networkerror',
    'connection refused',
    'connection reset',
    'connection closed',
    'err_connection_refused',
    'err_network_changed',
    'err_internet_disconnected',
    'fetch failed',
    'load failed',
    'cors',
    'refused to connect',
    'unable to connect',
    'cannot connect',
    'connection timeout',
    'timeout',
  ];

  return connectionErrorPatterns.some(pattern => errorString.includes(pattern));
}

/**
 * Checks if a fetch response indicates backend is unavailable
 */
export function isBackendUnavailable(response: Response | null): boolean {
  if (!response) return true;
  
  // Connection errors typically result in no response or network errors
  // HTTP errors (4xx, 5xx) are different - backend is available but returned an error
  return false;
}


