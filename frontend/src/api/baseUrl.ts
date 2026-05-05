const DEFAULT_API_BASE_URL = "/api";

interface ViteImportMeta extends ImportMeta {
  env?: {
    VITE_API_BASE_URL?: string;
  };
}

function resolveApiBaseUrl(): string {
  const configuredBaseUrl = (import.meta as ViteImportMeta).env?.VITE_API_BASE_URL?.trim();
  return (configuredBaseUrl || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}

export function apiUrl(path: string): string {
  const normalizedPath = path.replace(/^\/+/, "");
  return `${resolveApiBaseUrl()}/${normalizedPath}`;
}
