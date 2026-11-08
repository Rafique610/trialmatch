import type { NextConfig } from "next";

/** The browser talks to /api/*, which Next proxies to the FastAPI backend (no CORS, one origin). */
export const API_URL = process.env.TRIALMATCH_API ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
  experimental: { proxyTimeout: 180_000 }, // a full benchmark run can take tens of seconds on a 6 GB laptop GPU
  devIndicators: false, // keep demo recordings and screenshots clean
};

export default config;
