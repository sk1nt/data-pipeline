#!/usr/bin/env python3
"""
Launcher script for the GEX Data Pipeline Server.

This script allows running the server from the repository root.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.data_pipeline:app",
        host="0.0.0.0",
        port=8877,
        reload=True,
        log_level="info"
    )