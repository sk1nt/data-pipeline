from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.gex_api import router as gex_router

app = FastAPI(
    title="Data Pipeline API",
    description="API for importing and querying GEX data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(gex_router, prefix="/api", tags=["GEX"])

@app.get("/")
async def root():
    return {"message": "Data Pipeline API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}