from fastapi import APIRouter

from .endpoints import datasets, pipelines, connectors, auth, monitoring

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(connectors.router, prefix="/connectors", tags=["connectors"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])