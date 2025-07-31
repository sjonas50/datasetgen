from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from core.database import get_db
from schemas.connector import ConnectorInfo, ConnectorTest
from services.connector_service import ConnectorService

router = APIRouter()


@router.get("/", response_model=List[ConnectorInfo])
async def list_connectors():
    """
    List all available data source connectors
    """
    service = ConnectorService()
    return service.list_available_connectors()


@router.get("/{connector_type}", response_model=ConnectorInfo)
async def get_connector(connector_type: str):
    """
    Get details about a specific connector
    """
    service = ConnectorService()
    connector = service.get_connector_info(connector_type)
    
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    
    return connector


@router.post("/{connector_type}/test")
async def test_connector(
    connector_type: str,
    config: ConnectorTest,
    db: AsyncSession = Depends(get_db)
):
    """
    Test connection to a data source
    """
    service = ConnectorService()
    
    try:
        result = await service.test_connection(connector_type, config.config)
        return {
            "status": "success" if result["success"] else "failed",
            "message": result.get("message", "Connection test completed"),
            "details": result.get("details", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


@router.post("/{connector_type}/schema")
async def get_schema(
    connector_type: str,
    config: ConnectorTest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get schema/structure from a data source
    """
    service = ConnectorService()
    
    try:
        schema = await service.get_schema(connector_type, config.config)
        return {
            "connector_type": connector_type,
            "schema": schema
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve schema: {str(e)}")