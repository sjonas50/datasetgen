from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class ConnectorField(BaseModel):
    """
    Configuration field for a connector
    """
    name: str
    type: str  # string, number, boolean, etc.
    required: bool
    description: str
    default: Optional[Any] = None


class ConnectorInfo(BaseModel):
    """
    Information about a data connector
    """
    type: str
    name: str
    description: str
    supported_formats: List[str]
    configuration_fields: List[ConnectorField]
    features: List[str]  # streaming, batch, incremental, etc.


class ConnectorTest(BaseModel):
    """
    Schema for testing a connector configuration
    """
    config: Dict[str, Any]