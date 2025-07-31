from typing import List, Dict, Any, Optional

from schemas.connector import ConnectorInfo, ConnectorField
from connectors import IntelligentCSVConnector, IntelligentJSONConnector
from connectors.base import ConnectorConfig, ConnectorResult


class ConnectorService:
    """
    Service for managing data connectors
    """
    
    def __init__(self):
        # Define available connectors
        self.connectors = {
            "csv": {
                "type": "csv",
                "name": "CSV File",
                "description": "Import data from CSV files",
                "supported_formats": [".csv"],
                "configuration_fields": [
                    ConnectorField(
                        name="file_path",
                        type="string",
                        required=True,
                        description="Path to the CSV file"
                    ),
                    ConnectorField(
                        name="delimiter",
                        type="string",
                        required=False,
                        description="Column delimiter",
                        default=","
                    ),
                    ConnectorField(
                        name="encoding",
                        type="string",
                        required=False,
                        description="File encoding",
                        default="utf-8"
                    )
                ],
                "features": ["batch", "schema_inference"]
            },
            "json": {
                "type": "json",
                "name": "JSON File",
                "description": "Import data from JSON files",
                "supported_formats": [".json", ".jsonl"],
                "configuration_fields": [
                    ConnectorField(
                        name="file_path",
                        type="string",
                        required=True,
                        description="Path to the JSON file"
                    ),
                    ConnectorField(
                        name="json_path",
                        type="string",
                        required=False,
                        description="JSONPath expression to extract data"
                    )
                ],
                "features": ["batch", "nested_data", "schema_inference"]
            },
            "postgresql": {
                "type": "postgresql",
                "name": "PostgreSQL Database",
                "description": "Connect to PostgreSQL databases",
                "supported_formats": ["table", "query"],
                "configuration_fields": [
                    ConnectorField(
                        name="host",
                        type="string",
                        required=True,
                        description="Database host"
                    ),
                    ConnectorField(
                        name="port",
                        type="number",
                        required=False,
                        description="Database port",
                        default=5432
                    ),
                    ConnectorField(
                        name="database",
                        type="string",
                        required=True,
                        description="Database name"
                    ),
                    ConnectorField(
                        name="username",
                        type="string",
                        required=True,
                        description="Database username"
                    ),
                    ConnectorField(
                        name="password",
                        type="string",
                        required=True,
                        description="Database password"
                    ),
                    ConnectorField(
                        name="table",
                        type="string",
                        required=False,
                        description="Table name (if not using query)"
                    ),
                    ConnectorField(
                        name="query",
                        type="string",
                        required=False,
                        description="SQL query (if not using table)"
                    )
                ],
                "features": ["batch", "streaming", "incremental", "schema_metadata"]
            }
        }
    
    def list_available_connectors(self) -> List[ConnectorInfo]:
        """
        List all available connectors
        """
        return [
            ConnectorInfo(**connector_data)
            for connector_data in self.connectors.values()
        ]
    
    def get_connector_info(self, connector_type: str) -> Optional[ConnectorInfo]:
        """
        Get information about a specific connector
        """
        if connector_type in self.connectors:
            return ConnectorInfo(**self.connectors[connector_type])
        return None
    
    async def test_connection(self, connector_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test connection to a data source
        """
        if connector_type not in self.connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        # Create connector instance
        connector = self._create_connector(connector_type, config)
        
        # Test connection
        result = await connector.connect()
        
        if result.success:
            # Try to read a small sample
            sample_result = await connector.read_data(limit=5)
            
            return {
                "success": True,
                "message": f"Successfully connected to {connector_type}",
                "details": {
                    "connector_type": connector_type,
                    "metadata": result.metadata,
                    "sample_rows": sample_result.data.shape[0] if sample_result.success and sample_result.data is not None else 0,
                    "columns": list(sample_result.data.columns) if sample_result.success and sample_result.data is not None else []
                }
            }
        else:
            return {
                "success": False,
                "message": result.error,
                "details": {}
            }
    
    async def get_schema(self, connector_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get schema from a data source
        """
        if connector_type not in self.connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        # Create connector instance
        connector = self._create_connector(connector_type, config)
        
        # Connect and infer schema
        connect_result = await connector.connect()
        if not connect_result.success:
            raise Exception(f"Failed to connect: {connect_result.error}")
        
        # Infer schema
        schema = await connector.infer_schema()
        
        return {
            "connector_type": connector_type,
            "schema": schema.dict() if schema else None
        }
    
    def _create_connector(self, connector_type: str, config: Dict[str, Any]):
        """Create a connector instance"""
        if connector_type == "csv":
            return IntelligentCSVConnector(config)
        elif connector_type == "json":
            return IntelligentJSONConnector(config)
        elif connector_type == "postgresql":
            # TODO: Implement DatabaseConnector
            raise NotImplementedError("PostgreSQL connector not yet implemented")
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")