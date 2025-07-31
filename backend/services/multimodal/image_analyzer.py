import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
import base64

from services.llm.llm_factory import LLMFactory
from core.logging import get_logger

logger = get_logger(__name__)


class ImageAnalyzer:
    """
    Analyze images to extract data, charts, and visual information
    """
    
    def __init__(self, llm_provider: str = "claude"):
        self.llm_service = LLMFactory.create(llm_provider)
    
    async def extract_chart_data(
        self,
        image_path: str,
        chart_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract data points from charts and graphs
        """
        logger.info(f"Extracting chart data from: {image_path}")
        
        # Pre-process image for better extraction
        processed_image = self._preprocess_chart_image(image_path)
        
        prompt = """Analyze this chart/graph and extract all data points.

Identify:
1. Chart type (bar, line, pie, scatter, etc.)
2. Axes labels and units
3. Data series names/legend
4. All data points with precise values
5. Any annotations or notable features

Return the data in a structured format that can be converted to a dataset."""
        
        if chart_type:
            prompt += f"\n\nThe chart type is: {chart_type}"
        
        result = await self.llm_service.analyze_image(processed_image, prompt)
        
        # Parse the result
        import json
        try:
            chart_data = json.loads(result.content)
            
            # Convert to DataFrame if possible
            if "data_points" in chart_data:
                df = pd.DataFrame(chart_data["data_points"])
                chart_data["dataframe"] = df
            
            return chart_data
            
        except json.JSONDecodeError:
            # Return raw content if not JSON
            return {
                "raw_extraction": result.content,
                "chart_type": chart_type,
                "image_path": image_path
            }
    
    async def extract_table_from_image(
        self,
        image_path: str,
        preprocess: bool = True
    ) -> pd.DataFrame:
        """
        Extract tabular data from an image
        """
        logger.info(f"Extracting table from image: {image_path}")
        
        if preprocess:
            # Apply image processing for better table detection
            processed_image = self._preprocess_table_image(image_path)
        else:
            processed_image = image_path
        
        prompt = """Extract the table data from this image.

Requirements:
1. Identify all column headers
2. Extract all row data accurately
3. Preserve the table structure
4. Handle merged cells appropriately

Return as JSON with:
{
    "headers": ["col1", "col2", ...],
    "rows": [
        ["val1", "val2", ...],
        ...
    ]
}"""
        
        result = await self.llm_service.analyze_image(processed_image, prompt)
        
        # Convert to DataFrame
        import json
        try:
            table_data = json.loads(result.content)
            df = pd.DataFrame(
                table_data.get("rows", []),
                columns=table_data.get("headers", [])
            )
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse table data: {str(e)}")
            return pd.DataFrame()
    
    async def detect_and_extract_text_regions(
        self,
        image_path: str,
        region_types: List[str] = ["title", "caption", "annotation"]
    ) -> Dict[str, List[str]]:
        """
        Detect and extract text from specific regions in images
        """
        logger.info(f"Extracting text regions from: {image_path}")
        
        prompt = f"""Identify and extract text from these regions in the image:
{', '.join(region_types)}

For each region found:
1. Identify the region type
2. Extract the exact text
3. Note the approximate location (top, bottom, left, right, center)

Return as JSON:
{{
    "regions": [
        {{
            "type": "title",
            "text": "...",
            "location": "top-center"
        }},
        ...
    ]
}}"""
        
        result = await self.llm_service.analyze_image(image_path, prompt)
        
        # Parse and organize by type
        import json
        try:
            data = json.loads(result.content)
            
            text_regions = {}
            for region in data.get("regions", []):
                region_type = region.get("type", "unknown")
                if region_type not in text_regions:
                    text_regions[region_type] = []
                text_regions[region_type].append(region.get("text", ""))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Failed to parse text regions: {str(e)}")
            return {}
    
    async def analyze_dashboard_screenshot(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Analyze dashboard screenshots to extract metrics and KPIs
        """
        logger.info(f"Analyzing dashboard screenshot: {image_path}")
        
        prompt = """Analyze this dashboard screenshot and extract:

1. **Metrics/KPIs**: All numerical values with their labels
2. **Charts**: Identify all charts and their key data points
3. **Trends**: Any trend indicators (up/down arrows, percentages)
4. **Time Period**: Date ranges or time stamps
5. **Filters**: Any active filters or segments

Structure the output as:
{
    "metrics": [
        {"name": "...", "value": "...", "trend": "..."},
        ...
    ],
    "charts": [
        {"title": "...", "type": "...", "key_insights": "..."},
        ...
    ],
    "time_period": "...",
    "filters": [...]
}"""
        
        result = await self.llm_service.analyze_image(image_path, prompt)
        
        import json
        try:
            dashboard_data = json.loads(result.content)
            
            # Convert metrics to DataFrame for easy analysis
            if "metrics" in dashboard_data:
                metrics_df = pd.DataFrame(dashboard_data["metrics"])
                dashboard_data["metrics_dataframe"] = metrics_df
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to parse dashboard data: {str(e)}")
            return {"raw_analysis": result.content}
    
    def _preprocess_chart_image(self, image_path: str) -> str:
        """Preprocess chart images for better extraction"""
        # Load image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Save processed image
        processed_path = image_path.replace('.', '_processed.')
        cv2.imwrite(processed_path, enhanced)
        
        return processed_path
    
    def _preprocess_table_image(self, image_path: str) -> str:
        """Preprocess images containing tables"""
        # Load image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Detect lines (table borders)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_structure = cv2.add(horizontal_lines, vertical_lines)
        
        # Save processed image
        processed_path = image_path.replace('.', '_table_processed.')
        cv2.imwrite(processed_path, table_structure)
        
        return processed_path
    
    async def batch_analyze_images(
        self,
        image_paths: List[str],
        analysis_type: str = "chart"
    ) -> Dict[str, Any]:
        """
        Analyze multiple images in batch
        """
        import asyncio
        
        results = {}
        
        # Create analysis tasks
        tasks = []
        for image_path in image_paths:
            if analysis_type == "chart":
                task = self.extract_chart_data(image_path)
            elif analysis_type == "table":
                task = self.extract_table_from_image(image_path)
            elif analysis_type == "dashboard":
                task = self.analyze_dashboard_screenshot(image_path)
            else:
                continue
            
            tasks.append(task)
        
        # Execute in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for image_path, result in zip(image_paths, batch_results):
            if isinstance(result, Exception):
                results[image_path] = {"error": str(result)}
            else:
                results[image_path] = result
        
        return results
    
    def detect_visual_elements(self, image_path: str) -> Dict[str, Any]:
        """
        Detect various visual elements in an image using CV
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        elements = {
            "shapes": [],
            "text_regions": [],
            "charts": [],
            "tables": []
        }
        
        # Detect rectangles (potential charts/tables)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Classify based on aspect ratio and size
                if 0.5 < aspect_ratio < 2.0 and area > 5000:
                    elements["charts"].append({"bbox": [x, y, w, h], "area": area})
                elif aspect_ratio > 2.0:
                    elements["tables"].append({"bbox": [x, y, w, h], "area": area})
        
        return elements