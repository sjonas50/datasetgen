"""
SQLite database setup for development persistence
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import bcrypt

# Database file path
DB_PATH = Path("datasetgen.db")

class SQLiteDB:
    def __init__(self):
        self.db_path = DB_PATH
        self.init_db()
    
    def get_connection(self):
        """Get a database connection with row factory for dict results"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                size INTEGER NOT NULL,
                preview_data TEXT,
                owner_id TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_id) REFERENCES users (id)
            )
        ''')
        
        # Datasets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                schema_config TEXT,
                owner_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                row_count INTEGER,
                column_count INTEGER,
                quality_score REAL,
                quality_report TEXT,
                FOREIGN KEY (owner_id) REFERENCES users (id)
            )
        ''')
        
        # Dataset files relationship
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_files (
                dataset_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                PRIMARY KEY (dataset_id, file_id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (file_id) REFERENCES files (id)
            )
        ''')
        
        # Pipelines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipelines (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                dataset_id TEXT NOT NULL,
                steps TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_execution TIMESTAMP,
                status TEXT DEFAULT 'draft',
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (owner_id) REFERENCES users (id)
            )
        ''')
        
        # Pipeline executions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                error TEXT,
                results TEXT,
                FOREIGN KEY (pipeline_id) REFERENCES pipelines (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # User operations
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        cursor.execute('''
            INSERT INTO users (id, email, username, hashed_password, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, user_data['email'], user_data['username'], 
              user_data['hashed_password'], now))
        
        conn.commit()
        conn.close()
        
        return {
            "id": user_id,
            "email": user_data['email'],
            "username": user_data['username'],
            "is_active": True,
            "created_at": now
        }
    
    def get_user_by_username_or_email(self, username: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM users WHERE username = ? OR email = ?
        ''', (username, username))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    # File operations
    def create_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO files (id, filename, file_type, file_path, size, preview_data, owner_id, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (file_data['id'], file_data['filename'], file_data['file_type'],
              file_data['file_path'], file_data['size'], 
              json.dumps(file_data.get('preview_data', {})),
              file_data['owner_id'], file_data['uploaded_at']))
        
        conn.commit()
        conn.close()
        
        return file_data
    
    def get_files_by_owner(self, owner_id: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM files WHERE owner_id = ?', (owner_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        files = []
        for row in rows:
            file_dict = dict(row)
            if file_dict.get('preview_data'):
                file_dict['preview_data'] = json.loads(file_dict['preview_data'])
            files.append(file_dict)
        
        return files
    
    def get_file_by_id(self, file_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM files WHERE id = ? AND owner_id = ?', (file_id, owner_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            file_dict = dict(row)
            if file_dict.get('preview_data'):
                file_dict['preview_data'] = json.loads(file_dict['preview_data'])
            return file_dict
        return None
    
    # Dataset operations
    def create_dataset(self, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Insert dataset
        cursor.execute('''
            INSERT INTO datasets (id, name, description, schema_config, owner_id, 
                                created_at, updated_at, row_count, column_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dataset_data['id'], dataset_data['name'], dataset_data.get('description'),
              json.dumps(dataset_data.get('schema_config')), dataset_data['owner_id'],
              dataset_data['created_at'], dataset_data['updated_at'],
              dataset_data.get('row_count'), dataset_data.get('column_count')))
        
        # Link files to dataset
        for file_id in dataset_data.get('file_ids', []):
            cursor.execute('''
                INSERT INTO dataset_files (dataset_id, file_id) VALUES (?, ?)
            ''', (dataset_data['id'], file_id))
        
        conn.commit()
        conn.close()
        
        return dataset_data
    
    def get_datasets_by_owner(self, owner_id: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get datasets with their files
        cursor.execute('''
            SELECT d.*, GROUP_CONCAT(df.file_id) as file_ids
            FROM datasets d
            LEFT JOIN dataset_files df ON d.id = df.dataset_id
            WHERE d.owner_id = ?
            GROUP BY d.id
        ''', (owner_id,))
        
        rows = cursor.fetchall()
        
        datasets = []
        for row in rows:
            dataset = dict(row)
            if dataset.get('schema_config'):
                dataset['schema_config'] = json.loads(dataset['schema_config'])
            if dataset.get('quality_report'):
                dataset['quality_report'] = json.loads(dataset['quality_report'])
            
            # Get files for this dataset
            file_ids = dataset.pop('file_ids')
            if file_ids:
                file_ids = file_ids.split(',')
                dataset['files'] = []
                for file_id in file_ids:
                    cursor.execute('SELECT * FROM files WHERE id = ?', (file_id,))
                    file_row = cursor.fetchone()
                    if file_row:
                        file_dict = dict(file_row)
                        if file_dict.get('preview_data'):
                            file_dict['preview_data'] = json.loads(file_dict['preview_data'])
                        dataset['files'].append(file_dict)
            else:
                dataset['files'] = []
            
            datasets.append(dataset)
        
        conn.close()
        return datasets
    
    # Pipeline operations
    def create_pipeline(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pipelines (id, name, description, dataset_id, steps, owner_id,
                                 created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pipeline_data['id'], pipeline_data['name'], pipeline_data.get('description'),
              pipeline_data['dataset_id'], json.dumps(pipeline_data['steps']),
              pipeline_data['owner_id'], pipeline_data['created_at'],
              pipeline_data['updated_at'], pipeline_data.get('status', 'draft')))
        
        conn.commit()
        conn.close()
        
        return pipeline_data
    
    def get_pipelines_by_owner(self, owner_id: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM pipelines WHERE owner_id = ?', (owner_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        pipelines = []
        for row in rows:
            pipeline = dict(row)
            if pipeline.get('steps'):
                pipeline['steps'] = json.loads(pipeline['steps'])
            pipelines.append(pipeline)
        
        return pipelines
    
    def update_pipeline(self, pipeline_id: str, updates: Dict[str, Any]) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if 'steps' in updates:
            updates['steps'] = json.dumps(updates['steps'])
        
        if 'last_execution' in updates:
            cursor.execute('''
                UPDATE pipelines SET last_execution = ?, status = ? WHERE id = ?
            ''', (updates['last_execution'], updates.get('status', 'active'), pipeline_id))
        
        conn.commit()
        conn.close()

# Global database instance
db = SQLiteDB()