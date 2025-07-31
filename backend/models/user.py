from sqlalchemy import Column, String, Boolean
from sqlalchemy.orm import relationship

from .base import BaseModel


class User(BaseModel):
    """
    User model for authentication and authorization
    """
    __tablename__ = "users"
    
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    pipelines = relationship("Pipeline", back_populates="owner", cascade="all, delete-orphan")