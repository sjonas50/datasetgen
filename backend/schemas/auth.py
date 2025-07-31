from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
import uuid


class UserBase(BaseModel):
    """
    Base user schema
    """
    email: EmailStr
    username: str


class UserCreate(UserBase):
    """
    Schema for user registration
    """
    password: str


class UserResponse(UserBase):
    """
    Schema for user response
    """
    id: uuid.UUID
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """
    JWT token response
    """
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Token payload data
    """
    username: Optional[str] = None
    user_id: Optional[uuid.UUID] = None