from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    phone_number: str = Field(min_length=10, max_length=15)
    password: str = Field(min_length=8, max_length=64)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(max_length=64)


class UserResponse(BaseModel):
    id: str = Field(alias="_id", default=None)
    first_name: str
    last_name: str
    email: EmailStr
    phone_number: str

    model_config = {"populate_by_name": True}