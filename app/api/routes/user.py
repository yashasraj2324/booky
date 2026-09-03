from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.models.users import UserRegister, UserResponse
from app.api.routes.components.security import hash_password, verify_password, create_access_token
from app.database.mongodb import users_collection

router = APIRouter(prefix="/users", tags=["Users"])


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user_dict = user.model_dump()
    user_dict["password"] = hash_password(user_dict["password"])
    
    await users_collection.insert_one(user_dict)
    
    return UserResponse(**user_dict)


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db_user = await users_collection.find_one({"email": form_data.username})
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    if not verify_password(form_data.password, db_user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    access_token = create_access_token(
        data={"sub": db_user["email"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}

