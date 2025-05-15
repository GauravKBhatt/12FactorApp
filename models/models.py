from pydantic import BaseModel

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    house_age: int
    was_renovated: int
    total_sqft: float