from pydantic import BaseModel,Field,ConfigDict
from typing import Literal
from src.constants.enums import SegmentMode
class RunInput(BaseModel):
 model_config=ConfigDict(allow_inf_nan=False)
 batch_id:int=Field(gt=0)
 name:str=Field(min_length=2,max_length=180)
 min_size:int=Field(ge=1,le=600,strict=True)
 max_segments:int=Field(ge=1,le=12,strict=True)
 penalty:float=Field(ge=0,le=1e12)
 penalties:list[float]=Field(min_length=2,max_length=12)
 mode:Literal['free','monotone_mean']=SegmentMode.FREE
