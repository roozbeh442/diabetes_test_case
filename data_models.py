from pydantic import BaseModel
from pydantic.dataclasses import dataclass
 
 
class properties(BaseModel):
    value: float| None = 1.0
    max:   float = 1.0
    min:   float = 1.0
    mean:  float = 1.0
    std:   float = 1.0


class features(BaseModel):
    age: properties = properties()
    sex: properties = properties()
    bmi: properties = properties()
    bp: properties = properties()
    s1: properties = properties()
    s2: properties = properties()
    s3: properties = properties()
    s4: properties = properties()
    s5: properties = properties()
    s6: properties = properties()
    
class target(BaseModel):
    output: properties = properties()
    
class line_input(BaseModel):
    x: list[float] = [1]
    w: float = 1.0
    b: float = 1.0    