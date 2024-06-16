
from multi_regression_utils import simple_line
from data_models import line_input
from fastapi import FastAPI
import uvicorn


app = FastAPI()

@app.get('/')
def root():
    print('this is a simple fucntion that returns a line value')
    
@app.post('/calc')
def line_calc(data:line_input):
    return simple_line(data)   