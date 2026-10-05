from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from ml_agent import load_trained_model, solve_cube_agent

app = FastAPI(title="Rubiks ML Agent API")

# Change this line in main.py:
cfop_model = load_trained_model("rubiks_rl_weights.pth")

class ScrambleState(BaseModel):
    scramble_alg: str

@app.post("/solve")
async def solve_endpoint(state: ScrambleState):
    solution_steps = solve_cube_agent(state.scramble_alg, cfop_model)
    move_sequence = [step["ideal_move"] for step in solution_steps]
    
    return {
        "status": "success",
        "ml_metadata": solution_steps,
        "sequence": move_sequence
    }

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)