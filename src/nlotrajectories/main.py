from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import yaml
from pathlib import Path
import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS, etc.
    allow_headers=["*"],  # allow content-type, etc.
)

# Serve the image
from fastapi.staticfiles import StaticFiles
current_dir = os.path.dirname(__file__)
benchmark_dir = os.path.join(current_dir, "benchmarks")
app.mount("/benchmarks", StaticFiles(directory=benchmark_dir), name="benchmarks")

YAML_FILE = os.path.join(benchmark_dir, "generated.yaml")
IMAGE_FILE = Path('benchmark') / "generated.png"

class YamlRequest(BaseModel):
    config: dict

@app.post("/generate-and-run")
def generate_and_run(request: YamlRequest):
    os.makedirs(benchmark_dir, exist_ok=True)

    # Save YAML config
    with open(YAML_FILE, "w") as f:
        yaml.dump(request.config, f)

    # Run your benchmark using poetry
    result = subprocess.run(
        ["poetry", "run", "run-benchmark", "--config", YAML_FILE],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    print(result.stderr)


    return {
        "status": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "image_path": "benchmarks/generated.png"
    }
