# Beatforge

## Prerequisites

- A PC that can handle running AI locally
  - A GPU with CUDA support and 16 GB memory
  - run through CPU, but may be slower, more taxing, and a lessar output
 
- Docker Desktop or equivilant to run docker compose
- Python 3.9 or 3.10
- Golang 1.22

## Running
- To run for first time, make sure Docker Desktop is running then:

`docker compose up --build`

- After model is built, run `docker compose up`

## Prompting
- Two ways to prompt
  - Run a curl POST, example:
    `curl -X POST http://localhost:8001/generate   -H "Content-Type: application/json"   -d '{
    "prompt": "A cinematic orchestral theme with strings and drums",
    "outpath": "orchestral_theme.wav",
    "duration": 30
  }'`
  - clone [beatforge-ui](https://github.com/Tim-McCann/beatforge-ui) and run to prompt through UI
