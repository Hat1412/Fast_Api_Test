from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}
