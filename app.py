from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image

# Load the trained model (replace with your model path)
model = torch.load(
    "model.pth",
    map_location=torch.device("cpu"),
)
model.eval()

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Jaundice Detection API!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Open and process the uploaded image
        image = Image.open(file.file)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image).argmax(dim=1).item()

        # Interpret the result
        result = "Positive for Jaundice" if prediction == 1 else "Negative for Jaundice"
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
