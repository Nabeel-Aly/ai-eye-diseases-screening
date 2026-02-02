from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from src.api.inference import predict

app = FastAPI(
    title="AI Eye Disease Screener",
    description="AI-powered eye disease screening system",
    version="1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="src/api/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict(image)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result
        }
    )
