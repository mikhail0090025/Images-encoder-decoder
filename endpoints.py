from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import neural_net as nn
import io
from PIL import Image

app = FastAPI()

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root(request: Request):
    try:
        return templates.TemplateResponse("mainpage.html", {"request": request})
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.get("/test_image")
def root():
    try:
        image_pairs = nn.combine_image_pairs([
            (nn.tanh_to_img(nn.images[0]), nn.tanh_to_img(nn.code_and_decode(nn.images[0]))),
            (nn.tanh_to_img(nn.images[500]), nn.tanh_to_img(nn.code_and_decode(nn.images[500]))),
            (nn.tanh_to_img(nn.images[1000]), nn.tanh_to_img(nn.code_and_decode(nn.images[1000]))),
            (nn.tanh_to_img(nn.images[1500]), nn.tanh_to_img(nn.code_and_decode(nn.images[1500]))),
            (nn.tanh_to_img(nn.images[2000]), nn.tanh_to_img(nn.code_and_decode(nn.images[2000]))),
            (nn.tanh_to_img(nn.images[2500]), nn.tanh_to_img(nn.code_and_decode(nn.images[2500]))),
            (nn.tanh_to_img(nn.images[3000]), nn.tanh_to_img(nn.code_and_decode(nn.images[3000]))),
            (nn.tanh_to_img(nn.images[-1]), nn.tanh_to_img(nn.code_and_decode(nn.images[-1]))),
        ])
        img = Image.fromarray(image_pairs)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.post("/one_epoch")
def one_epoch_endpoint():
    try:
        nn.one_epoch(nn.full_encoder, nn.dataloader, nn.nn.MSELoss(), nn.optimizer, 0)
        return Response("One epoch passed")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.post("/five_epochs")
def five_epochs_endpoint():
    try:
        for i in range(5):
            nn.one_epoch(nn.full_encoder, nn.dataloader, nn.nn.MSELoss(), nn.optimizer, i)
        return Response("5 epochs passed")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.post("/ten_epochs")
def ten_epochs_endpoint():
    try:
        for i in range(10):
            nn.one_epoch(nn.full_encoder, nn.dataloader, nn.nn.MSELoss(), nn.optimizer, i)
        return Response("10 epochs passed")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

if __name__ == '__main__':
    import os
    os.system("uvicorn endpoints:app --host 0.0.0.0 --port 5000")