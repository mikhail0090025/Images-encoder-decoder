from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import neural_net as nn
import io
from PIL import Image
import plotly.graph_objects as go

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
def test_image_endpoint():
    try:
        image_pairs = nn.combine_image_pairs([
            (nn.tanh_to_img(nn.images[0]), nn.tanh_to_img(nn.code_and_decode(nn.images[0]))),
            (nn.tanh_to_img(nn.images[100 * 4]), nn.tanh_to_img(nn.code_and_decode(nn.images[100 * 4]))),
            (nn.tanh_to_img(nn.images[200 * 4]), nn.tanh_to_img(nn.code_and_decode(nn.images[200 * 4]))),
            (nn.tanh_to_img(nn.images[300 * 4]), nn.tanh_to_img(nn.code_and_decode(nn.images[300 * 4]))),
            (nn.tanh_to_img(nn.images[400 * 4]), nn.tanh_to_img(nn.code_and_decode(nn.images[400 * 4]))),
            (nn.tanh_to_img(nn.images[500 * 4]), nn.tanh_to_img(nn.code_and_decode(nn.images[500 * 4]))),
            (nn.tanh_to_img(nn.images[600 * 4]), nn.tanh_to_img(nn.code_and_decode(nn.images[600 * 4]))),
            (nn.tanh_to_img(nn.images[700 * 4]), nn.tanh_to_img(nn.code_and_decode(nn.images[700 * 4]))),
            (nn.tanh_to_img(nn.images[-1]), nn.tanh_to_img(nn.code_and_decode(nn.images[-1]))),
        ])
        img = Image.fromarray(image_pairs)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.get("/get_graphics")
def get_graphics_endpoint():
    try:
        # Создаём график
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in range(len(nn.all_losses))],
            y=nn.all_losses,
            mode='lines+markers',
            name='Loss'
        ))
        fig.update_layout(
            title="Loss Graph",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark"
        )

        # Конвертируем в JSON
        graph_json = fig.to_json()
        return {"graph": graph_json}
    except Exception as e:
        return Response(f"Unexpected error has occured: {e}")

@app.post("/one_epoch")
def one_epoch_endpoint():
    try:
        nn.one_epoch(nn.full_encoder, nn.dataloader, nn.nn.MSELoss(), nn.optimizer, 0)
        return Response("One epoch passed")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.post("/one_batch")
def one_batch_endpoint():
    try:
        nn.one_batch(nn.full_encoder, nn.dataloader, nn.nn.MSELoss(), nn.optimizer)
        return Response("One batch passed")
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

@app.post("/30_epochs")
def ten_epochs_endpoint():
    try:
        for i in range(30):
            nn.one_epoch(nn.full_encoder, nn.dataloader, nn.nn.MSELoss(), nn.optimizer, i)
        return Response("10 epochs passed")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.get("/random_image")
def random_image_endpoint():
    random_latent = (np.random.random(nn.full_encoder.all_layers[1].latent_dim) * 2) - 1
    img = nn.full_encoder.image_from_latent(random_latent)
    return Response(content=nn.array_to_image(img), media_type="image/png") 

import matplotlib.pyplot as plt
@app.get("/representation_graphics")
def representation_graphics_endpoint():
    latents = nn.get_all_latents()
    plt.figure(figsize=(10, 8))
    plt.scatter(latents[:, 0], latents[:, 1])  # Assuming hidden_dim is 2 for visualization
    plt.title("Encoded Representations")
    plt.xlabel("Encoded Dimension 1")
    plt.ylabel("Encoded Dimension 2")
    plt.show()

    return Response("") 

from torchvision import transforms
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import requests

@app.get("/process_image")
async def process_image(url: str):
    try:
        transform = transforms.Compose([
            transforms.Resize((100, 100)),  # Сжатие до 100x100
            transforms.ToTensor(),          # Преобразование в тензор
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация в [-1, 1]
        ])
        # Скачиваем изображение по URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Проверяем, что запрос успешен
        img = Image.open(io.BytesIO(response.content)).convert('RGB')

        # Преобразуем в тензор
        img_tensor = transform(img).unsqueeze(0).to(nn.device)  # (1, 3, 100, 100)

        # Прогоняем через модель
        with nn.torch.no_grad():
            output_tensor = nn.full_encoder(img_tensor)  # (1, 3, 100, 100)

        # Преобразуем обратно в изображение
        output_tensor = output_tensor.squeeze(0).cpu().numpy()  # (3, 100, 100)
        output_tensor = (output_tensor + 1) * 127.5  # Денармализация из [-1, 1] в [0, 255]
        output_img = Image.fromarray(output_tensor.astype(np.uint8).transpose(1, 2, 0))

        # Сохраняем в буфер и отправляем
        img_byte_arr = io.BytesIO()
        output_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(
            io.BytesIO(img_byte_arr.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=reconstructed_image.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    import os
    os.system("uvicorn endpoints:app --host 0.0.0.0 --port 5000")