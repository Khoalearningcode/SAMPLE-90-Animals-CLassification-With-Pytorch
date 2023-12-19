from PIL import Image
import requests
from torchvision import transforms
from dataset import class_names, val_tf
from model import efficientnet_b0
import torch
def predict_with_url(url, model):
    with open('image.jpg', 'wb') as f:
        f.write(requests.get(url).content)

    tf = transforms.ToTensor()
    img = tf(Image.open('image.jpg')).unsqueeze(dim=0)
    model.eval()
    with torch.inference_mode():
        y_logits = model(img)
        y_pred = y_logits.argmax(dim=1)
    return class_names[y_pred]

def predict_with_path(img_path, model):
    img = Image.open(img_path)
    model.eval()
    with torch.inference_mode():
        y_pred = model(val_tf(img).unsqueeze(dim=0))
        idx = y_pred.argmax(dim=1)
    return class_names[idx]

