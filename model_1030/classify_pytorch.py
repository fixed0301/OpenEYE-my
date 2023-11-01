import numpy as np
import torch
from model_class import Model

def classify_img(image): # input은 한 쪽 눈 이미지
    model_path = 'model_fold_20.pth'
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 1 : closed, 0 : opened
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        output = model(image)

    predicted_class = output.argmax(dim=1).item()

    return predicted_class


