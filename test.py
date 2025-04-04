from recon import ReconstructorTrainer
import torch
from PIL import Image
import numpy as np
from utils import resize_image
from arch import Decoder
import torch.optim as optim

img_path = r'D:\codePJ\RESEARCH\Evaluating-the-Robustness-of-Visual-Question-Answering-Systems\test\dog1.jpg'
image = Image.open(img_path)
image = resize_image(image, (128, 128)).cuda()
img = image.cpu().detach().numpy()[0]  
if img.shape[0] == 3:
    img = np.transpose(img, (1, 2, 0))
img = np.clip(img, 0, 1)
img = (img * 255).astype(np.uint8)
pil_img = Image.fromarray(img)

decoder = Decoder(latent_dim=768)
optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
reconstructor = ReconstructorTrainer(decoder=decoder, 
                                         optimizer=optimizer, scheduler=scheduler)
reconstructor.load_model(r"D:\codePJ\RESEARCH\Evaluating-the-Robustness-of-Visual-Question-Answering-Systems\pretrained\best_model.pth")
# model.eval()
print(f"Model {reconstructor}")
test_inference_vec = torch.randn(1, 768)
test_inf, res = reconstructor.eval(image.unsqueeze(0), test_inference_vec)
print("output ", test_inf.shape)

print("L2 distance:", res)
reconstructor.save_image(test_inf, "test_inf.png")
# reconstructor.save_model("test_model.pth")
