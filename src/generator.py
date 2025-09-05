import torch
from generator import Generator
import matplotlib.pyplot as plt
import os

latent_dim = 100
generator = Generator(latent_dim)
generator.load_state_dict(torch.load("outputs/models/generator.pth"))
generator.eval()

z = torch.randn(25, latent_dim)
gen_imgs = generator(z)

os.makedirs("outputs/generated_images", exist_ok=True)

fig, axs = plt.subplots(5,5, figsize=(5,5))
cnt = 0
for i in range(5):
    for j in range(5):
        axs[i,j].imshow(gen_imgs[cnt,0].detach().numpy(), cmap='gray')
        axs[i,j].axis('off')
        cnt += 1
plt.savefig("outputs/generated_images/sample.png")
plt.show()
