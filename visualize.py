import torch
import torchvision as tv
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

class SaveOutput():
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    def clear(self):
        self.outputs=[]
        save_output = SaveOutput()

save_output = SaveOutput()

model = torch.load("model save path")
model.eval()
model.cuda()
hook_handles = []
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

image = Image.open('./visual/cifar_test.jpg')
transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
X = transform(image).unsqueeze(dim=0).cuda()
out = model(X)

plt.figure(figsize = (30,30))
plt.imshow(tv.utils.make_grid(save_output.outputs[2].cpu().permute(1, 0, 2, 3),nrow=11)
           .permute(1, 2, 0))
plt.axis('off')
plt.savefig("./visual/fsfc3.jpg",bbox_inches='tight',pad_inches = 0)
plt.show()