import torch


model = torch.hub.load(
            'facebookresearch/dino:main', 'dino_vits16')
model.cuda()

a = torch.ones((16, 3, 224, 224), dtype=torch.float32, device='cuda')

a = torch.load('/ws/f_tensor.pth')
b = model(a)

print(b.shape)