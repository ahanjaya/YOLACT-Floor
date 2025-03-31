import torch

# path = 'weights/resnet18-5c106cde.pth'
# path = 'weights/resnet34-333f7ec4.pth'
# path = 'weights/resnet50-19c8e357.pth'
# path = 'weights/resnet101_reducedfc.pth'
path = 'weights/resnet152-b121ed2d.pth'

state_dict = torch.load(path)
print(path)
print('len dict: {}'.format(len(state_dict)))

keys = list(state_dict)
for key in keys:
    if key.startswith('layer'):
        idx = int(key[5])
        new_key = 'layers.' + str(idx-1) + key[6:]
        state_dict[new_key] = state_dict.pop(key)

# print(state_dict)

for i in state_dict:
    size_tensor = state_dict[i].size()
    print('{}: \t{}'.format(i, size_tensor))