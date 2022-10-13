from utils.data_loading import get_data_loaders



train, test = get_data_loaders()


for i, (image_train, mask_train) in enumerate(train):
  break

for i, (image_test, mask_test) in enumerate(test):
  break


print('train')
print(f'image shape {image_train.shape}')
print(f'mask shape {mask_train.shape}')

print('test')
print(f'image shape {image_test.shape}')
print(f'mask shape {mask_test.shape}')
'''

model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True, n_classes  = 2)


#pytorch_total_params = sum(p.numel() for p in model.parameters())


for name, child in model.named_children():
  print(f'name is: {name}')
  print(f'module is: {child}')


# freeze the backbone (it will freeze the body and fpn params)
for p in model.backbone.parameters():
  p.requires_grad = False

# freeze the fc6 layer in roi_heads
for p in model.roi_heads.box_head.fc6.parameters():
  p.requires_grad = False
'''