from utils.models import *
from config import *
from utils.data_loading import *
import matplotlib.pyplot as plt

train_df, test_df = get_dataframes()
_, test_transforms = get_segmentation_transforms()
dataset = segmentationDataset(dataFrame = train_df, transform = test_transforms)
image, mask = dataset[1]
image_ori = image.numpy()
mask_ori = mask.numpy().squeeze()
image_ori = image_ori.transpose((1,2,0))
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,12))
ax1.imshow(image_ori)
ax2.imshow(mask_ori)


image = image.unsqueeze(0)
image = image.to(DEVICE)

model = get_deep_lab_v3()

model.load_state_dict(torch.load('/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/segmentation/checkpoint_18.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
output = model(image)["out"][0]


output = torch.sigmoid(output)
output = (output > 0.6)*1.0
output = output.cpu().numpy()
output = np.where(output[0] == 1 , 1, 0)
print(output.shape)
print(output)
ax3.imshow(output)
plt.show()




