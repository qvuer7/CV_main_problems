import os

annotations = 'data_segmentation/annotations/'
ans = os.listdir(annotations)
for i in ans:
    c = i.replace('_refined_mask', '')
    os.rename(annotations + '/' + i, annotations+'/'+c)


