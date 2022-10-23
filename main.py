
from utils.vizualisation import *
from config import *
from utils.data_loading import *



if TASK == 1:
    image_to_infer_N = 12
    train_df, test_df = get_dataframes()
    _, test_transforms = get_segmentation_transforms()
    dataset = segmentationDataset(dataFrame = test_df, transform = test_transforms)
    image, mask = dataset[image_to_infer_N ]
    output_mask = inference_deep_lab(image_m = image, checkpoint_path = CHECKPOINT_FOR_INFERENCE)
    vizualize_segmentation_output(image_ori = image, mask_ori = mask, mask = output_mask)

if TASK == 2:

    image_to_infer_N = 200

    train_df, test_df  = get_dataframes()

    train_t, test_t = get_detection_transforms()
    dataset = DetectionDataset(dataFrame = train_df, transform = test_t, labels_map = LABELS_MAP)
    image, target = dataset[image_to_infer_N]

    out = inference_faster_rcnn(checkpoint_path=CHECKPOINT_FOR_INFERENCE, image_ori = image)

    draw_detection_output(image_ori = image, target = target, output = out)

if TASK == 3:
    image_to_infer_N = 12

    train_df, test_df = get_instance_segmentation_dataframes()
    train_t, test_t = get_detection_transforms()
    md = InstanceSegmentationDataset(test_df, transforms = test_t, labels_map = LABELS_MAP)
    i = 12
    image_m, target = md[i]
    out = inference_mask_rcnn(image_m = image_m, checkpoint_path = CHECKPOINT_FOR_INFERENCE, threshold = THRESHOLD)
    draw_instance_inference_and_original(original_image = image_m, original_target = target, out_target =out)





'''
if TASK==1:
    model, params = get_deep_lab_v3()
    train_dl, test_dl = get_segmentation_data_loaders()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(params, lr=LR)
    train_segmentation(model = model, optimizer=optimizer, trainDataLoader= train_dl, testDataLoader = test_dl, criterion = criterion)

if TASK == 2:
    model, params = get_faster_rcnn()
    train_dl, test_dl = get_detection_data_loaders()
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    train_detector(model = model, optimizer = optimizer, trainLoader = train_dl, testLoader = test_dl)

if TASK == 3:
    model, params = get_mask_rcnn()
    train_dl, test_dl = get_instance_segmentation_data_loaders()
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    train_instance(model=model, trainLoader=train_dl, testLoader=test_dl, optimizer=optimizer)


'''