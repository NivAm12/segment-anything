import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import wandb
from tqdm import tqdm


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def test_iou():
    loss_list = []
    columns = ["image", "pred_mask", "gt_mask", "iou"]
    test_data = []

    current_run = wandb.init(
        project="sam",
        name=f"busi_dataset_iou",
        config={
            "model": "sam",
            "vit": "vit_h",
            "dataset": "Dataset_BUSI_with_GT",
            "num_examples": 100,
            "loss": "iou"
        }
    )

    for i in tqdm(range(1, 101)):
        image = cv2.imread(
            f'/home/projects/yonina/SAMPL_training/public_datasets/Dataset_BUSI_with_GT/malignant/malignant ({i}).png',
            cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        gt_mask = cv2.imread(
            f'/home/projects/yonina/SAMPL_training/public_datasets/Dataset_BUSI_with_GT/malignant/malignant ({i})_mask.png',
            cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)

        mask_generator = load_mask_generator()
        masks = mask_generator.generate(image)

        best_mask = None
        best_iou = 1.1

        for mask in masks:
            iou = iou_loss(mask['segmentation'], gt_mask)

            if iou < best_iou:
                best_mask = mask['segmentation']
                best_iou = iou

        loss_list.append(best_iou)
        test_data.append([wandb.Image(image), wandb.Image(gt_mask), wandb.Image(best_mask),
                          best_iou])

    images_table = wandb.Table(columns=columns, data=test_data)
    current_run.log({"results": images_table})

    avg_loss = sum(loss_list) / len(loss_list)
    current_run.log({"average iou": avg_loss})


def load_mask_generator():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


def iou_loss(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    loss = 1 - iou

    return loss


def plot_images(original, predicted, target, loss):
    # Create a figure with two subplots, side by side
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax2.imshow(target)
    ax2.set_title('Gt mask')
    ax3.imshow(predicted)
    ax3.set_title('Predicted mask')
    ax3.set_title('Predicted Image\nLoss: {:.2f}'.format(loss))

    plt.show()


if __name__ == '__main__':
    test_iou()
