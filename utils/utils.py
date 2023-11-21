import numpy as np
from PIL import Image, ImageDraw

def save_sample(img, target, evaluate_result, path, save=True):
    """
    Show a sample image
    :param img: Tensor of shape (3, H, W)
    :param target: target dictionary. Should have 'boxes' as key
    :param show: (boolean) if True show the image
    :return: PIL image of the sample with bounding boxes
    """
    img_arr = img.numpy()
    img_arr = img_arr.transpose((1, 2, 0)) * 255
    img_arr = img_arr.astype(np.uint8)

    image = Image.fromarray(img_arr)
    bboxes = target['boxes'].numpy()
    bboxes = np.ceil(bboxes)

    draw = ImageDraw.Draw(image)
    for bbox_id in range(bboxes.shape[0]):
        bbox = list(bboxes[bbox_id, :])
        if evaluate_result[bbox_id]['score'] > 0.45:
            draw.rectangle(bbox, outline=(255, 201, 14), width=2)

    if save:
        image.save(path)

    return image


def relu(x):
    return max(x, 0)


def collate_fn(batch):
    return tuple(zip(*batch))