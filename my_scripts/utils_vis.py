import numpy as np
import matplotlib.pyplot as plt

# get the masked image
def get_masked_image(frame_RGB, mask):
    if not (mask==True).any():
        print("No mask found, skipping frame")
        return None
    # masked_image = torch.tensor(frame_RGB.copy(), device="cuda").permute(2, 0, 1)  # [3, 360, 800]
    # masked_rgb_image = frame_RGB * mask[..., None] 
    # bbox = bbox1
    # x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
    # x_max, y_max = int(bbox[1][0]), int(bbox[1][1])
    # x_min = max(0, min(x_min, width-1))
    # y_min = max(0, min(y_min, height-1))
    # x_max = max(0, min(x_max, width-1))
    # y_max = max(0, min(y_max, height-1))
    # bbox_region = masked_image[:, y_min:y_max, x_min:x_max]
    # bbox_region_np = bbox_region.permute(1, 2, 0).cpu().numpy()

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Crop both RGB and mask to bounding box
    cropped_rgb = frame_RGB[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

    masked_rgb_cropped = cropped_rgb * cropped_mask[:, :, None]
    # plt.imshow(masked_rgb_cropped)
    return masked_rgb_cropped


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_bbox(bbox, ax, marker_size=200):
    tl, br = bbox[0], bbox[1]
    w, h = (br - tl)[0], (br - tl)[1]
    x, y = tl[0], tl[1]
    print(x, y, w, h)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))