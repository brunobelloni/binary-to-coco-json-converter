import glob
import json
import os

from skimage import io

from src.create_annotations import *

# Label ids of the dataset
category_ids = {
    "EX": 0,
    "HE": 1,
    "SE": 2,
    "MA": 3,
}


# Get "images" and "annotations" info
def images_annotations_info(maskpath):
    annotation_id = 0
    annotations = []
    images = []

    for mask_image in glob.glob(os.path.join(maskpath, '**', "*.tif")):
        original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = io.imread(mask_image)
        w, h = mask_image_open.shape

        # "images" info
        if original_file_name not in map(lambda img: img['file_name'], images):
            image = create_image_annotation(original_file_name, w, h)
            images.append(image)
        else:
            image = [element for element in images if element['file_name'] == original_file_name][0]

        # sub_masks = create_sub_masks(mask_image_open, w, h)
        category_id = category_ids[mask_image.split('/')[2]]

        # "annotations" info
        polygons, segmentations = create_sub_mask_annotation(mask_image_open)

        for polygon in polygons:
            # Cleaner to recalculate this variable
            segmentation = [np.array(polygon.exterior.coords).ravel().tolist()]
            annotation = create_annotation_format(polygon, segmentation, image['id'], category_id, annotation_id)
            annotations.append(annotation)
            annotation_id += 1
    return images, annotations, annotation_id


if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    for keyword in ["valid", "test", "train"]:
        mask_path = "dataset/{}_mask/".format(keyword)

        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)

        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        with open("output/{}.json".format(keyword), "w") as outfile:
            json.dump(coco_format, outfile)

        print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))
