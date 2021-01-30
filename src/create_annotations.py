import cv2

image_id = 0


def find_contours(sub_mask):
    gray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def create_image_annotation(file_name, width, height):
    global image_id
    image_id += 1
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images


def create_annotation_format(contour, image_id_, category_id, annotation_id):
    annotation = {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id_,
        "category_id": category_id,
        "bbox": cv2.boundingRect(contour),
        "area": cv2.contourArea(contour),
        "segmentation": [contour.flatten().tolist()],
    }
    return annotation


def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }
    return coco_format
