import cv2

image_id = 0


def find_contours(sub_mask):
    gray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


def create_category_annotation(category_dict):
    category_list = []
    for key, value in category_dict.items():
        category = {"id": value, "name": key, "supercategory": key}
        category_list.append(category)
    return category_list


def create_image_annotation(file_name, width, height):
    global image_id
    image_id += 1
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }


def create_annotation_format(contour, image_id_, category_id, annotation_id):
    return {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id_,
        "category_id": category_id,
        "bbox": cv2.boundingRect(contour),
        "area": cv2.contourArea(contour),
        "segmentation": [contour.flatten().tolist()],
    }


def get_coco_json_format():
    return {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}],
    }
