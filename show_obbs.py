import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import math
import numpy as np


def show_obb():
    image_file_path = ""
    label_file_path = ""

    label_extension = Path(label_file_path).suffix

    if label_extension == ".xml":
        boxes = parse_xml(label_file_path)       
    elif label_extension == ".txt":
        boxes = parse_txt(label_file_path)

    image = cv2.imread(image_file_path)

    for box in boxes:
        box = np.array(box[:4], np.int32)
        print(box)
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_xml(xml_label_file_path):
    boxes = []
    tree = None
    with open(xml_label_file_path, "r") as label_file:
        tree = ET.parse(label_file)
    
    for obj in tree.getroot().iter("object"):
        xmlbox = obj.find("robndbox")
        cx = float(xmlbox.find("cx").text)
        cy = float(xmlbox.find("cy").text)
        w = float(xmlbox.find("w").text)
        h = float(xmlbox.find("h").text)
        angle = float(xmlbox.find("angle").text)
        name = obj.find("name").text

        x1 = cx + (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle)
        y1 = cy + (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)
        x2 = cx - (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle)
        y2 = cy - (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)
        x3 = cx - (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle)
        y3 = cy - (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle)
        x4 = cx + (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle)
        y4 = cy + (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle)

        boxes.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4), name])

    return boxes


def parse_txt(txt_label_file_path):
    boxes = []
    entries = None
    with open(txt_label_file_path, "r") as label_file:
        entries = label_file.read().splitlines()

    for entry in entries:
        sp = entry.split()
        if len(sp) < 9:
            continue

        b = [float(v) for v in sp[:8]]
        boxes.append([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7]), sp[8]])

    return boxes


if __name__ == "__main__":
    show_obb()
