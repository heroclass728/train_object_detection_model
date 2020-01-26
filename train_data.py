import csv
import os
import cv2
import json
import glob

OBJ_LBLs = ['door', 'window', 'staircase']


def cvt_png2jpg(folder_path):

    paths = [os.path.join(folder_path, fn) for fn in os.listdir(folder_path) if os.path.splitext(fn)[1] == '.png']
    cnt = 0
    for path in paths:
        print(path)
        img = cv2.imread(path)
        new_jpg_path = os.path.splitext(path)[0] + '.jpg'
        cv2.imwrite(new_jpg_path, img)
        cnt += 1
    print(cnt)


def object_insert(x_min, x_max, y_min, y_max, obj_description):

    object_string = "\t" + "<object>" + "\n" + "\t" + "\t" + "<name>" + obj_description + "</name>" + "\n" + "\t" + \
                    "\t" + "<bndbox>" + "\n" + "\t" + "\t" + "\t" + "<xmin>" + str(x_min) + "</xmin>" + "\n" + "\t" + \
                    "\t" + "\t" + "<ymin>" + str(y_min) + "</ymin>" + "\n" + "\t" + "\t" + "\t" + "<xmax>" + \
                    str(x_max) + "</xmax>" + "\n" + "\t" + "\t" + "\t" + "<ymax>" + str(y_max) + "</ymax>" + "\n" + \
                    "\t" + "\t" + "</bndbox>" + "\n" + "\t" + "</object>" + "\n"

    return object_string


def close_tag(file_dir, filename):

    xml_file = file_dir + filename + ".xml"
    xml_data = open(xml_file, 'a')
    xml_data.write('</annotation>')
    xml_data.close()


def train_val_record(cur_dir):

    img_files = glob.glob(os.path.join(cur_dir, 'images/', '*.jpg'))

    train_val_path = os.path.join(cur_dir, 'annotations/trainval.txt')

    for i, img_filename in enumerate(img_files):

        filename = img_filename[img_filename.rfind("/") + 1:img_filename.rfind(".")]
        if i == 0:

            train_val = open(train_val_path, "w")
        else:

            train_val = open(train_val_path, "a")

        train_val.write(filename + "\n")
        train_val.close()


# convert training_data/Annotation/csv file into annotations/xmls and training_data/image/png files to images/jpeg files
def cvt_csv2xml(cur_dir):

    csv_file = os.path.join(cur_dir, 'training_data/Annotation/via_region_data (5).csv')
    png_img_dir = os.path.join(cur_dir, 'training_data/image/')
    jpg_img_dir = os.path.join(cur_dir, 'images')

    xml_file_dir = 'annotations/xmls/'

    csv_data = csv.reader(open(csv_file))

    prev_filename = []
    rf_value = True

    cnt = 0

    for i, row in enumerate(csv_data):

        if i == 0:
            continue

        img_filename = row[0]
        filename, _ = os.path.splitext(img_filename)
        xml_file = os.path.join(xml_file_dir, filename + ".xml")

        if row[5] != "{}":
            position = json.loads(row[5])
            obj_des = json.loads(row[6])

            x_min = position["x"]
            y_min = position["y"]
            x_max = x_min + position["width"]
            y_max = y_min + position["height"]
            obj_description = obj_des["description"]

            if obj_description not in OBJ_LBLs:

                if obj_description.find('door') != -1:
                    obj_description = 'door'
                elif obj_description.find('window') != -1:
                    obj_description = 'window'
                elif obj_description.find('staircase') != -1:
                    print(obj_description)
                    obj_description = 'staircase'


            png_img_path = os.path.join(png_img_dir, filename + ".png")
            img = cv2.imread(png_img_path)
            img_width = img.shape[1]
            img_height = img.shape[0]

            # ----------------resize image file into 1000*1000 at most
            fx = 1000 / img_width
            fy = 1000 / img_height

            if fx < 1.0 or fy < 1.0:
                f = min(fx, fy)
            else:
                f = 1.0

            dim = (int(img_width * f), int(img_height * f))
            img = cv2.resize(img, dim)
            # ----------------------------------------------------------

            if filename != prev_filename:

                if prev_filename != [] and rf_value:

                    close_tag(xml_file_dir, prev_filename)

                jpg_img_path = os.path.join(jpg_img_dir, filename + ".jpg")
                cv2.imwrite(jpg_img_path, img)

                xml_data = open(xml_file, 'w')
                xml_data.write('<annotation>' + "\n")
                xml_data.write("\t" + '<folder>' + "less_selected" + '</folder>' + "\n")
                xml_data.write("\t" + '<filename>' + filename + ".jpg" + '</filename>' + "\n")
                xml_data.write("\t" + '<size>' + "\n")
                xml_data.write("\t" + "\t" + '<width>' + str(int(img_width * f)) + '</width>' + "\n")
                xml_data.write("\t" + "\t" + '<height>' + str(int(img_height * f)) + '</height>' + "\n")
                xml_data.write("\t" + '</size>' + "\n")
                xml_data.write("\t" + '<segmented>' + "0" + '</segmented>' + "\n")
                # (np.array([x_min, x_max, y_max, y_min]) * f).tolist()

                xml_data.write(object_insert(int(x_min * f), int(x_max * f), int(y_min * f), int(y_max * f),
                                             obj_description))
                xml_data.close()
                rf_value = True

            else:

                xml_data = open(xml_file, 'a')
                xml_data.write(object_insert(int(x_min * f), int(x_max * f), int(y_min * f), int(y_max * f),
                                             obj_description))
                xml_data.close()
                rf_value = True

            cnt += 1

        else:
            rf_value = False

            if prev_filename:

                close_tag(xml_file_dir, prev_filename)

        prev_filename = filename

    if rf_value:

        close_tag(xml_file_dir, prev_filename)


if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))

    cvt_csv2xml(current_dir)
    train_val_record(current_dir)
