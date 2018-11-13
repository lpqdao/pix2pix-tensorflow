import os
import glob
import numpy as np
import cv2
import random

CROP_SIZE = 256


def get_data_files_list(data_dir, file_extension, read_recursively=False):
    """
    Read all data file names from the specified data directory

    :param data_dir: path to the directory
    :param file_extension: search for specific file extension
    :param read_recursively: recursively search in sub directories
    :return: list of sorted file names in specified directory
    """
    images_list = glob.glob(os.path.join(data_dir, "*." + file_extension), recursive=read_recursively)
    images_list.sort()

    return images_list


def generate_batch(data_files_list, arguments):
    """
    Generate a batch of data

    :param data_files_list: list of data file names
    :param arguments: batch_size, image_resize
    :return: numpy array of shape (batch_size, width, height, channels)
    """

    batch_files_list = np.random.choice(data_files_list, arguments.batch_size, replace=False)

    raw_data = []
    sketch_data = []
    for filename in batch_files_list:
        raw_img, sketch_img = read_input_data(filename, arguments)

        raw_data.append(raw_img)
        sketch_data.append(sketch_img)

    raw_np = np.array(raw_data)
    sketch_np = np.array(sketch_data)

    if arguments.which_direction == "AtoB":
        return raw_np, sketch_np
    elif arguments.which_direction == "BtoA":
        return sketch_np, raw_np
    else:
        raise Exception("invalid direction")


def read_input_data(filename, arguments):
    """
    Reads single input data

    :param filename: relative path to file
    :param arguments: image resize
    :return: left image, right image
    """

    cv_image = cv2.imread(filename)

    if cv_image is None:
        raise RuntimeError(f"Unable to open {filename}")

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = cv_image.astype("float32")

    cv_image /= 255.0

    # shape [height, width, channels]
    width = cv_image.shape[1]
    image_a = pre_process(cv_image[:, :width // 2, :])
    image_b = pre_process(cv_image[:, width // 2:, :])

    def transform(image):
        r = image

        # if arguments.flip:
        #     result = random.randint(0, 1)
        #     if result:
        #         r = cv2.flip(r, 0)

        r = cv2.resize(r, (arguments.scale_size, arguments.scale_size), interpolation=cv2.INTER_AREA)

        offset = np.int32(np.floor(np.random.uniform(0, arguments.scale_size - CROP_SIZE + 1, 2)))

        r = r[offset[0]:offset[0] + CROP_SIZE, offset[1]:offset[1] + CROP_SIZE]

        return r

    image_a = transform(image_a)
    image_b = transform(image_b)

    return image_a, image_b


def pre_process(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1

# data_files = get_data_files_list("tools/facades/train", "jpg")
# generate_batch(data_files, {"batch_size": 1, "scale_size": 286, "flip": True})
