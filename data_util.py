import os
import glob
import numpy as np
import cv2
import random
import collections
import tensorflow as tf
import math

CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")


def load_examples(arguments):
    # note~ this is known as the input pipeline
    if arguments.input_dir is None or not os.path.exists(arguments.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(arguments.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(arguments.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=arguments.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if arguments.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = pre_process_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = pre_process(raw_input[:, :width // 2, :])
            b_images = pre_process(raw_input[:, width // 2:, :])

    if arguments.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif arguments.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if arguments.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces arguments nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [arguments.scale_size, arguments.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, arguments.scale_size - CROP_SIZE + 1, seed=seed)),
                         dtype=tf.int32)
        if arguments.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif arguments.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                              batch_size=arguments.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / arguments.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def pre_process_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                    ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                    xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       arguments       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # arguments
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                    fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid arguments slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                    (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


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
