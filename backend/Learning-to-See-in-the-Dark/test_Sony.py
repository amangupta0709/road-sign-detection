#!venv/bin/python
# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37

from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from PIL import Image

input_dir = "./dataset/Sony/"
gt_dir = "./dataset/Sony/"
checkpoint_dir = "./checkpoint/Sony/"
result_dir = "./result_Sony/"

# get test IDs
##test_fns = glob.glob(gt_dir + '/1*.ARW')

test_fns = glob.glob(gt_dir + "*.dng")
print(test_fns[0])
test_ids = [int(os.path.basename(test_fn)[0:1]) for test_fn in test_fns]
print(test_ids[0])

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(
        tf.random.truncated_normal(
            [pool_size, pool_size, output_channels, in_channels], stddev=0.02
        )
    )
    deconv = tf.nn.conv2d_transpose(
        x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1]
    )

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(
        input, 32, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv1_1"
    )
    conv1 = slim.conv2d(
        conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv1_2"
    )
    pool1 = slim.max_pool2d(conv1, [2, 2], padding="SAME")

    conv2 = slim.conv2d(
        pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv2_1"
    )
    conv2 = slim.conv2d(
        conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv2_2"
    )
    pool2 = slim.max_pool2d(conv2, [2, 2], padding="SAME")

    conv3 = slim.conv2d(
        pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv3_1"
    )
    conv3 = slim.conv2d(
        conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv3_2"
    )
    pool3 = slim.max_pool2d(conv3, [2, 2], padding="SAME")

    conv4 = slim.conv2d(
        pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv4_1"
    )
    conv4 = slim.conv2d(
        conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv4_2"
    )
    pool4 = slim.max_pool2d(conv4, [2, 2], padding="SAME")

    conv5 = slim.conv2d(
        pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv5_1"
    )
    conv5 = slim.conv2d(
        conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv5_2"
    )

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(
        up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv6_1"
    )
    conv6 = slim.conv2d(
        conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv6_2"
    )

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(
        up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv7_1"
    )
    conv7 = slim.conv2d(
        conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv7_2"
    )

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv8_1")
    conv8 = slim.conv2d(
        conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv8_2"
    )

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv9_1")
    conv9 = slim.conv2d(
        conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope="g_conv9_2"
    )

    conv10 = slim.conv2d(
        conv9, 12, [1, 1], rate=1, activation_fn=None, scope="g_conv10"
    )
    out = tf.compat.v1.depth_to_space(conv10, 2)
    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)

    im = np.maximum(im - raw.black_level_per_channel[0], 0) / (
        1024 - raw.black_level_per_channel[0]
    )  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    R = im[
        0:H:2, 0:W:2, :
    ]  # Every alternating value starting from position (0,0) is red
    G = im[
        0:H:2, 1:W:2, :
    ]  # Every alternating value starting from position (0,1) is green
    B = im[
        1:H:2, 1:W:2, :
    ]  # Every alternating value starting from position (1,1) is blue
    G_e = im[
        1:H:2, 0:W:2, :
    ]  # Every alternating value starting from position (1,0) is green extra
    out = np.concatenate((B, G_e, R, G), axis=2)  # Always in R-G-B-G format
    return out


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.compat.v1.Session(config=config)
in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print("loaded " + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + "final/"):
    os.makedirs(result_dir + "final/")
print("AAAAAAAAAAAAAAAAA")
for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + "%d*.dng" % test_id)
    print(len(in_files))
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + "%d*.dng" % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        # in_exposure = float(in_fn[9:-5])
        # gt_exposure = float(gt_fn[9:-5])
        # ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        ##input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        input_full = np.expand_dims(pack_raw(raw), axis=0) * 100
        print("hello")
        im = raw.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
        )
        print("hello")
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        # scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)*100

        # gt_raw = rawpy.imread(gt_path)
        # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # print("hello")
        # gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)
        print("g")
        output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)

        output = output[0, :, :, :]
        # gt_full = gt_full[0, :, :, :]
        # scale_full = scale_full[0, :, :, :]
        # scale_full = scale_full * np.mean(gt_full) / np.mean(
        #     scale_full)  # scale the low-light image to the same mean of the groundtruth

        ##scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
        ##  result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        ##scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        ##   result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
        ##scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        ##  result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + "final/%d_out.png" % test_id
        )
        # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%d_out.png' % test_id)
        # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%d_out.png' % test_id)
