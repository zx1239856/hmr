"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import time

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
from cam import open_cam_onboard

import cv2

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    #rend_img_overlay = renderer(
    #    vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    #rend_img = renderer(
    #    vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    #rend_img_vp1 = renderer.rotated(
    #    vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    #rend_img_vp2 = renderer.rotated(
    #    vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])
    cv2.imshow('input',img)
    cv2.imshow('joint projection',skel_img)
    #cv2.imshow('3D Mesh overlay',rend_img_overlay)
    #cv2.imshow('3D mesh',rend_img)
    #cv2.imshow('diff vp',rend_img_vp1)
    #cv2.imshow('diff vp 2',rend_img_vp2)
    cv2.waitKey(25)
    """
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()
    """

def preprocess_image(img, json_path=None):
    #img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, json_path=None):
    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    count = 0
    st = time.time()
    cap = open_cam_onboard(640, 480)
    while True:
        image_np = cv2.imread('data/im1963.jpg')
        # print(image_np)
        if image_np is None:
            print("cannot read image")
            break
            
        # cv2.namedWindow('origin_img', cv2.WINDOW_NORMAL)
        # cv2.imshow('origin_img', image_np)
        # cv2.waitKey(3000)
        # cv2.destroyAllWindows()

        # ret, image_np = cap.read()
        # print('\n\nshape', image_np.shape)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        #cv2.imshow('p',image_np)
        input_img, proc_param, img = preprocess_image(image_np, json_path)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)
        joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)
        #print(joints,verts,cams,joints3d,theta)
        visualize(img, proc_param, joints[0], verts[0], cams[0])
        count +=1
        if(not count%30):
            st = time.time()
            count = 0
        if(count<5):
            continue
        print('FPS: %f'%(count/(time.time()-st)))
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
