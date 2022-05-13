import os

import cv2
import argparse
from openpose import Openpose, save_keypoints, draw_person_pose


class general_pose_model(object):
    def __init__(self, modelpath, mode="COCO"):
        # 指定采用的模型
        #   Body25: 25 points
        #   COCO:   18 points
        #   MPI:    15 points
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0
        if mode == "BODY25":
            self.pose_net = self.general_body25_model(modelpath)
        elif mode == "COCO":
            self.pose_net = self.general_coco_model(modelpath)
        elif mode == "MPI":
            self.pose_net = self.get_mpi_model(modelpath)


    def get_mpi_model(self, modelpath):
        self.points_name = {
            "Head": 0, "Neck": 1,
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist":
            7, "RHip": 8, "RKnee": 9, "RAnkle": 10,
            "LHip": 11, "LKnee": 12, "LAnkle": 13,
            "Chest": 14, "Background": 15 }
        self.num_points = 15
        self.point_pairs = [[0, 1], [1, 2], [2, 3],
                            [3, 4], [1, 5], [5, 6],
                            [6, 7], [1, 14],[14, 8],
                            [8, 9], [9, 10], [14, 11],
                            [11, 12], [12, 13]
                            ]
        prototxt = os.path.join(
            modelpath,
            "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        caffemodel = os.path.join(
            modelpath,
            "pose/mpi/pose_iter_160000.caffemodel")
        mpi_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return mpi_model


    def general_coco_model(self, modelpath):
        self.points_name = {
            "Nose": 0, "Neck": 1,
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7,
            "RHip": 8, "RKnee": 9, "RAnkle": 10,
            "LHip": 11, "LKnee": 12, "LAnkle": 13,
            "REye": 14, "LEye": 15,
            "REar": 16, "LEar": 17,
            "Background": 18}
        self.num_points = 18
        self.point_pairs = [[1, 0], [1, 2], [1, 5],
                            [2, 3], [3, 4], [5, 6],
                            [6, 7], [1, 8], [8, 9],
                            [9, 10], [1, 11], [11, 12],
                            [12, 13], [0, 14], [0, 15],
                            [14, 16], [15, 17]]
        prototxt   = os.path.join(
            modelpath,
            "pose/coco/pose_deploy_linevec.prototxt")
        caffemodel = os.path.join(
            modelpath,
            "pose/coco/pose_iter_440000.caffemodel")
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model


    def general_body25_model(self, modelpath):
        self.num_points = 25
        self.point_pairs = [[1, 0], [1, 2], [1, 5],
                            [2, 3], [3, 4], [5, 6],
                            [6, 7], [0, 15], [15, 17],
                            [0, 16], [16, 18], [1, 8],
                            [8, 9], [9, 10], [10, 11],
                            [11, 22], [22, 23], [11, 24],
                            [8, 12], [12, 13], [13, 14],
                            [14, 19], [19, 20], [14, 21]]
        prototxt   = os.path.join(
            modelpath,
            "pose/body_25/pose_deploy.prototxt")
        caffemodel = os.path.join(
            modelpath,
            "pose/body_25/pose_iter_584000.caffemodel")
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model


    def predict(self, imgfile):
        img_cv2 = cv2.imread(imgfile)
        img_height, img_width, _ = img_cv2.shape
        inpBlob = cv2.dnn.blobFromImage(img_cv2,
                                        1.0 / 255,
                                        (self.inWidth, self.inHeight),
                                        (0, 0, 0),
                                        swapRB=False,
                                        crop=False)
        self.pose_net.setInput(inpBlob)
        self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        output = self.pose_net.forward()

        H = output.shape[2]
        W = output.shape[3]
        print(output.shape)

        # vis heatmaps
        self.vis_heatmaps(img_file, output)

        #
        points = []
        for idx in range(self.num_points):
            probMap = output[0, idx, :, :] # confidence map.

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H

            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        return points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img_path', '-i', help='image file path')
    parser.add_argument('--precise', '-p', action='store_true', help='do precise inference')
    args = parser.parse_args()

    # load model
    openpose = Openpose(weights_file = args.weights, training = False)

    # read image
    img_names = os.listdir(args.img_path)
    for img in img_names:
        im = cv2.imread(args.img_path + '/' + img)

        # inference
        poses, _ = openpose.detect(im, precise=args.precise)

        # save_keypoints('/home/stu/2Dto3D/pose2smplx/frankmocap-main/dataset/lsp-json', str(img).split('.')[0], poses)
        # draw and save image
        im = draw_person_pose(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), poses)

        print('Saving result into result.png...')
        cv2.imwrite('result.png', im)