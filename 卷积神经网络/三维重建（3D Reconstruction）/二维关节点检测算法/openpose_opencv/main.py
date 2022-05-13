#!/usr/bin/python3
# !--*-- coding: utf-8 --*--
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class general_mulitpose_model(object):
    def __init__(self):
        self.point_names = ['Nose', 'Neck',
                            'R-Sho', 'R-Elb', 'R-Wr',
                            'L-Sho', 'L-Elb', 'L-Wr',
                            'R-Hip', 'R-Knee', 'R-Ank',
                            'L-Hip', 'L-Knee', 'L-Ank',
                            'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
        self.point_pairs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                            [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                            [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                            [2, 17], [5, 16]]

        # index of pafs correspoding to the self.point_pairs
        # e.g for point_pairs(1,2), the PAFs are located at indices (31,32) of output,
        #   Similarly, (1,5) -> (39,40) and so on.
        self.map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                        [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                        [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
                        [37, 38], [45, 46]]

        self.colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255],
                       [0, 100, 255], [0, 255, 255], [0, 100, 255],
                       [0, 255, 0], [255, 200, 100], [255, 0, 255],
                       [0, 255, 0], [255, 200, 100], [255, 0, 255],
                       [0, 0, 255], [255, 0, 0], [200, 200, 0],
                       [255, 0, 0], [200, 200, 0], [0, 0, 0]]

        self.num_points = 18
        self.pose_net = self.get_model()

    def get_model(self):
        prototxt = "./models/pose/body_25/pose_deploy.prototxt"
        caffemodel = "./models/pose/body_25/pose_iter_584000.caffemodel"
        coco_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_net

    def getKeypoints(self, probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)

        keypoints = []
        # find the blobs
        contours, hierarchy = cv2.findContours(mapMask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints

    def getValidPairs(self, output, detected_keypoints, img_width, img_height):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7

        for k in range(len(self.map_idx)):
            # A->B constitute a limb
            pafA = output[0, self.map_idx[k][0], :, :]
            pafB = output[0, self.map_idx[k][1], :, :]
            pafA = cv2.resize(pafA, (img_width, img_height))
            pafB = cv2.resize(pafB, (img_width, img_height))

            # Find the keypoints for the first and second limb
            candA = detected_keypoints[self.point_pairs[k][0]]
            candB = detected_keypoints[self.point_pairs[k][1]]
            nA = len(candA)
            nB = len(candB)

            if (nA != 0 and nB != 0):
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(
                            zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(
                                round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(
                                                   round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        # Check if the connection is valid
                        if (len(np.where(paf_scores > paf_score_th)[
                                    0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1

                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair,
                                               [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])

        return valid_pairs, invalid_pairs

    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs, keypoints_list):
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(self.map_idx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:, 0]
                partBs = valid_pairs[k][:, 1]
                indexA, indexB = np.array(self.point_pairs[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[
                                                                   partBs[i].astype(int), 2] + \
                                                               valid_pairs[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + \
                                  valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])

        return personwiseKeypoints

    def predict(self, inputparam):
        img_cv2 = cv2.imread(inputparam)
        img_width, img_height = img_cv2.shape[1], img_cv2.shape[0]

        net_height = 368
        net_width = int((net_height / img_height) * img_width)

        start = time.time()
        in_blob = cv2.dnn.blobFromImage(
            img_cv2,
            1.0 / 255,
            (net_width, net_height),
            (0, 0, 0),
            swapRB=False,
            crop=False)

        self.pose_net.setInput(in_blob)
        output = self.pose_net.forward()
        print("[INFO]Time Taken in Forward pass: {}".format(time.time() - start))

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1
        for part in range(self.num_points):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (img_cv2.shape[1], img_cv2.shape[0]))
            keypoints = self.getKeypoints(probMap, threshold)
            print("Keypoints - {} : {}".format(self.point_names[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = \
            self.getValidPairs(output,
                               detected_keypoints,
                               img_width,
                               img_height)
        personwiseKeypoints = \
            self.getPersonwiseKeypoints(valid_pairs,
                                        invalid_pairs,
                                        keypoints_list)

        return personwiseKeypoints, keypoints_list

    def vis_pose(self, img_file, personwiseKeypoints, keypoints_list):
        img_cv2 = cv2.imread(img_file)
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(self.point_pairs[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(img_cv2, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)

        plt.figure()
        plt.imshow(img_cv2[:, :, ::-1])
        plt.title("Results")
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    pic = "/home/stu/PycharmProjects/opscoco/11_img.jpg"
    print("[INFO]MultiPose estimation.")
    img_file = "11_img.jpg"

    start = time.time()
    multipose_model = general_mulitpose_model()
    print("[INFO]Time Taken in Model Loading: {}". \
          format(time.time() - start))
    personwiseKeypoints, keypoints_list = \
        multipose_model.predict(pic)
    multipose_model.vis_pose(img_file,
                             personwiseKeypoints,
                             keypoints_list)
    print(personwiseKeypoints)
    print(keypoints_list)
    print("[INFO]Done.")