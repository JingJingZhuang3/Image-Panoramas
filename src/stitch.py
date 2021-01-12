import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
from PIL import Image


class ImageInfo:
    def __init__(self, img):
        self.img = img
        self.key, self.des = sift_features(img)
        self.shape = img.shape

    def show(self):
        plt.imshow(self.img)
        plt.show()


class KnnInfo:
    dist = -1
    trainIdx = -1
    queryIdx = -1


# SIFT source from: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
def sift_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # key will be a list of keypoints, des is a numpy array of shape
    key, des = sift.detectAndCompute(img, None)
    return key, des


def euclidean_distance(row1, row2):
    dis = 0.0
    """ too slow
    for i in range(len(row1) - 1):
        dis += np.square(row1[i] - row2[i])"""
    dis = LA.norm(row1 - row2)
    return dis


# Reference from https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
def k_nearest_neighbors(des1, des2):
    match = []
    for x in range(len(des1)):
        knn_match = KnnInfo()
        knn_match.trainIdx = x
        for y in range(len(des2)):
            # calculate distance
            dis = euclidean_distance(des1[x], des2[y])
            if y == 0:
                knn_match.dist = dis
                knn_match.queryIdx = y
            if knn_match.dist > dis:
                knn_match.dist = dis
                knn_match.queryIdx = y
        # distance.sort(key=lambda tup: tup[1])
        # get neighbors
        check = True
        for i in range(len(des1)):
            dis = euclidean_distance(des2[knn_match.queryIdx], des1[i])
            if knn_match.dist > dis:
                check = False
        if check:
            match.append(knn_match)
    match.sort(key=lambda tup: tup.dist)
    return match


# this only sort two images
def sort2(img, key1, key2, match):
    img1 = img[0]
    img2 = img[1]
    h1, w1 = img1.shape[:2]  # Remind - H: height, W: width
    h2, w2 = img2.shape[:2]  # img.shape contains [height, width, channels]
    test_w1 = w1 / 5
    test_w2 = w2 / 5
    cnt1 = 0
    cnt2 = 0
    order_change = False
    if len(match) > 4:
        for x in range(len(match)):
            if key1[match[x].trainIdx][0] > test_w1:
                cnt1 += 1
            if key2[match[x].queryIdx][0] > test_w2:
                cnt2 += 1
        if cnt1 >= cnt2:
            print("--- Arrangement of 2 images -> Done! No order changed ---")
            return img, order_change
        else:
            img.reverse()  # reverse order
            order_change = True
            print("--- Arrangement of 2 images -> Done! Order changed ---")
            return img, order_change
    print("--- Match < 4! ---")
    return img, order_change


def float32(img):
    return np.float32([key_point.pt for key_point in img.key])


def arrange_img(img, n):
    # n is number of images
    if n == 2:
        img1 = img[0]
        img2 = img[1]
        key1 = float32(img1)
        key2 = float32(img2)
        match = k_nearest_neighbors(img1.des, img2.des)
        img, is_changed = sort2(img, key1, key2, match)
        if is_changed:
            ret_img = [img[1], img[0]]
            return ret_img, is_changed
        else:
            return img, is_changed
    if n == 3:
        tmp1 = [img[0], img[1]]
        tmp2 = [img[0], img[2]]
        tmp3 = [img[1], img[2]]
        # arrange 2 images at a time
        arr_img12, is_12change = arrange_img(tmp1, 2)
        arr_img13, is_13change = arrange_img(tmp2, 2)
        arr_img23, is_23change = arrange_img(tmp3, 2)
        if not is_12change:
            if not is_23change:
                return img, False
            else:
                if not is_13change:
                    ret_img = [img[0], img[2], img[1]]
                    return ret_img, True
                else:
                    ret_img = [img[2], img[0], img[1]]
                    return ret_img, True
        else:   # 12 changed
            if is_23change:
                ret_img = [img[2], img[1], img[0]]
                return ret_img, True
            else:
                if not is_13change:
                    ret_img = [img[1], img[0], img[2]]
                    return ret_img, True
                else:
                    ret_img = [img[1], img[2], img[0]]
                    return ret_img, True
    elif n == 4:
        tmp1 = [img[0], img[1]]
        tmp2 = [img[0], img[2]]
        tmp3 = [img[0], img[3]]
        tmp4 = [img[1], img[2]]
        tmp5 = [img[1], img[3]]
        tmp6 = [img[2], img[3]]
        arr_img12, is_12change = arrange_img(tmp1, 2)
        arr_img13, is_13change = arrange_img(tmp2, 2)
        arr_img14, is_14change = arrange_img(tmp3, 2)
        arr_img23, is_23change = arrange_img(tmp4, 2)
        arr_img24, is_24change = arrange_img(tmp5, 2)
        arr_img34, is_34change = arrange_img(tmp6, 2)
        idx = [0, 1, 2, 3]     # record the moves
        if is_12change:
            idx[0] += 1
            idx[1] -= 1
        if is_13change:
            idx[0] += 1
            idx[2] -= 1
        if is_14change:
            idx[0] += 1
            idx[3] -= 1
        if is_23change:
            idx[1] += 1
            idx[2] -= 1
        if is_24change:
            idx[1] += 1
            idx[3] -= 1
        if is_34change:
            idx[2] += 1
            idx[3] -= 1
        print(idx)
        if (0 in idx) and (1 in idx) and (2 in idx) and (3 in idx):
            ret_img = [img[idx.index(0)], img[idx.index(1)], img[idx.index(2)], img[idx.index(3)]]
            # print("True")
            return ret_img, True
        else:
            # print(n, "images not all matches")
            return img, False
    elif n == 5:
        # ---- not done ----
        tmp1 = [img[0], img[1]]
        tmp2 = [img[0], img[2]]
        tmp3 = [img[0], img[3]]
        tmp4 = [img[0], img[4]]
        tmp5 = [img[1], img[2]]
        tmp6 = [img[1], img[3]]
        tmp7 = [img[1], img[4]]
        tmp8 = [img[2], img[3]]
        tmp9 = [img[2], img[4]]
        tmp10 = [img[3], img[4]]
        arr_img12, is_12change = arrange_img(tmp1, 2)
        arr_img13, is_13change = arrange_img(tmp2, 2)
        arr_img14, is_14change = arrange_img(tmp3, 2)
        arr_img15, is_15change = arrange_img(tmp4, 2)
        arr_img23, is_23change = arrange_img(tmp5, 2)
        arr_img24, is_24change = arrange_img(tmp6, 2)
        arr_img25, is_25change = arrange_img(tmp7, 2)
        arr_img34, is_34change = arrange_img(tmp8, 2)
        arr_img35, is_35change = arrange_img(tmp9, 2)
        arr_img45, is_45change = arrange_img(tmp10, 2)
        idx = [0, 1, 2, 3, 4]     # record the moves
        if is_12change:
            idx[0] += 1
            idx[1] -= 1
        if is_13change:
            idx[0] += 1
            idx[2] -= 1
        if is_14change:
            idx[0] += 1
            idx[3] -= 1
        if is_15change:
            idx[0] += 1
            idx[4] -= 1
        if is_23change:
            idx[1] += 1
            idx[2] -= 1
        if is_24change:
            idx[1] += 1
            idx[3] -= 1
        if is_25change:
            idx[1] += 1
            idx[4] -= 1
        if is_34change:
            idx[2] += 1
            idx[3] -= 1
        if is_35change:
            idx[2] += 1
            idx[4] -= 1
        if is_45change:
            idx[3] += 1
            idx[4] -= 1
        print("--- New order is ", idx, " ---")
        """left = []
        right = []
        split_line = [img[idx.index(0)]]   # n < idx is before idx image, n > idx is after idx image
        for i in range(len(idx)):
            if idx[i] != 0:
                left.append(img[i])
            elif idx[i] > 0:
                right.append(img[i])
        print("size of left and right", len(left), "\t", len(right))
        if len(left) > 1:
            left, is_left_change = arrange_img(left, len(left))
        if len(right) > 1:
            right, is_right_change = arrange_img(right, len(right))
        ret_img = left + split_line + right"""
        if (0 in idx) and (1 in idx) and (2 in idx) and (3 in idx) and (4 in idx):
            # print("True")
            ret_img = [img[idx.index(0)], img[idx.index(1)], img[idx.index(2)], img[idx.index(3)], img[idx.index(4)]]
            return ret_img, True
        else:
            return img, False
    else:
        print("--- Alert: More than 5 images! ---")
        return img, False


# Reference from https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
def homography(p1, p2):
    coor_list = []
    for (x1, y1), (x2, y2) in zip(p1, p2):
        coor_list.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        coor_list.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
    matrix = np.array(coor_list)
    u, s, v = LA.svd(matrix)    # get svd composition
    # print("Check svd's v: \n", v)
    h = v[8].reshape(3, 3)      # v[8] is the minimum singular value
    h = (1 / h[-1, -1]) * h     # h[-1,-1] is the last element in 3x3 matrix
    return h


def ransac(match, k1, k2):
    final_h = None
    max_inlier = []
    threshold = 0.5
    pt1 = np.array([k1[i.trainIdx].pt for i in match])
    pt2 = np.array([k2[i.queryIdx].pt for i in match])
    for i in range(1000):
        idx = np.random.choice(np.arange(len(pt1)), 4)   # randomly select 4 point
        ram_pt1 = pt1[idx]
        ram_pt2 = pt2[idx]
        h = homography(ram_pt1, ram_pt2)    # get homography matrix
        # get new world coordinate
        inlier = []
        for x in range(len(pt1)):
            p1 = np.transpose([pt1[x][0], pt1[x][1], 1])
            new = np.dot(h, p1)
            new = (1/(new.item(2)+0.00000001)) * new    # avoid to divide by zero
            p2 = np.transpose([pt2[x][0], pt2[x][1], 1])
            d = euclidean_distance(p2, new)
            if d < 4:
                inlier.append([pt1[x], pt2[x]])
        if len(inlier) > len(max_inlier):
            final_h = h
            max_inlier = inlier
        if len(inlier) > (len(pt1) * threshold):
            break
    return final_h


# Reference from https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
def warp_2_img(img1, img2, h):
    # get corners of two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    pts2_ppt = cv2.perspectiveTransform(pts2, h)
    conn = np.concatenate((pts1, pts2_ppt), axis=0)
    [xmin, ymin] = np.array(conn.min(axis=0).flatten() - 0.5, dtype=np.int32)
    [xmax, ymax] = np.array(conn.max(axis=0).flatten() + 0.5, dtype=np.int32)
    # get the translation homography matrix
    warped = [-xmin, -ymin]
    trans = np.array([[1, 0, warped[0]], [0, 1, warped[1]], [0, 0, 1]])  # translate
    warped_img = cv2.warpPerspective(img2, trans.dot(h), (xmax - xmin, ymax - ymin))
    warped_img[warped[1]:h1 + warped[1], warped[0]:w1 + warped[0]] = img1
    return warped_img


def stitch_2_img(img1, img2):
    stitched_img = None
    # get matches
    match = k_nearest_neighbors(img1.des, img2.des)
    """print(len(img1.key))
    print(len(img2.key))"""
    # get homography from ransac
    h = ransac(match, img1.key, img2.key)
    stitched_img = ImageInfo(warp_2_img(img2.img, img1.img, h))
    return stitched_img


def main():
    # argv[0] = file name, argv[1] = directory
    if len(sys.argv) != 2:
        print("--- Use this command: \"python stitch.py [data directory]\" ---")
        sys.exit(0)
    data_dir = sys.argv[1]
    if data_dir.startswith("'") and data_dir.endswith("'"):
        data_dir = data_dir[1:len(data_dir)-1]
    elif data_dir.endswith("'"):
        data_dir = data_dir[:len(data_dir)-1]
    elif data_dir.startswith("'"):
        data_dir = data_dir[1:]
    if not data_dir.endswith("/"):
        data_dir += "/"
    data_dir = os.path.join(data_dir)
    print("--- Data directory is [" + data_dir + "] ---")
    # os.path.join("",data_dir)
    # read images from the given directory
    img_file = os.listdir(data_dir)  # contain image name e.g. [img_name.jpg]
    if len(img_file) < 1:
        print("No image found.")
        sys.exit(0)
    img_list = []  # a list contain image file path e.g. [../data/image_name.jpg]
    images = []
    # print(*img_file)
    # The program should read ALL jpg files in the data directory
    for img in img_file:
        if (img.endswith(".jpg") or img.endswith(".JPG")) and img != "panorama.jpg":
            img_list.append(data_dir + img)  # e.g. [../data/image_name.jpg]
    # --- if only one image found ---
    if len(img_list) == 1:
        read_img = cv2.imread(img_list[0])
        cv2.imwrite(data_dir + "panorama.jpg", read_img)
    # --- ignore identical image ---
    print("--- %d jpg images found in file ---" % len(img_list))
    new_img_list = []
    for i in range(len(img_list)):
        is_same = False
        for j in range(i+1, len(img_list)):
            img1 = Image.open(img_list[i])
            img2 = Image.open(img_list[j])
            if list(img1.getdata()) == list(img2.getdata()):
                is_same = True
                break
        if not is_same:
            new_img_list.append(img_list[i])
    print("--- Ignore identical images, %d images left ready for stitching ---" % len(new_img_list))
    # ---------
    for img_dir in new_img_list:
        read_img = cv2.imread(img_dir)
        """if read_img.shape[1] > 1000 and read_img.shape[0] > 1000:
            width = int(read_img.shape[1] * 0.8)
            height = int(read_img.shape[0] * 0.8)
            dim = (width, height)
            # resize image
            read_img = cv2.resize(read_img, dim, interpolation=cv2.INTER_AREA)"""
        images.append(ImageInfo(read_img))
    # determine the spatial arrangement of the images
    img_num = len(images)
    print("--- Arranging %d images ---" % len(images))
    stitch = None
    if img_num > 2:     # if only 2 images then no need to arrange
        images, is_changed = arrange_img(images, img_num)
    print("--- Arrangement is completed. Start stitching... ---")
    if img_num == 2:
        # show image make sure arrange is correct
        """for i in range(img_num):
            images[i].show()"""
        stitch = stitch_2_img(images[0], images[1])
    elif img_num == 3:
        """for i in range(img_num):
            images[i].show()"""
        # stitch 2 img
        stitch = stitch_2_img(images[0], images[1])
        # stitch again
        # stitch.show()
        stitch = stitch_2_img(stitch, images[2])
    elif img_num == 4:
        """for i in range(img_num):
            images[i].show()"""
        stitch = stitch_2_img(images[1], images[0])
        # stitch.show()
        stitch = stitch_2_img(stitch, images[2])
        # stitch.show()
        stitch = stitch_2_img(stitch, images[3])
    elif img_num == 5:
        """for i in range(img_num):
            images[i].show()"""
        stitch = stitch_2_img(images[1], images[0])
        # stitch.show()
        stitch = stitch_2_img(stitch, images[2])
        # stitch.show()
        stitch = stitch_2_img(stitch, images[3])
        # stitch.show()
        stitch = stitch_2_img(stitch, images[4])
        # stitch.show()
    else:
        print("--- Alert: More than 5 images! ---")
        return 0
    print("--- Generating panorama image ---")
    cv2.imwrite(data_dir + "panorama.jpg", stitch.img)
    print("--- Done! Result image is in [" + data_dir + "panorama.jpg] ---")


if __name__ == '__main__':
    main()
