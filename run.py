import cv2
import DeepLab
import numpy as np
from PIL import Image

if __name__ == '__main__':
    THRESHOLD_VALUE = 1
    deeplab = DeepLab.DeepLabModel('models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz')
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())  # False
    rval, frame = cap.read()
    while (rval):

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_img = Image.fromarray(img)

        resized_raw, seg_map = deeplab.run(raw_img)
        # bin_seg_map = seg_map.point(lambda p: p >= THRESHOLD_VALUE and 255)
        bin_seg_map = (np.asanyarray(seg_map) >= THRESHOLD_VALUE) * 255
        rev_bin_seg_map = (bin_seg_map < 255) * 255
        bin_seg_map = np.stack((bin_seg_map,), -1).astype(np.uint8)
        rev_bin_seg_map = np.stack((rev_bin_seg_map,), -1).astype(np.uint8)

        # Blur background and mask
        resized_img = np.asanyarray(resized_raw)
        front_img = resized_img.copy()
        bkg_img = cv2.GaussianBlur(resized_img, (15, 15), 0)

        masked_bkg_img = np.bitwise_and(bkg_img, rev_bin_seg_map)
        masked_front_img = np.bitwise_and(front_img, bin_seg_map)

        result_img = cv2.cvtColor(np.bitwise_or(masked_front_img, masked_bkg_img), cv2.COLOR_BGR2RGB)


        cv2.imshow('frame', result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rval, frame = cap.read()