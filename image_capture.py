import tensorflow as tf
import cv2
import time
import posenet
import  numpy as np


def main():
    # write = open_csv_file('../data/lie.csv', 'w')

    cam_width = 640
    cam_height = 480
    scale_factor = 0.7125
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']
        cap = cv2.VideoCapture(0)
        cap.set(3, cam_width)
        cap.set(4, cam_height)

        start = time.time()
        print(start)
        while True:
            background = np.zeros((480, 640, 3))
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor, output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            print(display_image.shape)
            cv2.imshow('posenet', overlay_image)
            print(time.time())
            if cv2.waitKey(1) & 0xFF == 27:
                break
            elif time.time() - start > 5:
                cv2.imwrite("example_images/stand_exp1.jpeg", overlay_image)
                break

if __name__ == "__main__":
    main()