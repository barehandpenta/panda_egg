import tensorflow as tf
import cv2
import posenet
import egg_model
import numpy as np


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']
        scale_factor = 0.7125

        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        eggNet = egg_model.PandaEgg()
        eggNet.load_weights('data/egg_model_weights.csv')
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=scale_factor, output_stride=output_stride)

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

            if np.array_equal(keypoint_coords, np.zeros((1, 17, 2))):
                text = 'Nope'
            else:
                res = eggNet.pose_detect(keypoint_coords)
                if res == 0:
                    text = 'STANDING'
                elif res == 1:
                    text = 'SITTING'
                elif res == 2:
                    text = 'LYING'

            cv2.putText(overlay_image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow('posenet', overlay_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break



if __name__ == "__main__":
    main()