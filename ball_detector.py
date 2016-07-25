import cv2
import numpy as np
from time import time


def _get_blob_detector():
    blob_detector_params = cv2.SimpleBlobDetector_Params()
    blob_detector_params.filterByCircularity = True
    blob_detector_params.minCircularity = 0.5
    blob_detector_params.maxCircularity = 1
    blob_detector_params.filterByArea = True
    blob_detector_params.minArea = 200
    blob_detector_params.maxArea = 225000
    blob_detector_params.filterByConvexity = True
    blob_detector_params.minConvexity = 0.5
    blob_detector_params.maxConvexity = 1
    blob_detector_params.filterByColor = True
    blob_detector_params.blobColor = 255

    blob_detector = cv2.SimpleBlobDetector_create(blob_detector_params)

    return blob_detector


def _filter_by_color(image, bottom_color_limit, top_color_limit):
    mask = cv2.inRange(image, bottom_color_limit, top_color_limit)

    white = np.ndarray((image.shape[0], image.shape[1]), dtype=np.uint8)
    white[0:, 0:] = 255

    result = cv2.bitwise_and(white, white, mask=mask)

    return result


def detect_ball(color, color_threshold, tracking_size_threshold):
    cap = cv2.VideoCapture(0)
    blob_detector = _get_blob_detector()

    # Calculate the color range
    bottom_color_limit = np.array(
        list(map(lambda x: max(0, min(255, x - (255 * color_threshold))),
                 color)), dtype=np.uint8)
    top_color_limit = np.array(
        list(map(lambda x: max(0, min(255, x + (255 * color_threshold))),
                 color)), dtype=np.uint8)

    last_size = None
    track = []
    lost_times = 0

    while True:
        # Read each frame
        has_at_least_one_frame, raw_image = cap.read()

        # Check the condition of exit
        if not has_at_least_one_frame or \
                (cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'))):
            break

        # Apply gaussian blur to get a more color-homogeneous image
        smooth_image = cv2.GaussianBlur(raw_image, (31, 31), 0)

        # Filter the image by range color, get a binary image of pixels that
        # are withing the color range
        binary_image = _filter_by_color(
            smooth_image, bottom_color_limit, top_color_limit)

        # Try to fill holes with dilate morphological operation
        binary_image = cv2.dilate(binary_image, (20, 20), iterations=5)

        # Get the blobs that shapes like a circumference
        possible_balls = blob_detector.detect(binary_image)

        if possible_balls:
            # Order blobs by size
            possible_balls.sort(key=lambda x: x.size)

            greater_blob = possible_balls[-1]
            if last_size is None:
                last_size = greater_blob.size
            # For the tracking, relate the new blob with the previous just
            # if size dod not change abruptly
            if np.isclose(greater_blob.size, last_size,
                          atol=tracking_size_threshold):
                x1 = int(max(greater_blob.pt[0] - greater_blob.size, 0))
                y1 = int(max(greater_blob.pt[1] - greater_blob.size, 0))
                x2 = int(min(greater_blob.pt[0] +
                             greater_blob.size, raw_image.shape[1]))
                y2 = int(min(greater_blob.pt[1] +
                             greater_blob.size, raw_image.shape[0]))
                cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                last_size = greater_blob.size
                lost_times = 0

                track.append((tuple(map(lambda x: int(x), greater_blob.pt)),
                              time()))

        else:
            # Count amount of frames without tracking to reset
            # the blob's size association
            lost_times += 1
            if lost_times == 5:
                last_size = None

        # Remove tracking lines older than 3 seconds
        while track and track[0][1] < time() - 3:
            track.pop(0)

        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        # Draw the tracked path
        for start_pt, end_pt in zip(track, track[1:]):
            cv2.line(raw_image, start_pt[0], end_pt[0], (255, 0, 0),
                     thickness=2)

        cv2.imshow(
            'Result', np.hstack((raw_image, smooth_image, binary_image)))


if __name__ == '__main__':
    ball_color = (3, 30, 148)  # (G, B, R)
    color_tolerance = 0.12
    tracking_size_tolerance = 20  # size to consider a not abruptly change

    detect_ball(ball_color, color_tolerance, tracking_size_tolerance)
