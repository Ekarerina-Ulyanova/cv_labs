import argparse
import cv2 as cv
import numpy as np

from pathlib import Path


LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 3.0
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.4
STRONG_SIFT_INLIERS = 12
FLOW_WINDOW = (21, 21)
FRAME_COLOR = (0, 255, 0)


def select_corners(frame):
    window = "Select corners - Enter: confirm | R: reset | Esc: cancel"
    points = []

    def on_mouse(event, x, y, _flags, _data):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv.namedWindow(window)
    try:
        cv.setMouseCallback(window, on_mouse)
        while True:
            preview = frame.copy()
            corners = np.array(points, dtype=np.int32)
            if len(points) >= 2:
                cv.polylines(preview, [corners], len(points) == 4, FRAME_COLOR, 2)
            for index, point in enumerate(points, start=1):
                cv.circle(preview, point, 4, FRAME_COLOR, -1)
                cv.putText(preview, str(index), (point[0] + 6, point[1] - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, FRAME_COLOR, 2)
            cv.imshow(window, preview)
            key = cv.waitKey(20) & 0xFF
            if key == 27 or cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                raise ValueError("Corner selection cancelled.")
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                corners = np.array(points, dtype=np.int32)
                if cv.isContourConvex(corners) and cv.contourArea(corners) > 0:
                    return corners
                print("Click the four corners in order around the object, then press Enter.")
                points.clear()
    finally:
        if cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow(window)


class SiftTracker:
    def __init__(self, frame, corners, mask=None):
        self.corners = corners.astype(np.float32).reshape(-1, 1, 2)
        self.sift = cv.SIFT_create()
        self.matcher = cv.BFMatcher(cv.NORM_L2)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.previous_gray = gray
        self.previous_corners = self.corners.reshape(-1, 2)
        self.reference_area = cv.contourArea(self.corners)
        self.reference_points, self.reference_descriptors = self.sift.detectAndCompute(gray, mask)
        if len(self.reference_points) < MIN_INLIERS:
            raise ValueError("Too few SIFT keypoints in the selected first-frame region.")

    def locate(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, inlier_count = self.match_reference(gray)
        if self.previous_corners is not None and inlier_count < STRONG_SIFT_INLIERS:
            flow_corners = self.follow_previous_frame(gray)
            if flow_corners is not None:
                diagonal = np.linalg.norm(np.ptp(flow_corners, axis=0))
                max_difference = max(5.0, 0.08 * diagonal)
                if corners is None or np.max(
                    np.linalg.norm(corners - flow_corners, axis=1)
                ) > max_difference:
                    corners = flow_corners

        self.previous_gray = gray
        self.previous_corners = corners
        return corners

    def match_reference(self, gray):
        points, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None, 0

        matches_by_point = {}
        for pair in self.matcher.knnMatch(self.reference_descriptors, descriptors, k=2):
            if len(pair) != 2:
                continue
            best, second = pair
            if best.distance >= LOWE_RATIO * second.distance:
                continue
            previous = matches_by_point.get(best.trainIdx)
            if previous is None or best.distance < previous.distance:
                matches_by_point[best.trainIdx] = best

        matches = list(matches_by_point.values())
        if len(matches) < MIN_INLIERS:
            return None, 0

        source = np.float32([self.reference_points[m.queryIdx].pt for m in matches])
        target = np.float32([points[m.trainIdx].pt for m in matches])
        homography, inliers = cv.findHomography(source, target, cv.RANSAC, RANSAC_THRESHOLD)
        if homography is None or inliers is None:
            return None, 0
        inliers = inliers.ravel().astype(bool)
        inlier_count = np.count_nonzero(inliers)
        if inlier_count < MIN_INLIERS or inlier_count / len(matches) < MIN_INLIER_RATIO:
            return None, 0
        if cv.contourArea(cv.convexHull(source[inliers])) < 0.08 * self.reference_area:
            return None, 0

        corners = self.project_corners(self.corners, homography)
        if corners is None:
            return None, 0
        return corners, inlier_count

    @staticmethod
    def project_corners(corners, homography):
        corners = corners.reshape(-1, 2)
        scales = corners @ homography[2, :2] + homography[2, 2]
        if not (np.all(scales > 0) or np.all(scales < 0)):
            return None
        projected = cv.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
        if not np.isfinite(projected).all() or np.max(np.abs(projected)) >= 2**30:
            return None
        if not cv.isContourConvex(projected) or cv.contourArea(projected) < 16:
            return None
        return projected

    def follow_previous_frame(self, gray):
        previous_corners = self.previous_corners
        mask = np.zeros_like(self.previous_gray)
        cv.fillConvexPoly(mask, np.rint(previous_corners).astype(np.int32), 255)
        mask = cv.erode(mask, np.ones((5, 5), dtype=np.uint8))
        points = cv.goodFeaturesToTrack(
            self.previous_gray, maxCorners=150, qualityLevel=0.01, minDistance=4, mask=mask
        )
        if points is None or len(points) < MIN_INLIERS:
            return None

        moved, status, errors = cv.calcOpticalFlowPyrLK(
            self.previous_gray, gray, points, None, winSize=FLOW_WINDOW
        )
        if moved is None:
            return None
        found = (status.ravel() != 0) & (errors.ravel() < 24)
        if np.count_nonzero(found) < MIN_INLIERS:
            return None
        points, moved = points[found], moved[found]
        returned, status, _ = cv.calcOpticalFlowPyrLK(
            gray, self.previous_gray, moved, None, winSize=FLOW_WINDOW
        )
        if returned is None:
            return None
        round_trip_error = np.linalg.norm((points - returned).reshape(-1, 2), axis=1)
        valid = (status.ravel() != 0) & (round_trip_error < 1.25)
        if np.count_nonzero(valid) < MIN_INLIERS:
            return None
        points, moved = points[valid], moved[valid]

        transform, inliers = cv.findHomography(points, moved, cv.RANSAC, 1.7)
        if transform is None or inliers is None:
            return None
        inliers = inliers.ravel().astype(bool)
        if np.count_nonzero(inliers) < MIN_INLIERS or inliers.mean() < 0.65:
            return None
        area = cv.contourArea(previous_corners)
        if cv.contourArea(cv.convexHull(points[inliers])) < 0.15 * area:
            return None
        corners = self.project_corners(previous_corners, transform)
        if corners is None:
            return None
        if not 0.5 * area < cv.contourArea(corners) < 2 * area:
            return None
        return corners


def draw_object(frame, corners, label):
    polygon = np.rint(corners).astype(np.int32)
    cv.polylines(frame, [polygon], True, FRAME_COLOR, 2, cv.LINE_AA)

    height, width = frame.shape[:2]
    font = cv.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv.getTextSize(label, font, 0.6, 2)
    top_x, top_y = polygon[polygon[:, 1].argmin()]
    x = int(np.clip(top_x, 0, max(0, width - text_width - 1)))
    y = int(np.clip(top_y - 6, text_height + 3, height - 1))
    cv.putText(frame, label, (x, y), font, 0.6, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(frame, label, (x, y), font, 0.6, FRAME_COLOR, 2, cv.LINE_AA)


def track_video(input_path, output_path, mode, label):
    if input_path.resolve() == output_path.resolve() or (
        output_path.exists() and input_path.exists() and input_path.samefile(output_path)
    ):
        raise ValueError("Input and output must be different files.")

    codecs = {".mp4": "mp4v", ".avi": "MJPG"}
    codec = codecs.get(output_path.suffix.lower())
    if codec is None:
        raise ValueError("Output must have an .mp4 or .avi extension.")

    capture = cv.VideoCapture(str(input_path))
    try:
        ok, first_frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read the first frame of {input_path}.")

        fps = capture.get(cv.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Could not determine the input video frame rate.")
        height, width = first_frame.shape[:2]
        mask = None
        if mode == "manual":
            corners = select_corners(first_frame)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv.fillConvexPoly(mask, corners, 255)
        else:
            corners = np.array(
                [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)],
                dtype=np.int32,
            )

        tracker = SiftTracker(first_frame, corners, mask)
        writer = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*codec), fps, (width, height))
        try:
            if not writer.isOpened():
                raise ValueError(f"Could not open {output_path} for writing.")
            draw_object(first_frame, corners, label)
            writer.write(first_frame)
            total = marked = 1

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                corners = tracker.locate(frame)
                if corners is not None:
                    draw_object(frame, corners, label)
                    marked += 1
                writer.write(frame)
                total += 1
        finally:
            writer.release()
    finally:
        capture.release()

    print(f"Saved {output_path}. Marked {marked} of {total} frames.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input video")
    parser.add_argument("output", type=Path, help="Output video (.mp4 or .avi)")
    parser.add_argument("--mode", choices=("full-frame", "manual"), default="full-frame")
    parser.add_argument("--label", default="Object", help="Label for the tracked object")
    args = parser.parse_args()

    try:
        track_video(args.input, args.output, args.mode, args.label)
    except (ValueError, OSError, cv.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()