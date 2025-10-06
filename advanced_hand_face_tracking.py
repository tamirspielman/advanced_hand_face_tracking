import argparse
import time
import os
import csv
import math
from collections import deque

import cv2
import numpy as np

# Optional heavy deps
try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except Exception as e:
    raise ImportError("mediapipe is required. Install with `pip install mediapipe`.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
except Exception:
    # training functionality will raise helpful error later if used
    pass


# ----------------------------- Utility Filters -----------------------------
class OneEuroFilter:
    """
    One Euro Filter for smoothing noisy signals while preserving responsiveness.
    Reference: "The One Euro Filter: A Simple Speed-based Low-pass Filter for Interactive Applications"
    """
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.last_time = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x
            return x
        # derivative
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # adapt cutoff
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        # store
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# Simple Kalman filter for 2D points (x,y)
class Kalman2D:
    def __init__(self, process_noise=1e-2, meas_noise=1e-1):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = meas_noise * np.eye(2, dtype=np.float32)
        self.initialized = False

    def predict(self):
        pred = self.kalman.predict()
        return pred[0][0], pred[1][0]

    def correct(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            # Initialize state
            self.kalman.statePre = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self.initialized = True
        corrected = self.kalman.correct(meas)
        return corrected[0][0], corrected[1][0]


# ----------------------------- Core Tracking Class -----------------------------
class AdvancedTracker:
    def __init__(self, source=0, cam_width=1280, cam_height=720, realtime_freq=30):
        self.source = source
        self.width = cam_width
        self.height = cam_height
        self.freq = realtime_freq

        # Create hand and face mesh instances
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Filters per landmark set
        self.hand_filters = {}  # key: (hand_id,index) -> OneEuroFilter
        self.face_filter = OneEuroFilter(self.freq, mincutoff=0.4, beta=0.3)

        # Bounding box Kalman
        self.box_kalman = Kalman2D()

        # Gesture classifier
        self.gesture_clf = None

        # For dataset capture
        self.capture_dir = "gesture_dataset"
        os.makedirs(self.capture_dir, exist_ok=True)

    # --------------------------- Helpers ---------------------------------
    def _landmarks_to_np(self, lm_list, shape):
        h, w = shape
        arr = np.array([[int(lm.x * w), int(lm.y * h), lm.z] for lm in lm_list], dtype=float)
        return arr

    def _smooth_hand_landmarks(self, hand_id, landmarks):
        # landmarks: N x 3 numpy array
        smoothed = np.zeros_like(landmarks)
        for i, (x,y,z) in enumerate(landmarks):
            key = (hand_id, i)
            if key not in self.hand_filters:
                self.hand_filters[key] = OneEuroFilter(self.freq, mincutoff=1.0, beta=0.01)
            filt = self.hand_filters[key]
            sm = filt.filter(np.array([x,y,z]))
            smoothed[i] = sm
        return smoothed

    def _face_landmarks_to_np(self, landmarks):
        # landmarks: list of normalized landmarks from MediaPipe Face Mesh
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
        return arr

    # ------------------------ Head Pose Estimation -----------------------
    def estimate_head_pose(self, face_landmarks, image_shape):
        # Select a subset of landmark indices that are stable across faces
        # These indices are commonly used with MediaPipe FaceMesh
        indices = [33, 263, 1, 61, 291, 199]
        try:
            image_points = []
            for idx in indices:
                lm = face_landmarks[idx]
                image_points.append((lm.x * image_shape[1], lm.y * image_shape[0]))
            image_points = np.array(image_points, dtype='double')

            # Approximate 3D model points of the selected features in mm
            model_points = np.array([
                (-30.0,  -90.0, -30.0),  # left eye outer
                (30.0,   -90.0, -30.0),  # right eye outer
                (0.0,    0.0,    0.0  ),  # nose tip
                (-60.0,  80.0,  -30.0),  # left mouth corner
                (60.0,   80.0,  -30.0),  # right mouth corner
                (0.0,    120.0, -30.0)   # chin-ish
            ])

            focal_length = image_shape[1]
            center = (image_shape[1] / 2, image_shape[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype='double')

            dist_coeffs = np.zeros((4,1))
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            rot_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rot_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            pitch, yaw, roll = euler_angles.flatten()
            return True, (pitch, yaw, roll), (rotation_vector, translation_vector)
        except Exception as e:
            return False, (0,0,0), None

    # ------------------------ Gesture dataset / training -----------------
    def capture_gesture_frame(self, landmark_vector, label):
        # Save a single sample as CSV row (label + flattened landmarks)
        path = os.path.join(self.capture_dir, f"gestures.csv")
        row = [label] + landmark_vector.tolist()
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def train_gesture_classifier(self, dataset_path=None, model_out='gesture_clf.joblib'):
        path = dataset_path or os.path.join(self.capture_dir, 'gestures.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        X = []
        y = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                label = row[0]
                feats = list(map(float, row[1:]))
                X.append(feats)
                y.append(label)
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Gesture classifier trained. Test accuracy: {acc:.3f}")
        joblib.dump(clf, model_out)
        self.gesture_clf = clf
        return clf, acc

    def load_gesture_classifier(self, path):
        self.gesture_clf = joblib.load(path)

    # ------------------------ Main loop ---------------------------------
    def start(self, record=False, save_output=None, capture_label=None):
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera source {self.source}.")
            print("Try a different camera index: --source 1 or --source 2")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        print(f"Camera opened successfully. Press 'q' or ESC to quit.")

        out = None
        if save_output:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(save_output, fourcc, 20.0, (self.width, self.height))
            if not out.isOpened():
                print("Warning: Could not initialize video writer. Video will not be saved.")
                out = None

        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame from camera.")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(image_rgb)
            results_face = self.face_mesh.process(image_rgb)

            image = frame.copy()

            if hasattr(results_face, 'multi_face_landmarks') and results_face.multi_face_landmarks:  # type: ignore
                face_landmarks = results_face.multi_face_landmarks[0].landmark  # type: ignore
                ok, euler, raw_pose = self.estimate_head_pose(face_landmarks, image.shape)
                if ok:
                    pitch, yaw, roll = euler
                    cv2.putText(image, f"Head pose P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                h, w = image.shape[:2]
                for start_idx, end_idx in mp_face_mesh.FACEMESH_TESSELATION:  # type: ignore
                    s = face_landmarks[start_idx]
                    e = face_landmarks[end_idx]
                    x1, y1 = int(s.x * w), int(s.y * h)
                    x2, y2 = int(e.x * w), int(e.y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (80, 110, 10), 1)
                for lm in face_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            if hasattr(results_hands, 'multi_hand_landmarks') and results_hands.multi_hand_landmarks:  # type: ignore
                for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):  # type: ignore
                    lm_np = self._landmarks_to_np(hand_landmarks.landmark, image.shape[:2])
                    sm = self._smooth_hand_landmarks(hand_idx, lm_np)

                    for (x, y, z) in sm:
                        cv2.circle(image, (int(x), int(y)), 2, (0, 200, 255), -1)

                    x_min, y_min = int(np.min(sm[:, 0])), int(np.min(sm[:, 1]))
                    x_max, y_max = int(np.max(sm[:, 0])), int(np.max(sm[:, 1]))
                    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                    self.box_kalman.correct(cx, cy)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                    flat = sm[:, :2].flatten()
                    if capture_label is not None:
                        self.capture_gesture_frame(flat, capture_label)
                        cv2.putText(image, f"Capturing: {capture_label}",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if self.gesture_clf is not None:
                        try:
                            pred = self.gesture_clf.predict([flat])[0]
                            cv2.putText(image, f"Gesture: {pred}",
                                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        except Exception as e:
                            print("Gesture prediction error:", e)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(image, f"FPS: {int(fps)}", 
                       (self.width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('AdvancedTracker', image)
            if out is not None and out.isOpened():
                out.write(image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

# ----------------------------- CLI ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Advanced hand + face tracking pipeline')
    parser.add_argument('--source', type=int, default=0, help='Camera source (int) or video file path')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--capture-label', type=str, default=None, 
                       help='Label name to capture gesture frames (press any key while running to capture)')
    parser.add_argument('--save-output', type=str, default=None, 
                       help='Save annotated output video path (avi recommended)')
    parser.add_argument('--train', action='store_true', 
                       help='Train gesture classifier from dataset and save model')
    parser.add_argument('--load-model', type=str, default=None, 
                       help='Load a pretrained gesture classifier (joblib)')
    args = parser.parse_args()

    tracker = AdvancedTracker(source=args.source, cam_width=args.width, cam_height=args.height)

    if args.load_model:
        tracker.load_gesture_classifier(args.load_model)
    if args.train:
        print('Training gesture classifier...')
        tracker.train_gesture_classifier()
        return

    tracker.start(record=False, save_output=args.save_output, capture_label=args.capture_label)


if __name__ == '__main__':
    main()
