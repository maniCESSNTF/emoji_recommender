import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        # MediaPipe drawing tools
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        landmarks_list = []

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            for lm in face_landmarks.landmark:
                landmarks_list.append((lm.x, lm.y, lm.z))

        return landmarks_list, results.multi_face_landmarks

    def draw_landmarks(self, frame, multi_face_landmarks):
        """
        This function draws a 3D mesh over the face in the image.
        """
        if multi_face_landmarks:
            for face_landmarks in multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        return frame