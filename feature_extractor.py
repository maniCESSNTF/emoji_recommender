import math


class FeatureExtractor:
    def __init__(self):
        # دهان
        self.LIPS_UPPER = 13
        self.LIPS_LOWER = 14
        self.LIPS_LEFT = 61
        self.LIPS_RIGHT = 291

        # چشم‌ها
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_OUTER = 263

        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.LEFT_EYE_INNER = 133
        self.LEFT_EYE_OUTER = 33

        # ابروها
        self.LEFT_EYEBROW_INNER = 105
        self.RIGHT_EYEBROW_INNER = 334

    def calculate_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def extract_features(self, landmarks):
        if not landmarks or len(landmarks) < 468:
            return None

        # 1. عرض مرجع صورت
        face_width = self.calculate_distance(landmarks[self.LEFT_EYE_OUTER], landmarks[self.RIGHT_EYE_OUTER])
        if face_width == 0: face_width = 1.0

        # 2. ویژگی‌های دهان
        mouth_vertical = self.calculate_distance(landmarks[self.LIPS_UPPER], landmarks[self.LIPS_LOWER])
        mouth_horizontal = self.calculate_distance(landmarks[self.LIPS_LEFT], landmarks[self.LIPS_RIGHT])

        mar = mouth_vertical / mouth_horizontal if mouth_horizontal != 0 else 0
        mwr = mouth_horizontal / face_width

        center_lip_y = landmarks[self.LIPS_UPPER][1]
        avg_corners_y = (landmarks[self.LIPS_LEFT][1] + landmarks[self.LIPS_RIGHT][1]) / 2.0
        lip_turn = (center_lip_y - avg_corners_y) / face_width

        # 3. ویژگی‌های چشم
        right_eye_vertical = self.calculate_distance(landmarks[self.RIGHT_EYE_TOP], landmarks[self.RIGHT_EYE_BOTTOM])
        right_eye_horizontal = self.calculate_distance(landmarks[self.RIGHT_EYE_INNER], landmarks[self.RIGHT_EYE_OUTER])
        right_ear = right_eye_vertical / right_eye_horizontal if right_eye_horizontal != 0 else 0

        left_eye_vertical = self.calculate_distance(landmarks[self.LEFT_EYE_TOP], landmarks[self.LEFT_EYE_BOTTOM])
        left_eye_horizontal = self.calculate_distance(landmarks[self.LEFT_EYE_INNER], landmarks[self.LEFT_EYE_OUTER])
        left_ear = left_eye_vertical / left_eye_horizontal if left_eye_horizontal != 0 else 0

        # (جدید) میانگین باز بودن دو چشم برای تشخیص خیرگی 😳
        avg_ear = (right_ear + left_ear) / 2.0

        # 4. ویژگی‌های ابرو
        eyebrow_distance = self.calculate_distance(landmarks[self.LEFT_EYEBROW_INNER],
                                                   landmarks[self.RIGHT_EYEBROW_INNER])
        eyebrow_ratio = eyebrow_distance / mouth_horizontal if mouth_horizontal != 0 else 0

        # (جدید) عدم تقارن ابروها برای تشخیص شک 🤨
        left_brow_lift = self.calculate_distance(landmarks[self.LEFT_EYE_INNER],
                                                 landmarks[self.LEFT_EYEBROW_INNER]) / face_width
        right_brow_lift = self.calculate_distance(landmarks[self.RIGHT_EYE_INNER],
                                                  landmarks[self.RIGHT_EYEBROW_INNER]) / face_width
        brow_asymmetry = abs(left_brow_lift - right_brow_lift)

        return {
            "mar": mar,
            "mwr": mwr,
            "lip_turn": lip_turn,
            "right_ear": right_ear,
            "left_ear": left_ear,
            "avg_ear": avg_ear,  # متغیر جدید
            "eyebrow_ratio": eyebrow_ratio,
            "brow_asymmetry": brow_asymmetry  # متغیر جدید
        }