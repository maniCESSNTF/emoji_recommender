import math


class FeatureExtractor:
    def __init__(self):
        # دهان
        self.LIPS_UPPER = 13
        self.LIPS_LOWER = 14
        self.LIPS_LEFT = 61
        self.LIPS_RIGHT = 291
        self.LIPS_INNER_UPPER = 13  # برای دندان
        self.LIPS_INNER_LOWER = 14  # برای دندان

        # نقاط کمکی برای محاسبه دقیق‌تر باز بودن دهان
        self.LIPS_UPPER_LEFT = 37
        self.LIPS_LOWER_LEFT = 84
        self.LIPS_UPPER_RIGHT = 267
        self.LIPS_LOWER_RIGHT = 314

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
        self.LEFT_EYEBROW_CENTER = 66
        self.LEFT_EYEBROW_OUTER = 70
        self.RIGHT_EYEBROW_INNER = 334
        self.RIGHT_EYEBROW_CENTER = 296
        self.RIGHT_EYEBROW_OUTER = 300

        # گونه‌ها
        self.LEFT_CHEEK = 205
        self.RIGHT_CHEEK = 425

        # نقاط مرجع صورت
        self.NOSE_TIP = 1
        self.NOSE_BRIDGE = 8  # پل بینی (مرجع عمودی پایدار)
        self.CHIN = 152

    def distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def extract_features(self, landmarks):
        if not landmarks or len(landmarks) < 468:
            return None

        # ==========================================
        # 0. مقیاس‌های مرجع ضدضربه (استخوانی)
        # ==========================================
        # عرض صورت: بر اساس فاصله گوشه‌های داخلی چشم (تغییر نمی‌کند)
        face_width = self.distance(landmarks[self.LEFT_EYE_INNER], landmarks[self.RIGHT_EYE_INNER])

        # ارتفاع صورت: بر اساس فاصله پل بینی تا چانه (تغییر نمی‌کند)
        face_height = self.distance(landmarks[self.NOSE_BRIDGE], landmarks[self.CHIN])

        if face_width == 0: face_width = 1.0
        if face_height == 0: face_height = 1.0

        # ==========================================
        # 1. گروه دهان
        # ==========================================
        # محاسبه میانگین باز بودن دهان از ۳ نقطه برای دقت بیشتر
        mouth_vert_center = self.distance(landmarks[self.LIPS_UPPER], landmarks[self.LIPS_LOWER])
        mouth_vert_left = self.distance(landmarks[self.LIPS_UPPER_LEFT], landmarks[self.LIPS_LOWER_LEFT])
        mouth_vert_right = self.distance(landmarks[self.LIPS_UPPER_RIGHT], landmarks[self.LIPS_LOWER_RIGHT])
        avg_mouth_vert = (mouth_vert_center + mouth_vert_left + mouth_vert_right) / 3.0

        mouth_horiz = self.distance(landmarks[self.LIPS_LEFT], landmarks[self.LIPS_RIGHT])

        mouth_openness = avg_mouth_vert / face_width
        mouth_width = mouth_horiz / face_width
        mouth_ratio = mouth_horiz / avg_mouth_vert if avg_mouth_vert != 0 else 0

        center_lip_y = landmarks[self.LIPS_UPPER][1]
        avg_corner_y = (landmarks[self.LIPS_LEFT][1] + landmarks[self.LIPS_RIGHT][1]) / 2.0

        lip_elevation = max(0, (center_lip_y - avg_corner_y) / face_width)
        lip_depression = max(0, (avg_corner_y - center_lip_y) / face_width)
        mouth_asymmetry = abs(landmarks[self.LIPS_LEFT][1] - landmarks[self.LIPS_RIGHT][1]) / face_width

        # دیده شدن دندان (میزان فاصله لبه‌های داخلی لب)
        teeth_visibility = self.distance(landmarks[self.LIPS_INNER_UPPER],
                                         landmarks[self.LIPS_INNER_LOWER]) / face_width

        # ==========================================
        # 2. گروه چشم‌ها
        # ==========================================
        left_eye_openness = self.distance(landmarks[self.LEFT_EYE_TOP], landmarks[self.LEFT_EYE_BOTTOM]) / face_width
        right_eye_openness = self.distance(landmarks[self.RIGHT_EYE_TOP], landmarks[self.RIGHT_EYE_BOTTOM]) / face_width
        eye_difference = abs(left_eye_openness - right_eye_openness)

        # ==========================================
        # 3. گروه ابروها
        # ==========================================
        left_brow_height = self.distance(landmarks[self.LEFT_EYEBROW_INNER],
                                         landmarks[self.LEFT_EYE_INNER]) / face_width
        right_brow_height = self.distance(landmarks[self.RIGHT_EYEBROW_INNER],
                                          landmarks[self.RIGHT_EYE_INNER]) / face_width
        brow_difference = abs(left_brow_height - right_brow_height)

        left_brow_tilt = (landmarks[self.LEFT_EYEBROW_OUTER][1] - landmarks[self.LEFT_EYEBROW_INNER][1]) / face_width
        right_brow_tilt = (landmarks[self.RIGHT_EYEBROW_OUTER][1] - landmarks[self.RIGHT_EYEBROW_INNER][1]) / face_width

        brow_symmetry = abs(left_brow_tilt - right_brow_tilt) + brow_difference

        avg_left_ends_y = (landmarks[self.LEFT_EYEBROW_INNER][1] + landmarks[self.LEFT_EYEBROW_OUTER][1]) / 2
        left_brow_arch = max(0, (avg_left_ends_y - landmarks[self.LEFT_EYEBROW_CENTER][1]) / face_width)

        avg_right_ends_y = (landmarks[self.RIGHT_EYEBROW_INNER][1] + landmarks[self.RIGHT_EYEBROW_OUTER][1]) / 2
        right_brow_arch = max(0, (avg_right_ends_y - landmarks[self.RIGHT_EYEBROW_CENTER][1]) / face_width)

        brow_arch = (left_brow_arch + right_brow_arch) / 2.0

        # ==========================================
        # 4. گروه سر و زاویه (تخمین 3 بعدی)
        # ==========================================
        dx_eyes = landmarks[self.RIGHT_EYE_OUTER][0] - landmarks[self.LEFT_EYE_OUTER][0]
        dy_eyes = landmarks[self.RIGHT_EYE_OUTER][1] - landmarks[self.LEFT_EYE_OUTER][1]
        head_roll = math.degrees(math.atan2(dy_eyes, dx_eyes))

        center_eye_x = (landmarks[self.LEFT_EYE_INNER][0] + landmarks[self.RIGHT_EYE_INNER][0]) / 2
        head_yaw = (landmarks[self.NOSE_TIP][0] - center_eye_x) / face_width

        center_eye_y = (landmarks[self.LEFT_EYE_INNER][1] + landmarks[self.RIGHT_EYE_INNER][1]) / 2
        head_pitch = (landmarks[self.NOSE_TIP][1] - center_eye_y) / face_height

        # ==========================================
        # 5. گروه گونه‌ها
        # ==========================================
        # بررسی فاصله گونه نسبت به گوشه چشم.
        # در هنگام خنده واقعی، عضلات گونه به سمت بالا و چشم کشیده می‌شوند.
        cheek_prominence_left = self.distance(landmarks[self.LEFT_CHEEK], landmarks[self.LEFT_EYE_OUTER]) / face_width
        cheek_prominence_right = self.distance(landmarks[self.RIGHT_CHEEK],
                                               landmarks[self.RIGHT_EYE_OUTER]) / face_width

        return {
            "mouth_openness": mouth_openness,
            "mouth_width": mouth_width,
            "mouth_ratio": mouth_ratio,
            "lip_elevation": lip_elevation,
            "lip_depression": lip_depression,
            "mouth_asymmetry": mouth_asymmetry,
            "teeth_visibility": teeth_visibility,
            "left_eye_openness": left_eye_openness,
            "right_eye_openness": right_eye_openness,
            "eye_difference": eye_difference,
            "left_brow_height": left_brow_height,
            "right_brow_height": right_brow_height,
            "brow_difference": brow_difference,
            "left_brow_tilt": left_brow_tilt,
            "right_brow_tilt": right_brow_tilt,
            "brow_arch": brow_arch,
            "brow_symmetry": brow_symmetry,
            "head_roll": head_roll,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "cheek_prominence_left": cheek_prominence_left,
            "cheek_prominence_right": cheek_prominence_right
        }