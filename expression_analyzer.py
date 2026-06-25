class ExpressionAnalyzer:
    def __init__(self):
        self.base = {
            "mouth_openness": 0.05, "mouth_width": 0.42, "mouth_ratio": 1.5,
            "lip_elevation": 0.0, "lip_depression": 0.0, "mouth_asymmetry": 0.0,
            "teeth_visibility": 0.0,
            "left_eye_openness": 0.045, "right_eye_openness": 0.045, "eye_difference": 0.0,
            "left_brow_height": 0.25, "right_brow_height": 0.25, "brow_difference": 0.0,
            "left_brow_tilt": -0.02, "right_brow_tilt": -0.02,
            "brow_arch": 0.0, "brow_symmetry": 0.0,
            "head_roll": 0.0, "head_yaw": 0.0, "head_pitch": 0.0,
            "cheek_prominence_left": 0.3, "cheek_prominence_right": 0.3
        }

    def update_baselines(self, calibrated_baselines):
        self.base.update(calibrated_baselines)

    def analyze(self, f):
        b = self.base

        # ۱. شاخص‌های خنده
        smile_mwr = max(0, (f["mouth_width"] - b["mouth_width"]) * 10)
        smile_elev = max(0, (f["lip_elevation"] - b["lip_elevation"]) * 20)
        base_smile_index = (0.6 * smile_mwr) + (0.4 * smile_elev)

        avg_cheek_base = (b["cheek_prominence_left"] + b["cheek_prominence_right"]) / 2
        avg_cheek_curr = (f["cheek_prominence_left"] + f["cheek_prominence_right"]) / 2
        cheek_raise = max(0, (avg_cheek_base - avg_cheek_curr) * 20)

        avg_eye_base = (b["left_eye_openness"] + b["right_eye_openness"]) / 2
        avg_eye_curr = (f["left_eye_openness"] + f["right_eye_openness"]) / 2
        eye_squeeze = max(0, (avg_eye_base - avg_eye_curr) * 15)

        duchenne_index = (0.6 * cheek_raise) + (0.4 * eye_squeeze)

        teeth_shown = max(0, (f["teeth_visibility"] - b["teeth_visibility"]) * 15)
        mouth_open_smile = max(0, (f["mouth_openness"] - b["mouth_openness"]) * 8)
        laugh_teeth_index = (0.6 * teeth_shown) + (0.4 * mouth_open_smile)

        smile_index = max(base_smile_index, duchenne_index, laugh_teeth_index)

        # ۲. شاخص تعجب
        surp_mouth_open = max(0, (f["mouth_openness"] - b["mouth_openness"]) * 5)
        surp_mouth_round = max(0, (b["mouth_ratio"] - f["mouth_ratio"]) * 2)
        surp_eye_wide = max(0, (avg_eye_curr - avg_eye_base) * 15)

        avg_brow_base = (b["left_brow_height"] + b["right_brow_height"]) / 2
        avg_brow_curr = (f["left_brow_height"] + f["right_brow_height"]) / 2
        surp_brow_raise = max(0, (avg_brow_curr - avg_brow_base) * 15)

        surprise_index = (0.40 * surp_mouth_open) + (0.15 * surp_mouth_round) + \
                         (0.30 * surp_eye_wide) + (0.15 * surp_brow_raise)

        # ۳. شاخص چشمک
        wink_eye_diff = max(0, (f["eye_difference"] - b["eye_difference"]) * 15)
        wink_head_roll = min(abs(f["head_roll"]) / 30, 1.0)
        wink_index = (0.85 * wink_eye_diff) + (0.15 * wink_head_roll)

        # ۴. شاخص اخم
        anger_brow_drop = max(0, (avg_brow_base - avg_brow_curr) * 15)
        avg_tilt_base = (b["left_brow_tilt"] + b["right_brow_tilt"]) / 2
        avg_tilt_curr = (f["left_brow_tilt"] + f["right_brow_tilt"]) / 2
        anger_brow_tilt = max(0, (avg_tilt_base - avg_tilt_curr) * 20)
        anger_lip_down = max(0, (f["lip_depression"] - b["lip_depression"]) * 20)

        anger_index = (0.4 * anger_brow_drop) + (0.3 * anger_brow_tilt) + (0.3 * anger_lip_down)

        # ۵. شاخص غم
        sad_lip_down = max(0, (f["lip_depression"] - b["lip_depression"]) * 25)
        sad_eye_drop = max(0, (avg_eye_base - avg_eye_curr) * 10)
        sad_brow_arch = max(0, (f["brow_arch"] - b["brow_arch"]) * 10)

        sadness_index = (0.60 * sad_lip_down) + (0.25 * sad_eye_drop) + (0.15 * sad_brow_arch)

        # ۶. شاخص بوسه
        kiss_mwr_drop = max(0, (b["mouth_width"] - f["mouth_width"]) * 10)
        kiss_mouth_open = max(0, (f["mouth_openness"] - b["mouth_openness"]) * 10)
        kiss_index = (0.6 * kiss_mwr_drop) + (0.4 * kiss_mouth_open)

        # ۷. شاخص شک/کنجکاوی 🤨
        suspicion_index = max(0, (f["brow_difference"] - b["brow_difference"]) * 15)

        # =========================================================================
        # شاخص‌های مستقل (اصلاح شده با منطق ضربی برای جلوگیری از تداخل)
        # =========================================================================

        # 👈 شاخص چشمک واقعی (استفاده شده در 😜)
        left_eye_closed = max(0, (b["left_eye_openness"] - f["left_eye_openness"]) * 20)
        right_eye_closed = max(0, (b["right_eye_openness"] - f["right_eye_openness"]) * 20)
        true_wink_index = max(left_eye_closed, right_eye_closed)

        # 👈 شاخص زبان‌درازی (استفاده شده در 😜)
        mouth_zone = max(0, 1.0 - abs(f["mouth_openness"] - 0.15) * 5)
        ratio_drop = max(0, (b["mouth_ratio"] - f["mouth_ratio"]) * 2)
        mouth_asym_effect = max(0, (f["mouth_asymmetry"] - b["mouth_asymmetry"]) * 10)
        tongue_mischief_index = (mouth_zone * ratio_drop) + (0.2 * mouth_asym_effect)

        # 🤪 مجنون: استفاده از ضرب. سر کج * (چشم یا دهان نامتقارن). اگر یکی صفر باشد، کل خروجی صفر است.
        head_tilt = min(abs(f["head_roll"]) / 15, 1.0)
        face_asym = min((f["eye_difference"] * 10) + (f["mouth_asymmetry"] * 10), 1.0)
        crazy_face_index = head_tilt * face_asym

        # 👈 مظلوم 🥺: ترکیب درشت شدن چشم و ابروی هلالی
        eye_wide = max(0, (f["left_eye_openness"] + f["right_eye_openness"]) / 2.0 - (
                    b["left_eye_openness"] + b["right_eye_openness"]) / 2.0) * 15
        arch_up = max(0, f["brow_arch"] - b["brow_arch"]) * 15
        puppy_eyes_index = (0.5 * eye_wide) + (0.5 * arch_up)

        # 😑 بی‌حوصله
        eye_squint = max(0, (b["left_eye_openness"] + b["right_eye_openness"]) / 2.0 - (
                    f["left_eye_openness"] + f["right_eye_openness"]) / 2.0) * 20
        bored_flat_index = max(0, eye_squint - max(0, cheek_raise))

        # 😮‍💨 آه کشیدن: دهان فوت کردن * چشم خمار. (بوسه چشم خمار ندارد، پس از هم جدا می‌شوند)
        mouth_o_shape = max(0, (b["mouth_ratio"] - f["mouth_ratio"]) * 3)
        eyes_tired = max(0, avg_eye_base - avg_eye_curr) * 15
        sigh_exhausted_index = min(mouth_o_shape * eyes_tired, 1.0)

        # 😭 گریه شدید: دهان باز * چشم فشرده. (خمیازه چشم فشرده ندارد، پس اشتباه نمی‌شود)
        cry_mouth = max(0, f["mouth_openness"] - b["mouth_openness"]) * 8
        cry_eye_squeeze = max(0, avg_eye_base - avg_eye_curr) * 15
        intense_cry_index = min(cry_mouth * cry_eye_squeeze, 1.0)

        # 👈 😂 خنده شدید: ضرب خنده پایه در بسته شدن چشم‌ها
        tear_laugh_index = min(base_smile_index, 1.0) * min(eye_squeeze, 1.0)

        return {
            "smile_index": min(smile_index, 1.0),
            "surprise_index": min(surprise_index, 1.0),
            "wink_index": min(wink_index, 1.0),
            "anger_index": min(anger_index, 1.0),
            "sadness_index": min(sadness_index, 1.0),
            "kiss_index": min(kiss_index, 1.0),
            "suspicion_index": min(suspicion_index, 1.0),
            "raw_mouth_openness": f["mouth_openness"],
            "base_mouth_openness": b["mouth_openness"],
            "true_wink_index": min(true_wink_index, 1.0),
            "tongue_mischief_index": min(tongue_mischief_index, 1.0),
            "crazy_face_index": min(crazy_face_index, 1.0),
            "puppy_eyes_index": min(puppy_eyes_index, 1.0),
            "sigh_exhausted_index": min(sigh_exhausted_index, 1.0),
            "bored_flat_index": min(bored_flat_index, 1.0),
            "intense_cry_index": min(intense_cry_index, 1.0),
            "tear_laugh_index": min(tear_laugh_index, 1.0)
        }