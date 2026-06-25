class RuleEngine:
    def __init__(self):
        # هسته پایدار شما + اضافه شدن فاز اول ایموجی‌های جدید و مستقل
        self.emoji_profiles = {
            # ===== گروه اول: هسته پایدار اصلی (بدون تغییر) =====
            "😄": {
                "smile_index": 0.80,
                "surprise_index": -0.20,
                "anger_index": -0.30,
                "sadness_index": -0.30,
                "kiss_index": -0.20
            },
            "😉": {
                "wink_index": 0.90,
                "smile_index": 0.20
            },
            "😲": {
                "surprise_index": 1.00,
                "smile_index": -0.20
            },
            "😠": {
                "anger_index": 1.00,
                "smile_index": -0.30
            },
            "☹️": {
                "sadness_index": 1.00,
                "smile_index": -0.40
            },
            "😗": {
                "kiss_index": 1.00,
                "smile_index": -0.20
            },
            "🤨": {
                "suspicion_index": 1.00
            },

            # ===== گروه دوم: ایموجی‌های جدید با شاخص‌های ایزوله =====
            "🤪": {
                "crazy_face_index": 1.00,
                "smile_index": -0.20  # برای جلوگیری از تداخل با خنده عادی
            },
            "😭": {
                "intense_cry_index": 1.00,
                "smile_index": -0.30
            },
            "😮‍💨": {
                "sigh_exhausted_index": 1.00,
                "smile_index": -0.20,
                "anger_index": -0.20
            }
        }

    def score_emoji(self, indices, profile):
        """محاسبه ضرب داخلی شاخص‌ها در وزن‌های پروفایل هر ایموجی"""
        score = 0.0
        for feature_name, weight in profile.items():
            value = indices.get(feature_name, 0.0)
            score += value * weight
        return max(score, 0.0)

    def get_probabilities(self, indices):
        scores = {}
        activation_threshold = 0.20  # آستانه ددزون برای فیلتر نویزها

        # ۱. محاسبه امتیاز اولیه و اعمال فیلتر آستانه فعال‌سازی
        for emoji, profile in self.emoji_profiles.items():
            raw_score = self.score_emoji(indices, profile)
            scores[emoji] = raw_score if raw_score >= activation_threshold else 0.0

        # ۲. مدیریت اختصاصی ایموجی خمیازه 🥱
        raw_open = indices.get("raw_mouth_openness", 0.0)
        base_open = indices.get("base_mouth_openness", 0.0)
        if raw_open > base_open + 0.25:
            yawn_score = min((raw_open - (base_open + 0.25)) * 4, 1.0)
            scores["🥱"] = yawn_score if yawn_score >= activation_threshold else 0.0
            if "😲" in scores and scores["🥱"] > 0:
                scores["😲"] *= 0.2
        else:
            scores["🥱"] = 0.0

        # ۳. محاسبه پویا و مقتدرانه امتیاز پوکر فیس (😐)
        max_emotion_score = max(scores.values()) if scores else 0.0
        scores["😐"] = max(0.0, 0.35 - max_emotion_score)

        # ۴. محاسبه کل امتیازات
        total_score = sum(scores.values())

        if total_score <= 0:
            return [
                ("😐", 100), ("😄", 0), ("😉", 0), ("😲", 0),
                ("😠", 0), ("☹️", 0), ("😗", 0), ("🤨", 0), ("🥱", 0),
                ("🤪", 0), ("😭", 0), ("😮‍💨", 0)
            ]

        # ۵. محاسبه درصدها
        probabilities = {}
        for emoji, score in scores.items():
            probabilities[emoji] = int((score / total_score) * 100)

        # مرتب‌سازی نزولی برای خروجی نهایی
        sorted_emojis = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_emojis