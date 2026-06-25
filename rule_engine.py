class RuleEngine:
    def __init__(self):
        self.emojis = ["😐", "😲", "😉", "😠", "😄", "😗", "☹️", "🤨", "🥱", "😳"]

        # مقادیر پایه پیش‌فرض (اگر کالیبراسیون انجام نشود)
        self.base = {
            "mar": 0.05,
            "mwr": 0.42,
            "lip_turn": 0.0,
            "avg_ear": 0.35,
            "eyebrow_ratio": 1.8
        }

    def update_baselines(self, calibrated_baselines):
        """
        دریافت مقادیر واقعی چهره کاربر در حالت خنثی
        """
        self.base.update(calibrated_baselines)
        print(f"[RuleEngine] Baselines updated dynamically: {self.base}")

    def get_probabilities(self, features):
        mar = features['mar']
        mwr = features['mwr']
        lip_turn = features['lip_turn']
        r_ear = features['right_ear']
        l_ear = features['left_ear']
        avg_ear = features['avg_ear']
        brows = features['eyebrow_ratio']
        brow_asymmetry = features['brow_asymmetry']

        scores = {emoji: 0 for emoji in self.emojis}

        # ۱. تعجب 😲 (دهان بازتر از حالت خنثی خود کاربر)
        if mar > self.base["mar"] + 0.10:
            scores["😲"] = min(max(int(((mar - (self.base["mar"] + 0.10)) / 0.35) * 100), 0), 99)

        # ۲. خمیازه 🥱 (دهان خیلی بازتر از حالت خنثی)
        if mar > self.base["mar"] + 0.45:
            scores["🥱"] = min(max(int(((mar - (self.base["mar"] + 0.45)) / 0.4) * 100), 0), 99)

        # ۳. چشمک 😉
        eye_diff = abs(r_ear - l_ear)
        if eye_diff > 0.08:
            scores["😉"] = min(max(int(((eye_diff - 0.08) / 0.15) * 100), 0), 99)

        # ۴. خیره شدن 😳 (چشم‌ها بازتر از حالت عادی کاربر)
        if avg_ear > self.base["avg_ear"] + 0.05:
            scores["😳"] = min(max(int(((avg_ear - (self.base["avg_ear"] + 0.05)) / 0.10) * 100), 0), 99)

        # ۵. شک/کنجکاوی 🤨
        if brow_asymmetry > 0.025:
            scores["🤨"] = min(max(int(((brow_asymmetry - 0.025) / 0.05) * 100), 0), 99)

        # ۶. اخم 😠 (ابروها نزدیک‌تر از حالت خنثی کاربر)
        if brows < self.base["eyebrow_ratio"] - 0.15:
            scores["😠"] = min(max(int((((self.base["eyebrow_ratio"] - 0.15) - brows) / 0.4) * 100), 0), 99)

        # ۷. لبخند 😄 (عرض دهان بزرگتر از خنثی و گوشه‌ها رو به بالا)
        smile_threshold = self.base["mwr"] + 0.04
        if mwr > smile_threshold or lip_turn > self.base["lip_turn"] + 0.01:
            smile_score = 0
            if mwr > smile_threshold:
                smile_score += ((mwr - smile_threshold) / 0.10) * 50
            if lip_turn > self.base["lip_turn"] + 0.01:
                smile_score += ((lip_turn - (self.base["lip_turn"] + 0.01)) / 0.05) * 50
            scores["😄"] = min(max(int(smile_score), 0), 99)

        # ۸. بوسه 😗 (عرض دهان کوچک‌تر از حالت خنثی)
        kiss_threshold = self.base["mwr"] - 0.04
        if mwr < kiss_threshold and mar < self.base["mar"] + 0.10:
            scores["😗"] = min(max(int(((kiss_threshold - mwr) / 0.10) * 100), 0), 99)

        # ۹. غم ☹️ (گوشه‌های لب پایین‌تر از حالت خنثی)
        if lip_turn < self.base["lip_turn"] - 0.01:
            scores["☹️"] = min(max(int((((self.base["lip_turn"] - 0.01) - lip_turn) / 0.05) * 100), 0), 99)

        # ۱۰. خنثی 😐
        max_other_score = max([scores[e] for e in self.emojis if e != "😐"])
        scores["😐"] = max(99 - max_other_score, 0)

        sorted_emojis = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_emojis