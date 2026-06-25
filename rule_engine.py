class RuleEngine:
    def __init__(self):
        # 10 ایموجی پشتیبانی‌شده در سیستم
        self.emojis = ["😐", "😲", "😉", "😠", "😄", "😗", "☹️", "🤨", "🥱", "😳"]

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

        # 1. تعجب 😲 (دهان کمی تا متوسط باز است)
        if 0.15 < mar < 0.5:
            scores["😲"] = min(max(int(((mar - 0.15) / 0.35) * 100), 0), 99)

        # 2. خمیازه 🥱 (دهان به شدت باز است - بیشتر از 0.5)
        if mar >= 0.5:
            scores["🥱"] = min(max(int(((mar - 0.5) / 0.5) * 100), 0), 99)

        # 3. چشمک 😉
        eye_diff = abs(r_ear - l_ear)
        if eye_diff > 0.08:
            scores["😉"] = min(max(int(((eye_diff - 0.08) / 0.15) * 100), 0), 99)

        # 4. خیره شدن/تعجب شدید 😳 (چشم‌ها بیشتر از حد معمول باز هستند)
        # میانگین EAR معمولاً 0.35 است.
        if avg_ear > 0.42:
            scores["😳"] = min(max(int(((avg_ear - 0.42) / 0.10) * 100), 0), 99)

        # 5. یک ابرو بالا / شک 🤨 (عدم تقارن ابروها)
        # در حالت عادی تفاوت دو ابرو زیر 0.015 است.
        if brow_asymmetry > 0.025:
            scores["🤨"] = min(max(int(((brow_asymmetry - 0.025) / 0.05) * 100), 0), 99)

        # 6. اخم 😠
        if brows < 1.6:
            scores["😠"] = min(max(int(((1.6 - brows) / 0.4) * 100), 0), 99)

        # 7. لبخند 😄
        if mwr > 0.48 or lip_turn > 0.01:
            smile_score = 0
            if mwr > 0.48: smile_score += ((mwr - 0.48) / 0.12) * 50
            if lip_turn > 0.01: smile_score += ((lip_turn - 0.01) / 0.06) * 50
            scores["😄"] = min(max(int(smile_score), 0), 99)

        # 8. بوسه 😗
        if mwr < 0.38 and mar < 0.15:
            scores["😗"] = min(max(int(((0.38 - mwr) / 0.12) * 100), 0), 99)

        # 9. غم ☹️
        if lip_turn < -0.01:
            scores["☹️"] = min(max(int(((-0.01 - lip_turn) / 0.06) * 100), 0), 99)

        # 10. خنثی 😐
        max_other_score = max([scores[e] for e in self.emojis if e != "😐"])
        scores["😐"] = max(99 - max_other_score, 0)

        sorted_emojis = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_emojis