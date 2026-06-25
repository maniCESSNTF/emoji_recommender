import cv2
import numpy as np
from collections import Counter
from PIL import Image, ImageFont, ImageDraw
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine


def draw_text_with_pil(frame, text, position, font_size=50):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("seguiemj.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(255, 255, 255, 0), embedded_color=True)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam!")
        return

    print("Webcam connected. Dynamic Tracking & Smoothing Active. Press 'q' to exit.")

    detector = FaceDetector()
    extractor = FeatureExtractor()
    engine = RuleEngine()

    frame_count = 0
    cached_text = "😐 99%"
    cached_raw_data = None

    # بافر تاریخچه برای پایداری ایموجی‌ها و جلوگیری از پرش
    emoji_history = []
    history_size = 5  # تعداد فریم‌های مورد نظر برای رای‌گیری

    # مختصات پیشانی برای تعقیب پویا
    emoji_position = (50, 50)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        if frame_count % 2 == 1:
            landmarks, raw_face_data = detector.get_landmarks(frame)
            cached_raw_data = raw_face_data

            if landmarks:
                # ۱. محاسبه موقعیت پویای پیشانی (نقطه 10 مدیاپایپ)
                # تبدیل مختصات 0 تا 1 به پیکسل‌های واقعی تصویر
                forehead_x = int(landmarks[10][0] * w)
                forehead_y = int(landmarks[10][1] * h)

                # تنظیم آفست برای قرارگیری ایموجی کمی بالاتر از سر و در مرکز
                emoji_position = (forehead_x - 50, forehead_y - 80)

                # ۲. استخراج ویژگی‌ها و پردازش در موتور قوانین
                features = extractor.extract_features(landmarks)
                if features:
                    emoji_probs = engine.get_probabilities(features)
                    best_emoji, best_prob = emoji_probs[0]

                    # اضافه کردن ایموجی فعلی به تاریخچه
                    emoji_history.append(best_emoji)
                    if len(emoji_history) > history_size:
                        emoji_history.pop(0)  # حذف قدیمی‌ترین فریم

                    # ۳. رای‌گیری اکثریت برای انتخاب پایدارترین ایموجی
                    majority_counter = Counter(emoji_history)
                    stable_emoji = majority_counter.most_common(1)[0][0]

                    cached_text = f"{stable_emoji} {best_prob}%"

        # رسم المان‌ها روی تصویر
        if cached_raw_data:
            frame = detector.draw_landmarks(frame, cached_raw_data)

        # چاپ ایموجی در موقعیت داینامیک پیشانی به جای نقطه ثابت بالای صفحه
        frame = draw_text_with_pil(frame, cached_text, emoji_position, font_size=55)

        cv2.imshow('Emoji Recommender - Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()