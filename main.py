import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine


def draw_text_with_pil(frame, text, position, font_size=50):
    """
    OpenCV cannot draw emojis natively. We use Pillow to draw Unicode characters.
    """
    # تبدیل فریم از BGR (استاندارد OpenCV) به RGB (استاندارد Pillow)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        # استفاده از فونت ایموجی پیش‌فرض ویندوز
        # اگر از مک یا لینوکس استفاده می‌کنید، مسیر فونت باید تغییر کند
        font = ImageFont.truetype("seguiemj.ttf", font_size)
    except IOError:
        # در صورتی که فونت پیدا نشد، از فونت پیش‌فرض ساده استفاده کن
        font = ImageFont.load_default()

    # رسم متن (ایموجی و درصد) روی تصویر
    # در نسخه‌های جدید Pillow، به جای جایگذاری مستقیم رنگ، از embedded_color استفاده می‌شود
    draw.text(position, text, font=font, fill=(255, 255, 255, 0), embedded_color=True)

    # برگرداندن تصویر به فرمت OpenCV
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame_bgr


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam!")
        return

    print("Webcam connected successfully. Press 'q' to exit.")

    detector = FaceDetector()
    extractor = FeatureExtractor()
    engine = RuleEngine()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        landmarks, raw_face_data = detector.get_landmarks(frame)

        # رسم شبکه نقاط صورت (اختیاری: می‌توانید این خط را کامنت کنید تا فقط ایموجی دیده شود)
        frame = detector.draw_landmarks(frame, raw_face_data)

        if landmarks:
            features = extractor.extract_features(landmarks)
            if features:
                emoji_probs = engine.get_probabilities(features)

                # گرفتن بهترین ایموجی (ایندکس 0 لیست مرتب‌شده)
                best_emoji, best_prob = emoji_probs[0]

                # ساخت متن نهایی
                display_text = f"{best_emoji} {best_prob}%"

                # چاپ روی فریم در مختصات (50, 50)
                frame = draw_text_with_pil(frame, display_text, (50, 50), font_size=60)

        cv2.imshow('Emoji Recommender - Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()