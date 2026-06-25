import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from expression_analyzer import ExpressionAnalyzer
from rule_engine import RuleEngine


def draw_text_with_pil(frame, text, position, font_size=50, color=(255, 255, 255)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("seguiemj.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color, embedded_color=True)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam!")
        return

    print("Webcam connected. Starting complete Vector Space Model loop...")

    detector = FaceDetector()
    extractor = FeatureExtractor()
    analyzer = ExpressionAnalyzer()
    engine = RuleEngine()

    frame_count = 0
    cached_text = "😐 100%"
    cached_raw_data = None

    # دیکشنری برای نرم کردن (Smoothing) درصد ایموجی‌ها
    smoothed_probs = {}

    # فاز کالیبراسیون هوشمند 3 ثانیه‌ای (60 فریم)
    calibration_frames = 60
    calibration_data = []
    is_calibrated = False

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
                # مرحله 1: استخراج ویژگی‌های هندسی توسعه‌یافته
                raw_features = extractor.extract_features(landmarks)

                if raw_features:
                    # فاز اسکن و کالیبراسیون چهره خنثی کاربر
                    if not is_calibrated and len(calibration_data) < calibration_frames:
                        calibration_data.append(raw_features)
                        cached_text = "Calibrating...\nKeep Neutral Face"

                        if len(calibration_data) == calibration_frames:
                            avg_baselines = {}
                            for key in raw_features.keys():
                                avg_baselines[key] = np.mean([f[key] for f in calibration_data])

                            # کالیبره کردن لایه تحلیل‌گر بر اساس چهره اختصاصی شما
                            analyzer.update_baselines(avg_baselines)
                            is_calibrated = True

                    # فاز اجرای عادی پس از اتمام کالیبراسیون
                    elif is_calibrated:
                        # مرحله 2: تبدیل ویژگی‌های خام به شاخص‌های حسی کالیبره‌شده
                        emotion_indices = analyzer.analyze(raw_features)

                        # مرحله 3: اجرای مدل ضرب برداری و گرفتن درصد احتمالات (تمامی ایموجی‌ها)
                        emoji_probs = engine.get_probabilities(emotion_indices)

                        current_probs_dict = dict(emoji_probs)

                        # اعمال میانگین متحرک نمایی (EMA) برای نرم شدن تغییرات درصدها
                        if not smoothed_probs:
                            smoothed_probs = current_probs_dict
                        else:
                            for k in current_probs_dict:
                                smoothed_probs[k] = int(0.6 * smoothed_probs.get(k, 0) + 0.4 * current_probs_dict[k])

                        # مرتب‌سازی مجدد بعد از نرم‌سازی
                        sorted_smoothed = sorted(smoothed_probs.items(), key=lambda item: item[1], reverse=True)

                        # استخراج 5 ایموجی برتر
                        top_5 = sorted_smoothed[:5]

                        # ساخت متن چندخطی برای نمایش در تصویر
                        cached_text = "Top 5 Predictions:\n"
                        for idx, (emoji, prob) in enumerate(top_5):
                            cached_text += f"{idx + 1}. {emoji} {prob}%\n"

        if cached_raw_data:
            frame = detector.draw_landmarks(frame, cached_raw_data)

        # مدیریت رنگ، موقعیت و فونت متن (پنل ثابت در گوشه تصویر)
        text_color = (0, 255, 255) if not is_calibrated else (255, 255, 255)
        text_pos = (30, 30)  # قفل کردن موقعیت در گوشه سمت چپ بالا برای جلوگیری از پرش متن
        size = 35 if not is_calibrated else 40

        frame = draw_text_with_pil(frame, cached_text, text_pos, font_size=size, color=text_color)

        cv2.imshow('Emoji Recommender - Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()