import cv2
import numpy as np
from collections import Counter
from PIL import Image, ImageFont, ImageDraw
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
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

    detector = FaceDetector()
    extractor = FeatureExtractor()
    engine = RuleEngine()

    frame_count = 0
    cached_text = "😐 99%"
    cached_raw_data = None
    emoji_history = []
    history_size = 5
    emoji_position = (50, 50)

    # متغیرهای جدید برای کالیبراسیون
    calibration_frames = 60  # کالیبراسیون در 60 فریم اول (حدود 2 تا 3 ثانیه)
    calibration_data = []
    is_calibrated = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        # پردازش فریم‌ها یک فریم در میان برای بهینه‌سازی سرعت
        if frame_count % 2 == 1:
            landmarks, raw_face_data = detector.get_landmarks(frame)
            cached_raw_data = raw_face_data

            if landmarks:
                # محاسبه موقعیت داینامیک بالای سر
                forehead_x = int(landmarks[10][0] * w)
                forehead_y = int(landmarks[10][1] * h)
                emoji_position = (forehead_x - 50, forehead_y - 80)

                features = extractor.extract_features(landmarks)
                if features:
                    # فاز کالیبراسیون
                    if not is_calibrated and len(calibration_data) < calibration_frames:
                        calibration_data.append(features)
                        cached_text = "Calibrating... Keep Neutral Face"

                        # اگر به فریم ۶۰ رسیدیم، میانگین بگیر و به موتور قوانین بده
                        if len(calibration_data) == calibration_frames:
                            avg_baselines = {}
                            for key in features.keys():
                                avg_baselines[key] = np.mean([f[key] for f in calibration_data])

                            engine.update_baselines(avg_baselines)
                            is_calibrated = True

                    # فاز اجرای عادی (بعد از کالیبراسیون)
                    elif is_calibrated:
                        emoji_probs = engine.get_probabilities(features)
                        best_emoji, best_prob = emoji_probs[0]

                        emoji_history.append(best_emoji)
                        if len(emoji_history) > history_size:
                            emoji_history.pop(0)

                        stable_emoji = Counter(emoji_history).most_common(1)[0][0]
                        cached_text = f"{stable_emoji} {best_prob}%"

        # رسم اجزا
        if cached_raw_data:
            frame = detector.draw_landmarks(frame, cached_raw_data)

        # تغییر رنگ متن در زمان کالیبراسیون به زرد برای جذابیت بصری
        text_color = (0, 255, 255) if not is_calibrated else (255, 255, 255)
        text_pos = (50, 50) if not is_calibrated else emoji_position
        size = 35 if not is_calibrated else 55

        frame = draw_text_with_pil(frame, cached_text, text_pos, font_size=size, color=text_color)

        cv2.imshow('Emoji Recommender - Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()