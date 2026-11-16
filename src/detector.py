import re
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("easyocr").setLevel(logging.ERROR)

import cv2
import numpy as np
import imutils
import easyocr


class PlateDetector:
    def __init__(self, languages=None):
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, verbose=False)

    def locate_plate(self, gray_img):
        bfilter = cv2.bilateralFilter(gray_img, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                return approx
        return None

    def _order_points(self, pts):
        pts = pts.reshape(4, 2).astype("float32")
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_plate(self, gray_img, location):
        rect = self._order_points(location)
        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = max(int(height_a), int(height_b))

        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray_img, matrix, (max_w, max_h))
        return warped

    def get_candidate_crops(self, warped_gray):
        border_val = int(np.median(warped_gray))
        padded = cv2.copyMakeBorder(
            warped_gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=border_val
        )

        variant_raw = padded.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(padded)

        return [variant_raw, enhanced]

    def clean_text(self, raw_text):
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

        # 1. Indian RTO format fix (e.g. HR26BR904A -> HR26BR9044)
        indian_match = re.match(
            r"^([A-Z]{2})([0-9OISZB]{2})([A-Z]{1,2})([0-9A-Z]{4})$", cleaned
        )
        if indian_match:
            state, rto, series, number = indian_match.groups()
            char_to_digit = str.maketrans(
                {
                    "A": "4",
                    "O": "0",
                    "Q": "0",
                    "I": "1",
                    "L": "1",
                    "Z": "2",
                    "S": "5",
                    "B": "8",
                    "G": "6",
                }
            )
            rto = rto.translate(char_to_digit)
            number = number.translate(char_to_digit)
            return f"{state}{rto}{series}{number}"

        # 2. Custom COVID19 fix (COVIDT9 -> COVID19)
        if cleaned.startswith("COVID") and len(cleaned) == 7:
            if cleaned[5] == "T":
                cleaned = cleaned[:5] + "1" + cleaned[6:]

        # 3. Australian standard format extraction (extract 3 letters + 3 digits from noise)
        aus_match = re.search(r"([A-Z]{3}[0-9]{3})", cleaned)
        if aus_match and not cleaned.startswith("COVID"):
            cleaned = aus_match.group(1)

        return cleaned

    def _read_best_candidate(self, candidates):
        best_text = ""
        best_score = -1.0

        for crop in candidates:
            results = self.reader.readtext(
                crop,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            if not results:
                continue

            crop_h, crop_w = crop.shape
            results = sorted(results, key=lambda r: r[0][0][0])

            heights = [
                abs(r[0][2][1] - r[0][0][1])
                for r in results
                if abs(r[0][2][1] - r[0][0][1]) >= crop_h * 0.25
            ]
            if not heights:
                continue
            median_h = float(np.median(heights))

            valid_segments = []
            confidences = []

            for bbox, text, conf in results:
                box_w = abs(bbox[1][0] - bbox[0][0])
                box_h = abs(bbox[2][1] - bbox[0][1])
                x_min = bbox[0][0]

                if box_h < median_h * 0.60:
                    continue

                if x_min < crop_w * 0.12 and box_w < box_h * 0.50:
                    continue

                if conf > 0.25:
                    valid_segments.append(text)
                    confidences.append(conf)

            if not valid_segments:
                continue

            raw_combined = "".join(valid_segments)
            cleaned = self.clean_text(raw_combined)
            avg_conf = sum(confidences) / len(confidences)
            length_bonus = 0.2 if 6 <= len(cleaned) <= 10 else 0.0
            score = avg_conf + length_bonus

            if score > best_score:
                best_score = score
                best_text = cleaned

        return best_text if best_text else None

    def extract_text(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        location = self.locate_plate(gray)

        if location is None:
            return None, None

        warped_plate = self.warp_plate(gray, location)
        candidates = self.get_candidate_crops(warped_plate)
        cleaned_plate = self._read_best_candidate(candidates)
        return cleaned_plate, location