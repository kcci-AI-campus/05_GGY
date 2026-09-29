import os
import time

import cv2

from box_detect_onboard_module import BoxDetector
from damage_detect_module import DamageDetector


class BoxVideoChecker:
    def __init__(
        self,
        box_class_id=0,
        confidence_threshold=0.4,
        check_seconds=1,
        detect_frame_count=5,
        frame_interval=3,
        damage_threshold=0.5,
    ):
        self.box_class_id = box_class_id
        self.confidence_threshold = confidence_threshold
        self.check_seconds = check_seconds
        self.detect_frame_count = detect_frame_count
        self.frame_interval = frame_interval

        # 박스탐지 모델 객체 생성
        self.model = BoxDetector(
            confidence_threshold=confidence_threshold,
            class_id=box_class_id,
        )
        # 손상탐지 모델 객체 생성
        self.damage_model = DamageDetector(
            model_path="damage_detect_v2_float32.tflite",
            damaged_class_id=0,
            damage_threshold=damage_threshold,
        )

        # 손상탐지 추론 시간과 박스탐지-손상탐지까지 걸린 시간 기록용 리스트
        self.damage_inference_times = []
        self.box_to_damage_times = []
        # 디버그용 파손 판별 모델의 추론시간 출력
    def get_damage_timing_summary(self):
        damage = self.damage_inference_times
        total = self.box_to_damage_times

        if not damage:
            return {
                "count": 0,
                "average_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "box_to_damage_average_ms": 0.0,
                "box_to_damage_min_ms": 0.0,
                "box_to_damage_max_ms": 0.0,
            }

        return {
            "count": len(damage),
            "average_ms": sum(damage) / len(damage),
            "min_ms": min(damage),
            "max_ms": max(damage),
            "box_to_damage_average_ms": sum(total) / len(total) if total else 0.0,
            "box_to_damage_min_ms": min(total) if total else 0.0,
            "box_to_damage_max_ms": max(total) if total else 0.0,
        }

    # 영상길이,영상에서 박스를 탐지랑 시간구간 계산
    def _read_check_frames(self, cap, duration):
        check_duration = min(self.check_seconds, duration)
        start_time = max(0.0, duration - check_duration)

        print(f"Video duration : {duration:.2f} sec")
        print(f"Check duration : {check_duration:.2f} sec")

        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        return frames

    #박스 탐지 모델을 불러와 박스탐지후 TH를 넘긴 box class후보들을 값을 저장
    def _detect_boxes(self, frame):
        detections = self.model(frame)
        boxes = []

        for x1, y1, x2, y2, confidence, class_id in detections:
            if class_id != self.box_class_id:
                continue
            if confidence < self.confidence_threshold:
                continue

            boxes.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(confidence),
                "class_id": int(class_id),
            })

        return boxes

    # 영상의 뒤에서부터 interval만큼 건너뛰면서 프레임속 박스 탐지 하여 박스가 있는 마지막 프레임과 박스값을 return
    def _get_latest_detection(self, frames):
        # 마지막 프레임부터 일정 간격으로 최대 N개만 확인한다.
        # 최종 상태는 가장 최근 프레임으로 결정

        #영상에서 탐지된 값과 프레임을 저장함
        samples = []
        index = len(frames) - 1

        #박스가 탐지된 프레임의 수가 detect_frame_count보다 작으면, frame_interval만큼 건너뛰면서 탐지된 프레임을 samples에 저장
        while index >= 0 and len(samples) < self.detect_frame_count:
            boxes = self._detect_boxes(frames[index])
            samples.append((index, frames[index], boxes))
            index -= max(1, self.frame_interval)

        if not samples:
            return None, None

        latest_index, latest_frame, latest_boxes = samples[0]

        print(
            f"[YOLO] Latest frame {latest_index + 1}/{len(frames)} : "
            f"{len(latest_boxes)} box"
        )

        for sample_index, _, boxes in reversed(samples):
            print(
                f"[YOLO] frame {sample_index + 1}/{len(frames)} : "
                f"{len(boxes)} box"
            )

        # 최신 프레임이 NO BOX이면 과거 프레임의 검출 결과를 사용하지 않는다.
        if not latest_boxes:
            return latest_frame, []

        return latest_frame, latest_boxes

    #사용하는 인스턴스 변수가없는 staticmethod
    #바운딩 박스 좌표를 frame의 크기에 맞게 설정
    @staticmethod
    def _clip_bbox(bbox, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))

        return x1, y1, x2, y2

    #박스 후보를 받아서 손상탐지 모델을 통해 손상 여부를 판단하고 결과를 반환 
    def _check_damage(self, frame, box_index, box):
        started = time.perf_counter()

        x1, y1, x2, y2 = self._clip_bbox(
            box["bbox"],
            frame.shape,
        )

        if x2 <= x1 or y2 <= y1:
            print(f"[BOX {box_index}] 잘못된 BBox")
            return None

        # 상자의 위치로 frame을 crop
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[BOX {box_index}] Crop 실패")
            return None

        damage_start = time.perf_counter()
        # 손상탐지 모델을 통해 결과를 damage에 저장
        try:
            damage = self.damage_model(crop)
        except Exception as e:
            print(f"[BOX {box_index}] Damage Detector ERROR: {type(e).__name__}: {e}")
            return None
        damage_ms = (time.perf_counter() - damage_start) * 1000

        total_ms = (time.perf_counter() - started) * 1000
        self.damage_inference_times.append(damage_ms)
        self.box_to_damage_times.append(total_ms)

        # 전달받은 박스의 위치,스코어 점수와 훼손판별 모델로 확인된 결과를 return
        return {
            "box_id": box_index,
            "bbox": (x1, y1, x2, y2),
            "confidence": box["confidence"],
            "damaged": bool(damage["damaged"]),
            "damage_confidence": float(damage["confidence"]),
            "damage_score": float(damage["damage_score"]),
            "normal_score": float(damage["normal_score"]),
            "damage_class_id": int(damage["class_id"]),
        }

    # 위에 함수들을 불러오고, 영상이 제대로있는지 확인 후 박스,훼손여부 판별후 결과 저장
    def check(self, video_path):
        print("\n" + "=" * 70)
        print("[Box Check]")
        print(f"Video : {video_path}")
        print("=" * 70)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"영상 열기 실패 : {video_path}")
            return {"detected": False, "boxes": []}

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0 or frame_count <= 0:
                print("영상 정보를 가져올 수 없습니다.")
                return {"detected": False, "boxes": []}

            #불러온 영상의 길이와 검사할 구간을 계산하여 프레임을 읽어옴
            duration = frame_count / fps
            frames = self._read_check_frames(cap, duration)
        finally:
            cap.release()

        if not frames:
            print("검사할 프레임이 없습니다.")
            return {"detected": False, "boxes": []}

        #영상에서 탐지된 마지막 박스의 프레임과 박스값을 저장
        frame, boxes = self._get_latest_detection(frames)

        if frame is None or not boxes:
            print("\nFinal result : NO BOX")
            return {"detected": False, "boxes": []}

        # bbox를 x좌표기준으로 정렬
        boxes.sort(key=lambda box: box["bbox"][0])

        #탐지된 박스들중 훼손여부를 판별해 저장
        final_boxes = []
        for index, box in enumerate(boxes, start=1):
            result = self._check_damage(frame, index, box)
            if result is not None:
                final_boxes.append(result)

        print("\n" + "=" * 70)
        print(f"FINAL RESULT : {len(final_boxes)} BOX")
        print("=" * 70)

        return {
            "detected": bool(final_boxes),
            "boxes": final_boxes,
        }

    @staticmethod
    def delete_video(video_path):
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"영상 삭제 : {video_path}")
                return True
        except OSError as e:
            print(f"영상 삭제 실패 : {e}")
        return False

    def check_and_delete(self, video_path):
        result = self.check(video_path)

        if not result.get("detected", False):
            self.delete_video(video_path)

        return result
