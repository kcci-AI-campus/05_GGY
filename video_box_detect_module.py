# video_box_detect_module.py

import cv2
import os

from box_detect_onboard_module import BoxDetector
from damage_detect_module import DamageDetector


class BoxVideoChecker:

    def __init__(
        self,
        box_class_id=0,
        confidence_threshold=0.4,
        check_seconds=3,
        damage_confidence_threshold=0.5
    ):

        self.box_class_id = box_class_id

        self.confidence_threshold = \
            confidence_threshold

        self.check_seconds = check_seconds

        # --------------------------------------------------
        # Box Detector
        # --------------------------------------------------

        self.model = BoxDetector(
            confidence_threshold=self.confidence_threshold,
            class_id=self.box_class_id
        )

        # --------------------------------------------------
        # Damage Detector
        # --------------------------------------------------

        self.damage_model = DamageDetector(
            model_path="damage_detect_float32.tflite",
            damaged_class_id=1,
            confidence_threshold=damage_confidence_threshold
        )


    # ======================================================
    # 영상에서 Box 검사
    # ======================================================

    def check(self, video_path):

        print("\n[Box Check]")
        print(f"Video: {video_path}")

        # --------------------------------------------------
        # 영상 열기
        # --------------------------------------------------

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            print(
                f"영상 열기 실패: {video_path}"
            )

            return {
                "detected": False,
                "bbox": None,
                "confidence": 0.0,
                "damaged": False,
                "damage_confidence": 0.0,
                "damage_class_id": None
            }

        # --------------------------------------------------
        # 영상 정보
        # --------------------------------------------------

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:

            print(
                "영상 FPS를 가져올 수 없습니다."
            )

            cap.release()

            return {
                "detected": False,
                "bbox": None,
                "confidence": 0.0,
                "damaged": False,
                "damage_confidence": 0.0,
                "damage_class_id": None
            }

        duration = frame_count / fps

        # --------------------------------------------------
        # 검사 시간 설정
        # --------------------------------------------------

        # 영상이 3초보다 길면 마지막 3초
        # 영상이 3초보다 짧으면 영상 전체

        check_duration = min(
            self.check_seconds,
            duration
        )

        start_time = max(
            0,
            duration - check_duration
        )

        print(
            f"Video duration : {duration:.2f} sec"
        )

        print(
            f"Check duration : {check_duration:.2f} sec"
        )

        print(
            f"Checking from  : {start_time:.2f} sec"
        )

        print(
            f"Checking to    : {duration:.2f} sec"
        )

        # --------------------------------------------------
        # 검사 시작 위치
        # --------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            start_time * 1000
        )

        # --------------------------------------------------
        # 검출 관련 변수
        # --------------------------------------------------

        detection_ratio = 0.8

        total_frames = 0
        detected_frames = 0

        # 마지막으로 Box가 검출된 정보
        last_bbox = None
        last_confidence = 0.0

        # 마지막으로 Box가 검출된 원본 frame
        last_detected_frame = None

        try:

            # ==================================================
            # 영상 프레임 검사
            # ==================================================

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                total_frames += 1

                # ==================================================
                # YOLO 추론
                # ==================================================

                results = self.model(frame)

                box_detected = False

                # ==================================================
                # Detection 결과 확인
                # ==================================================

                for detection in results:

                    x1 = int(
                        detection[0]
                    )

                    y1 = int(
                        detection[1]
                    )

                    x2 = int(
                        detection[2]
                    )

                    y2 = int(
                        detection[3]
                    )

                    confidence = float(
                        detection[4]
                    )

                    class_id = int(
                        detection[5]
                    )

                    # --------------------------------------------------
                    # Box class 확인
                    # --------------------------------------------------

                    if class_id != self.box_class_id:
                        continue

                    # --------------------------------------------------
                    # Confidence 확인
                    # --------------------------------------------------

                    if confidence < self.confidence_threshold:
                        continue

                    # --------------------------------------------------
                    # 현재 프레임에서 Box 검출
                    # --------------------------------------------------

                    box_detected = True

                    last_bbox = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    last_confidence = confidence

                    # ----------------------------------------------
                    # 마지막 검출 프레임 저장
                    # ----------------------------------------------

                    last_detected_frame = frame.copy()

                    break

                # --------------------------------------------------
                # 검출 프레임 수 증가
                # --------------------------------------------------

                if box_detected:

                    detected_frames += 1

            # ==================================================
            # 검사 결과 계산
            # ==================================================

            if total_frames == 0:

                print(
                    "검사할 프레임이 없습니다."
                )

                return {
                    "detected": False,
                    "bbox": None,
                    "confidence": 0.0,
                    "damaged": False,
                    "damage_confidence": 0.0,
                    "damage_class_id": None
                }

            detection_rate = (
                detected_frames / total_frames
            )

            print(
                f"Total frames    : {total_frames}"
            )

            print(
                f"Detected frames : {detected_frames}"
            )

            print(
                f"Detection rate  : "
                f"{detection_rate * 100:.1f}%"
            )

            # ==================================================
            # Box가 80% 이상 검출된 경우
            # ==================================================

            if detection_rate >= detection_ratio:

                print(
                    "Box detected!"
                )

                print(
                    f"BBox       : {last_bbox}"
                )

                print(
                    f"Confidence : "
                    f"{last_confidence:.2f}"
                )

                # --------------------------------------------------
                # 마지막 검출 프레임 또는 BBox가 없는 경우
                # --------------------------------------------------

                if (
                    last_detected_frame is None
                    or last_bbox is None
                ):

                    print(
                        "Damage detection skipped."
                    )

                    return {
                        "detected": True,
                        "bbox": last_bbox,
                        "confidence": last_confidence,
                        "damaged": False,
                        "damage_confidence": 0.0,
                        "damage_class_id": None
                    }

                # ==================================================
                # BBox 영역 Crop
                # ==================================================

                frame_height, frame_width = \
                    last_detected_frame.shape[:2]

                x1, y1, x2, y2 = last_bbox

                # --------------------------------------------------
                # BBox 좌표 안전성 검사
                # --------------------------------------------------

                x1 = max(
                    0,
                    min(x1, frame_width)
                )

                y1 = max(
                    0,
                    min(y1, frame_height)
                )

                x2 = max(
                    0,
                    min(x2, frame_width)
                )

                y2 = max(
                    0,
                    min(y2, frame_height)
                )

                # --------------------------------------------------
                # 잘못된 BBox 확인
                # --------------------------------------------------

                if x2 <= x1 or y2 <= y1:

                    print(
                        "잘못된 BBox입니다."
                    )

                    return {
                        "detected": True,
                        "bbox": last_bbox,
                        "confidence": last_confidence,
                        "damaged": False,
                        "damage_confidence": 0.0,
                        "damage_class_id": None
                    }

                # --------------------------------------------------
                # Box 영역 Crop
                # --------------------------------------------------

                cropped_box = last_detected_frame[
                    y1:y2,
                    x1:x2
                ]

                print(
                    f"Crop size : "
                    f"{cropped_box.shape[1]}x"
                    f"{cropped_box.shape[0]}"
                )

                # ==================================================
                # Damage Model 추론
                # ==================================================

                print(
                    "\n[Damage Check]"
                )

                damage_result = \
                    self.damage_model(
                        cropped_box
                    )

                print(
                    f"Damaged    : "
                    f"{damage_result['damaged']}"
                )

                print(
                    f"Confidence : "
                    f"{damage_result['confidence']:.2f}"
                )

                print(
                    f"Class ID   : "
                    f"{damage_result['class_id']}"
                )

                # ==================================================
                # 최종 결과
                # ==================================================

                return {
                    "detected": True,

                    "bbox": last_bbox,

                    "confidence": last_confidence,

                    "damaged": damage_result[
                        "damaged"
                    ],

                    "damage_confidence": damage_result[
                        "confidence"
                    ],

                    "damage_class_id": damage_result[
                        "class_id"
                    ]
                }

            # ==================================================
            # Box 없음
            # ==================================================

            print(
                "Box not detected."
            )

            return {
                "detected": False,
                "bbox": None,
                "confidence": 0.0,
                "damaged": False,
                "damage_confidence": 0.0,
                "damage_class_id": None
            }

        finally:

            cap.release()


    # ======================================================
    # 영상 삭제
    # ======================================================

    def delete_video(
        self,
        video_path
    ):

        if os.path.exists(
            video_path
        ):

            os.remove(
                video_path
            )

            print(
                f"Video deleted: {video_path}"
            )

            return True

        print(
            f"Video not found: {video_path}"
        )

        return False


    # ======================================================
    # Box 검사 + 삭제
    # ======================================================

    def check_and_delete(
        self,
        video_path
    ):

        result = self.check(
            video_path
        )

        # ==================================================
        # Box 있음
        # ==================================================

        if result["detected"]:

            print(
                "Box exists."
            )

            print(
                "Video will be kept."
            )

            # --------------------------------------------------
            # 파손 여부 출력
            # --------------------------------------------------

            if result["damaged"]:

                print(
                    "Damage detected."
                )

            else:

                print(
                    "No damage detected."
                )

            return result

        # ==================================================
        # Box 없음
        # ==================================================

        else:

            print(
                "Box does not exist."
            )

            print(
                "Deleting video..."
            )

            self.delete_video(
                video_path
            )

            return result

