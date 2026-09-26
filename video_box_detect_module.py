# video_box_detect_module.py

import cv2
import os

from box_detect_onboard_module import BoxDetector


class BoxVideoChecker:

    def __init__(
        self,
        box_class_id=0,
        confidence_threshold=0.4,
        check_seconds=3
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

        print(f"영상 열기 실패: {video_path}")

        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0
        }

    # --------------------------------------------------
    # 영상 정보
    # --------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:

        print("영상 FPS를 가져올 수 없습니다.")

        cap.release()

        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0
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

    last_bbox = None
    last_confidence = 0.0

    try:

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

                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = int(detection[2])
                y2 = int(detection[3])

                confidence = float(
                    detection[4]
                )

                class_id = int(
                    detection[5]
                )

                # Box class 확인
                if class_id != self.box_class_id:
                    continue

                # Confidence 확인
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

            print("검사할 프레임이 없습니다.")

            return {
                "detected": False,
                "bbox": None,
                "confidence": 0.0
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
        # 80% 이상 검출
        # ==================================================

        if detection_rate >= detection_ratio:

            print("Box detected!")

            print(
                f"BBox       : {last_bbox}"
            )

            print(
                f"Confidence : "
                f"{last_confidence:.2f}"
            )

            return {
                "detected": True,
                "bbox": last_bbox,
                "confidence": last_confidence
            }

        # ==================================================
        # Box 없음
        # ==================================================

        print("Box not detected.")

        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0
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