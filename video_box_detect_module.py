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

        self.box_class_id = box_class_id  # 클래스 아이디 저장
        self.confidence_threshold = confidence_threshold # 컨피던스TH 저장
        self.check_seconds = check_seconds  # 영상의 마지막 몇초를 탐지할지

        self.model = BoxDetector(
            confidence_threshold=self.confidence_threshold,
            class_id=self.box_class_id
        )

    # ======================================================
    # Video Box Detection
    # ======================================================

    def check(self, video_path):

        print(
            "\n[Box Check]"
        )

        print(
            f"Video: {video_path}"
        )

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                f"영상 열기 실패: {video_path}"
            )

            return False

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

            return False

        duration = frame_count / fps

        start_time = max(
            0,
            duration - self.check_seconds
        )

        print(
            f"Video duration : "
            f"{duration:.2f} sec"
        )

        print(
            f"Checking from  : "
            f"{start_time:.2f} sec"
        )

        print(
            f"Checking to    : "
            f"{duration:.2f} sec"
        )

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            start_time * 1000
        )

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # ------------------------------------------
                # YOLO 추론
                # ------------------------------------------

                results = self.model(
                    frame
                )

                for detection in results:

                    confidence = float(
                        detection[4]
                    )

                    class_id = int(
                        detection[5]
                    )

                    if (
                        class_id == self.box_class_id
                        and
                        confidence >=
                        self.confidence_threshold
                    ):

                        print(
                            "Box detected!"
                        )

                        print(
                            f"confidence : "
                            f"{confidence:.2f}"
                        )

                        return True

            print(
                "Box not detected."
            )

            return False

        finally:

            cap.release()

    # ======================================================
    # Video Delete
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
                f"Video deleted: "
                f"{video_path}"
            )

            return True

        print(
            f"Video not found: "
            f"{video_path}"
        )

        return False

    # ======================================================
    # Check + Delete
    # ======================================================

    def check_and_delete(
        self,
        video_path
    ):

        box_detected = self.check(
            video_path
        )

        if box_detected:

            print(
                "Box exists."
            )

            print(
                "Video will be kept."
            )

            return True

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

            return False