# video_box_detect_module.py

import cv2
import os
import time

from box_detect_onboard_module import BoxDetector
from damage_detect_module import DamageDetector


class BoxVideoChecker:

    def __init__(
        self,
        box_class_id=0,
        confidence_threshold=0.4,
        check_seconds=3,
        detect_frame_count=5,
        frame_interval=3,
        damage_confidence_threshold=0.4
    ):

        self.box_class_id = box_class_id
        self.confidence_threshold = confidence_threshold

        self.check_seconds = check_seconds
        self.detect_frame_count = detect_frame_count
        self.frame_interval = frame_interval

        # ------------------------------------------
        # Box Detector
        # ------------------------------------------

        self.model = BoxDetector(
            confidence_threshold=confidence_threshold,
            class_id=box_class_id
        )

        # ------------------------------------------
        # Damage Detector
        # ------------------------------------------

        self.damage_model = DamageDetector(
            model_path="damage_detect_float32.tflite",
            damaged_class_id=0,
            confidence_threshold=damage_confidence_threshold
        )

        self.damage_inference_times = []
        self.box_to_damage_times = []


    # ======================================================
    # 영상 검사
    # ======================================================

    def check(self, video_path):

        print("\n[Box Check]")
        print(f"Video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            print(
                f"영상 열기 실패: {video_path}"
            )

            return {
                "detected": False,
                "boxes": []
            }


        # ==================================================
        # 영상 정보
        # ==================================================

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:

            cap.release()

            return {
                "detected": False,
                "boxes": []
            }


        duration = frame_count / fps


        # ==================================================
        # 마지막 3초 또는 전체 영상
        # ==================================================

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


        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            start_time * 1000
        )


        # ==================================================
        # 검사 구간 프레임 저장
        # ==================================================

        frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(
                frame
            )

        cap.release()


        if len(frames) == 0:

            return {
                "detected": False,
                "boxes": []
            }


        print(
            f"Frames in check range : {len(frames)}"
        )


        # ==================================================
        # 검출된 Box들을 저장
        # ==================================================

        detected_boxes = []


        # ==================================================
        # 마지막 프레임부터 검사
        # ==================================================

        frame_index = len(frames) - 1

        detected_frames = 0


        while frame_index >= 0:

            frame = frames[frame_index]

            print(
                f"[YOLO] Checking frame "
                f"{frame_index + 1}/"
                f"{len(frames)}"
            )


            results = self.model(frame)


            # ==================================================
            # 이 프레임에서 검출된 Box 전부 처리
            # ==================================================

            frame_boxes = []


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


                if class_id != self.box_class_id:
                    continue


                if confidence < self.confidence_threshold:
                    continue


                frame_boxes.append(
                    {
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "confidence": confidence,
                        "class_id": class_id
                    }
                )


            # ==================================================
            # Box가 검출된 경우
            # ==================================================

            if len(frame_boxes) > 0:

                detected_frames += 1


                print(
                    f"[YOLO] "
                    f"{len(frame_boxes)}개의 Box 검출"
                )


                # ------------------------------------------
                # 현재 프레임의 Box 정보 저장
                # ------------------------------------------

                detected_boxes = frame_boxes

                last_detected_frame = frame.copy()


                # ------------------------------------------
                # 충분한 프레임에서 검출되었으면 종료
                # ------------------------------------------

                if (
                    detected_frames
                    >= self.detect_frame_count
                ):

                    break


            else:

                print(
                    "[YOLO] Box not detected."
                )


            frame_index -= self.frame_interval


        # ==================================================
        # Box가 하나도 없는 경우
        # ==================================================

        if len(detected_boxes) == 0:

            print(
                "Box not detected."
            )

            return {
                "detected": False,
                "boxes": []
            }


        # ==================================================
        # 여러 Box 각각 Damage 검사
        # ==================================================

        final_boxes = []


        for box_index, box_info in enumerate(
            detected_boxes
        ):

            bbox = box_info["bbox"]

            confidence = box_info["confidence"]

            x1, y1, x2, y2 = bbox


            print(
                "\n"
                + "=" * 50
            )

            print(
                f"[Box {box_index + 1}]"
            )

            print(
                f"BBox       : {bbox}"
            )

            print(
                f"Confidence : {confidence:.2f}"
            )


            # ==================================================
            # BBox 안전성 검사
            # ==================================================

            frame_height, frame_width = \
                last_detected_frame.shape[:2]


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


            if x2 <= x1 or y2 <= y1:

                print(
                    "잘못된 BBox"
                )

                continue


            # ==================================================
            # Box Crop
            # ==================================================

            cropped_box = last_detected_frame[
                y1:y2,
                x1:x2
            ]


            # ==================================================
            # Damage 추론
            # ==================================================

            damage_start = time.perf_counter()


            damage_result = self.damage_model(
                cropped_box
            )


            damage_end = time.perf_counter()


            damage_time_ms = (
                damage_end
                -
                damage_start
            ) * 1000


            self.damage_inference_times.append(
                damage_time_ms
            )


            # ==================================================
            # 결과 저장
            # ==================================================

            final_boxes.append(
                {
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "confidence":
                        confidence,

                    "damaged":
                        damage_result["damaged"],

                    "damage_confidence":
                        damage_result["confidence"],

                    "damage_class_id":
                        damage_result["class_id"]
                }
            )


            print(
                f"Damage      : "
                f"{damage_result['damaged']}"
            )

            print(
                f"Damage conf : "
                f"{damage_result['confidence']:.2f}"
            )

            print(
                f"Damage time : "
                f"{damage_time_ms:.2f} ms"
            )


        # ==================================================
        # 최종 결과
        # ==================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"최종 Box 개수 : "
            f"{len(final_boxes)}"
        )

        print(
            "=" * 60
        )


        return {
            "detected": len(final_boxes) > 0,
            "boxes": final_boxes
        }


    # ======================================================
    # 영상 삭제
    # ======================================================

    def delete_video(self, video_path):

        if os.path.exists(video_path):

            os.remove(video_path)

            print(
                f"Video deleted: {video_path}"
            )

            return True

        return False


    # ======================================================
    # 검사 + 삭제
    # ======================================================

    def check_and_delete(self, video_path):

        result = self.check(
            video_path
        )


        if result["detected"]:

            print(
                f"Box {len(result['boxes'])}개 발견"
            )

            print(
                "Video will be kept."
            )

            return result


        print(
            "Box does not exist."
        )

        self.delete_video(
            video_path
        )

        return result