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
        confidence_threshold=0.5,
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

        # ==================================================
        # Box Detector
        # ==================================================

        self.model = BoxDetector(
            confidence_threshold=confidence_threshold,
            class_id=box_class_id
        )

        # ==================================================
        # Damage Detector
        # ==================================================

        self.damage_model = DamageDetector(
            model_path="damage_detect_v2_float32.tflite",

            # 현재 모델 기준
            # class 0 = damaged
            damaged_class_id=0,

            confidence_threshold=damage_confidence_threshold
        )

        # ==================================================
        # 성능 측정
        # ==================================================

        self.damage_inference_times = []

        self.box_to_damage_times = []


    # ======================================================
    # Damage Timing Summary
    # ======================================================

    def get_damage_timing_summary(self):

        damage_times = self.damage_inference_times
        box_to_damage_times = self.box_to_damage_times

        if len(damage_times) == 0:

            return {
                "count": 0,

                "average_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,

                "box_to_damage_average_ms": 0.0,
                "box_to_damage_min_ms": 0.0,
                "box_to_damage_max_ms": 0.0
            }


        damage_average = (
            sum(damage_times)
            /
            len(damage_times)
        )

        damage_min = min(
            damage_times
        )

        damage_max = max(
            damage_times
        )


        if len(box_to_damage_times) > 0:

            box_average = (
                sum(box_to_damage_times)
                /
                len(box_to_damage_times)
            )

            box_min = min(
                box_to_damage_times
            )

            box_max = max(
                box_to_damage_times
            )

        else:

            box_average = 0.0
            box_min = 0.0
            box_max = 0.0


        return {

            "count":
                len(damage_times),

            "average_ms":
                damage_average,

            "min_ms":
                damage_min,

            "max_ms":
                damage_max,

            "box_to_damage_average_ms":
                box_average,

            "box_to_damage_min_ms":
                box_min,

            "box_to_damage_max_ms":
                box_max
        }


    # ======================================================
    # 영상 검사
    # ======================================================

    def check(self, video_path):

        print("\n")
        print("=" * 70)
        print("[Box Check]")
        print("=" * 70)

        print(
            f"Video : {video_path}"
        )


        # ==================================================
        # 영상 열기
        # ==================================================

        cap = cv2.VideoCapture(
            video_path
        )


        if not cap.isOpened():

            print(
                f"영상 열기 실패 : {video_path}"
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

            print(
                "FPS 정보를 가져올 수 없습니다."
            )

            return {
                "detected": False,
                "boxes": []
            }


        duration = (
            frame_count
            /
            fps
        )


        # ==================================================
        # 마지막 3초 검사
        #
        # 영상이 3초보다 짧으면 전체 영상 검사
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
            f"Video duration : "
            f"{duration:.2f} sec"
        )

        print(
            f"Check duration : "
            f"{check_duration:.2f} sec"
        )


        # ==================================================
        # 검사 시작 위치
        # ==================================================

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

            print(
                "검사할 프레임이 없습니다."
            )

            return {
                "detected": False,
                "boxes": []
            }


        print(
            f"Frames in check range : "
            f"{len(frames)}"
        )


        # ==================================================
        # Box 검출 결과
        # ==================================================

        detected_boxes = []

        last_detected_frame = None


        # ==================================================
        # 마지막 프레임부터 검사
        # ==================================================

        frame_index = (
            len(frames) - 1
        )

        detected_frames = 0


        while frame_index >= 0:

            frame = frames[
                frame_index
            ]


            print(
                f"\n[YOLO] Checking frame "
                f"{frame_index + 1}/"
                f"{len(frames)}"
            )


            # ==================================================
            # YOLO 추론
            # ==================================================

            results = self.model(
                frame
            )


            frame_boxes = []


            # ==================================================
            # 이 프레임의 모든 Box 처리
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


                frame_boxes.append(
                    {

                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "confidence":
                            confidence,

                        "class_id":
                            class_id
                    }
                )


            # ==================================================
            # Box 검출
            # ==================================================

            if len(frame_boxes) > 0:

                detected_frames += 1


                print(
                    f"[YOLO] "
                    f"{len(frame_boxes)}개의 Box 검출"
                )


                # --------------------------------------------------
                # 현재 프레임의 Box 저장
                # --------------------------------------------------

                detected_boxes = frame_boxes

                last_detected_frame = (
                    frame.copy()
                )


                # --------------------------------------------------
                # 충분한 프레임에서 검출
                # --------------------------------------------------

                if (
                    detected_frames
                    >= self.detect_frame_count
                ):

                    print(
                        f"[YOLO] "
                        f"{self.detect_frame_count}개 "
                        f"검출 프레임 확보"
                    )

                    break


            else:

                print(
                    "[YOLO] Box not detected."
                )


            # --------------------------------------------------
            # 다음 프레임
            # --------------------------------------------------

            frame_index -= (
                self.frame_interval
            )


        # ==================================================
        # Box 미검출
        # ==================================================

        if (
            len(detected_boxes) == 0
            or
            last_detected_frame is None
        ):

            print(
                "\nBox not detected."
            )

            return {
                "detected": False,
                "boxes": []
            }


        # ==================================================
        # 중요
        #
        # Box를 X 좌표 기준으로 정렬
        #
        # 왼쪽 Box → BOX 1
        # 오른쪽 Box → BOX 2
        #
        # YOLO 결과 리스트 순서가 바뀌어도
        # 표시 순서를 일정하게 유지
        # ==================================================

        detected_boxes = sorted(
            detected_boxes,
            key=lambda box:
                box["bbox"][0]
        )


        print(
            "\n"
            + "=" * 70
        )

        print(
            "Final YOLO detections"
        )

        print(
            f"Box count : "
            f"{len(detected_boxes)}"
        )

        print(
            "=" * 70
        )


        for index, box_info in enumerate(
            detected_boxes,
            start=1
        ):

            print(
                f"[BOX {index}] "
                f"BBox = "
                f"{box_info['bbox']} "
                f"Conf = "
                f"{box_info['confidence']:.3f}"
            )


        # ==================================================
        # 각각의 Box에 대해 Damage 검사
        # ==================================================

        final_boxes = []


        for box_index, box_info in enumerate(
            detected_boxes,
            start=1
        ):

            box_start_time = (
                time.perf_counter()
            )


            bbox = box_info[
                "bbox"
            ]

            confidence = box_info[
                "confidence"
            ]


            x1, y1, x2, y2 = bbox


            print(
                "\n"
                + "=" * 60
            )

            print(
                f"[BOX {box_index}]"
            )

            print(
                f"BBox       : {bbox}"
            )

            print(
                f"Confidence : "
                f"{confidence:.3f}"
            )


            # ==================================================
            # BBox 안전성 검사
            # ==================================================

            frame_height, frame_width = \
                last_detected_frame.shape[:2]


            x1 = max(
                0,
                min(
                    x1,
                    frame_width - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    frame_height - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    frame_width
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    frame_height
                )
            )


            # ==================================================
            # BBox 유효성
            # ==================================================

            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                print(
                    f"[BOX {box_index}] "
                    "잘못된 BBox"
                )

                continue


            # ==================================================
            # Box Crop
            # ==================================================

            cropped_box = \
                last_detected_frame[
                    y1:y2,
                    x1:x2
                ]


            # ==================================================
            # Crop 검증
            # ==================================================

            if cropped_box.size == 0:

                print(
                    f"[BOX {box_index}] "
                    "Crop 실패"
                )

                continue


            crop_height, crop_width = \
                cropped_box.shape[:2]


            print(
                f"[BOX {box_index}] "
                f"Crop size : "
                f"{crop_width} x "
                f"{crop_height}"
            )


            # ==================================================
            # Damage 추론
            # ==================================================

            damage_start = \
                time.perf_counter()


            try:

                damage_result = \
                    self.damage_model(
                        cropped_box
                    )

            except Exception as e:

                print(
                    f"[BOX {box_index}] "
                    "Damage Detector ERROR"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                continue


            damage_end = \
                time.perf_counter()


            damage_time_ms = (

                damage_end
                -
                damage_start

            ) * 1000


            self.damage_inference_times.append(
                damage_time_ms
            )


            # ==================================================
            # Box → Damage 전체 처리 시간
            # ==================================================

            box_end_time = \
                time.perf_counter()


            box_to_damage_ms = (

                box_end_time
                -
                box_start_time

            ) * 1000


            self.box_to_damage_times.append(
                box_to_damage_ms
            )


            # ==================================================
            # Damage 결과
            # ==================================================

            damaged = bool(
                damage_result.get(
                    "damaged",
                    False
                )
            )


            damage_confidence = float(
                damage_result.get(
                    "confidence",
                    0.0
                )
            )


            damage_class_id = int(
                damage_result.get(
                    "class_id",
                    -1
                )
            )


            # ==================================================
            # 최종 Box 결과 저장
            # ==================================================

            final_boxes.append(
                {

                    # ------------------------------------------
                    # 명시적인 Box ID
                    # ------------------------------------------

                    "box_id":
                        box_index,

                    # ------------------------------------------
                    # BBox
                    # ------------------------------------------

                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    # ------------------------------------------
                    # YOLO confidence
                    # ------------------------------------------

                    "confidence":
                        confidence,

                    # ------------------------------------------
                    # Damage 결과
                    # ------------------------------------------

                    "damaged":
                        damaged,

                    "damage_confidence":
                        damage_confidence,

                    "damage_class_id":
                        damage_class_id
                }
            )


            # ==================================================
            # 상세 결과 출력
            # ==================================================

            print(
                f"[BOX {box_index}] "
                f"Damage      : "
                f"{damaged}"
            )

            print(
                f"[BOX {box_index}] "
                f"Damage conf : "
                f"{damage_confidence:.4f}"
            )

            print(
                f"[BOX {box_index}] "
                f"Damage class: "
                f"{damage_class_id}"
            )

            print(
                f"[BOX {box_index}] "
                f"Damage time : "
                f"{damage_time_ms:.2f} ms"
            )

            print(
                f"[BOX {box_index}] "
                f"Box→Damage : "
                f"{box_to_damage_ms:.2f} ms"
            )


        # ==================================================
        # 최종 결과
        # ==================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL RESULT"
        )

        print(
            "=" * 70
        )


        print(
            f"최종 Box 개수 : "
            f"{len(final_boxes)}"
        )


        for box in final_boxes:

            print(
                f"\n"
                f"[BOX {box['box_id']}]"
            )

            print(
                f"  BBox       : "
                f"{box['bbox']}"
            )

            print(
                f"  YOLO conf  : "
                f"{box['confidence']:.3f}"
            )

            print(
                f"  Damaged    : "
                f"{box['damaged']}"
            )

            print(
                f"  Damage conf: "
                f"{box['damage_confidence']:.3f}"
            )

            print(
                f"  Class ID   : "
                f"{box['damage_class_id']}"
            )


        print(
            "\n"
            + "=" * 70
        )


        return {

            "detected":
                len(final_boxes) > 0,

            "boxes":
                final_boxes
        }


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
                f"Video deleted: "
                f"{video_path}"
            )

            return True


        return False


    # ======================================================
    # 검사 + 삭제
    # ======================================================

    def check_and_delete(
        self,
        video_path
    ):

        result = self.check(
            video_path
        )


        if result.get(
            "detected",
            False
        ):

            print(
                f"\nBox "
                f"{len(result['boxes'])}개 발견"
            )

            print(
                "Video will be kept."
            )

            return result


        print(
            "\nBox does not exist."
        )


        self.delete_video(
            video_path
        )


        return result