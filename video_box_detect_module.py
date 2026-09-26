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
        damage_confidence_threshold=0.5
    ):

        self.box_class_id = box_class_id

        self.confidence_threshold = \
            confidence_threshold

        self.check_seconds = check_seconds

        # --------------------------------------------------
        # 몇 개의 프레임에서 Box가 검출되면
        # Box가 있다고 판단할 것인지
        # --------------------------------------------------

        self.detect_frame_count = \
            detect_frame_count

        # --------------------------------------------------
        # 몇 프레임마다 YOLO를 실행할 것인지
        #
        # 1 = 모든 프레임
        # 2 = 2프레임마다
        # 3 = 3프레임마다
        # --------------------------------------------------

        self.frame_interval = \
            frame_interval

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

        # --------------------------------------------------
        # 파손 탐지 시간 측정
        #
        # damage_inference_times
        #   → 파손 모델 자체의 순수 추론시간
        #
        # box_to_damage_times
        #   → Box 검출 확정부터 파손 판정 완료까지
        # --------------------------------------------------

        self.damage_inference_times = []

        self.box_to_damage_times = []


    # ======================================================
    # 영상에서 Box 검사
    # ======================================================

    def check(self, video_path):

        print("\n[Box Check]")
        print(
            f"Video: {video_path}"
        )

        # --------------------------------------------------
        # 영상 열기
        # --------------------------------------------------

        cap = cv2.VideoCapture(
            video_path
        )

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
        #
        # 영상이 3초보다 길면 마지막 3초
        # 영상이 3초보다 짧으면 전체 영상
        # --------------------------------------------------

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

        print(
            f"Checking from  : "
            f"{start_time:.2f} sec"
        )

        print(
            f"Checking to    : "
            f"{duration:.2f} sec"
        )

        print(
            f"Frame interval : "
            f"{self.frame_interval}"
        )

        print(
            f"Required detections : "
            f"{self.detect_frame_count}"
        )

        # --------------------------------------------------
        # 검사 시작 위치로 이동
        # --------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            start_time * 1000
        )

        # --------------------------------------------------
        # 검사 구간의 프레임 저장
        #
        # 여기서는 YOLO를 실행하지 않고
        # 마지막 3초의 프레임만 메모리에 저장한다.
        #
        # 이후 마지막 프레임부터 역순으로
        # 일정 간격마다 YOLO를 실행한다.
        # --------------------------------------------------

        frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(
                frame
            )

        cap.release()

        # --------------------------------------------------
        # 검사할 프레임이 없는 경우
        # --------------------------------------------------

        if len(frames) == 0:

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

        print(
            f"Frames in check range : "
            f"{len(frames)}"
        )

        # --------------------------------------------------
        # 검출 관련 변수
        # --------------------------------------------------

        detected_frames = 0

        box_to_damage_start = None

        # 마지막으로 Box가 검출된 정보
        last_bbox = None
        last_confidence = 0.0

        # 마지막으로 Box가 검출된 원본 frame
        last_detected_frame = None

        # --------------------------------------------------
        # 마지막 프레임부터 검사
        #
        # 예:
        #
        # frame 44
        # frame 41
        # frame 38
        # frame 35
        # ...
        #
        # frame_interval = 3
        # --------------------------------------------------

        frame_index = len(frames) - 1

        while frame_index >= 0:

            frame = frames[
                frame_index
            ]

            print(
                f"[YOLO] Checking frame "
                f"{frame_index + 1}/"
                f"{len(frames)}"
            )

            # ==================================================
            # YOLO 추론
            # ==================================================

            results = self.model(
                frame
            )

            box_detected = False

            # ==================================================
            # Detection 결과 확인
            #
            # 기존 BoxDetector 반환 형식 유지
            #
            # detection[0] = x1
            # detection[1] = y1
            # detection[2] = x2
            # detection[3] = y2
            # detection[4] = confidence
            # detection[5] = class_id
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

                if (
                    confidence
                    < self.confidence_threshold
                ):

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

                last_confidence = \
                    confidence

                # --------------------------------------------------
                # 현재 검출 프레임 저장
                # --------------------------------------------------

                last_detected_frame = \
                    frame.copy()

                break

            # ==================================================
            # Box 검출
            # ==================================================

            if box_detected:

                detected_frames += 1

                print(
                    f"[YOLO] Box detected "
                    f"({detected_frames}/"
                    f"{self.detect_frame_count}) "
                    f"confidence="
                    f"{last_confidence:.2f}"
                )

                # --------------------------------------------------
                # 필요한 검출 횟수에 도달하면
                # 즉시 YOLO 검사 종료
                # --------------------------------------------------

                if (
                    detected_frames
                    >= self.detect_frame_count
                ):

                    print(
                        "[YOLO] Required "
                        "detection count reached."
                    )

                    # --------------------------------------------------
                    # Box 검출 확정
                    #
                    # 이 시점부터
                    # Box → Damage 처리 시간을 측정
                    # --------------------------------------------------

                    box_to_damage_start = \
                        time.perf_counter()

                    break

            else:

                print(
                    "[YOLO] Box not detected."
                )

            # --------------------------------------------------
            # 다음 프레임
            #
            # 마지막부터 역순으로
            # frame_interval 만큼 이동
            # --------------------------------------------------

            frame_index -= \
                self.frame_interval

        # ======================================================
        # Box 탐지 결과
        # ======================================================

        print(
            f"\n[YOLO Result]"
        )

        print(
            f"Checked frames : "
            f"up to {self.detect_frame_count}"
        )

        print(
            f"Detected frames : "
            f"{detected_frames}"
        )

        # --------------------------------------------------
        # 필요한 횟수만큼 검출되지 않은 경우
        # --------------------------------------------------

        if (
            detected_frames
            < self.detect_frame_count
        ):

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

        # ======================================================
        # Box 탐지 성공
        # ======================================================

        print(
            "Box detected!"
        )

        print(
            f"BBox       : "
            f"{last_bbox}"
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

        # ======================================================
        # BBox 영역 Crop
        # ======================================================

        frame_height, frame_width = \
            last_detected_frame.shape[:2]

        x1, y1, x2, y2 = \
            last_bbox

        # --------------------------------------------------
        # BBox 좌표 안전성 검사
        # --------------------------------------------------

        x1 = max(
            0,
            min(
                x1,
                frame_width
            )
        )

        y1 = max(
            0,
            min(
                y1,
                frame_height
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

        # --------------------------------------------------
        # 잘못된 BBox 확인
        # --------------------------------------------------

        if (
            x2 <= x1
            or y2 <= y1
        ):

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

        # ======================================================
        # Box 영역 Crop
        # ======================================================

        cropped_box = \
            last_detected_frame[
                y1:y2,
                x1:x2
            ]

        print(
            f"Crop size : "
            f"{cropped_box.shape[1]}x"
            f"{cropped_box.shape[0]}"
        )

        # ======================================================
        # Damage Model 추론
        # ======================================================
        print(
            "\n[Damage Check]"
        )

        # ==================================================
        # 파손 모델 순수 추론시간 측정 시작
        # ==================================================

        damage_inference_start = \
            time.perf_counter()


        damage_result = \
            self.damage_model(
                cropped_box
            )


        # ==================================================
        # 파손 모델 순수 추론시간 측정 종료
        # ==================================================

        damage_inference_end = \
            time.perf_counter()


        damage_inference_time = (
            damage_inference_end
            -
            damage_inference_start
        )


        damage_inference_time_ms = (
            damage_inference_time
            * 1000
        )


        # --------------------------------------------------
        # 파손 모델 추론시간 저장
        # --------------------------------------------------

        self.damage_inference_times.append(
            damage_inference_time_ms
        )


        print(
            f"Damage inference time : "
            f"{damage_inference_time_ms:.2f} ms"
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
        # Box → Damage 전체 처리시간
        # ==================================================

        if box_to_damage_start is not None:

            box_to_damage_end = \
                time.perf_counter()


            box_to_damage_time = (
                box_to_damage_end
                -
                box_to_damage_start
            )


            box_to_damage_time_ms = (
                box_to_damage_time
                * 1000
            )


            self.box_to_damage_times.append(
                box_to_damage_time_ms
            )


            print(
                f"Box → Damage time : "
                f"{box_to_damage_time_ms:.2f} ms"
            )

        # ======================================================
        # 최종 결과
        # ======================================================

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

        print(
            f"Video not found: "
            f"{video_path}"
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

    # ======================================================
    # Damage 추론 시간 통계
    # ======================================================

    def get_damage_timing_summary(self):

        # --------------------------------------------------
        # 데이터가 없는 경우
        # --------------------------------------------------

        if not self.damage_inference_times:

            return {
                "count": 0,
                "average_ms": None,
                "min_ms": None,
                "max_ms": None,
                "box_to_damage_average_ms": None,
                "box_to_damage_min_ms": None,
                "box_to_damage_max_ms": None
            }


        # --------------------------------------------------
        # Damage 모델 순수 추론
        # --------------------------------------------------

        damage_average = (
            sum(self.damage_inference_times)
            /
            len(self.damage_inference_times)
        )


        damage_min = min(
            self.damage_inference_times
        )


        damage_max = max(
            self.damage_inference_times
        )


        # --------------------------------------------------
        # Box → Damage 전체 시간
        # --------------------------------------------------

        if self.box_to_damage_times:

            box_to_damage_average = (
                sum(self.box_to_damage_times)
                /
                len(self.box_to_damage_times)
            )

            box_to_damage_min = min(
                self.box_to_damage_times
            )

            box_to_damage_max = max(
                self.box_to_damage_times
            )

        else:

            box_to_damage_average = None
            box_to_damage_min = None
            box_to_damage_max = None


        return {

            "count":
                len(self.damage_inference_times),

            "average_ms":
                damage_average,

            "min_ms":
                damage_min,

            "max_ms":
                damage_max,

            "box_to_damage_average_ms":
                box_to_damage_average,

            "box_to_damage_min_ms":
                box_to_damage_min,

            "box_to_damage_max_ms":
                box_to_damage_max
        }