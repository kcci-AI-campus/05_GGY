# save_plus_boxdetect.py

import cv2
import time
import os
import re
import psutil

from threading import Thread, Lock
from queue import Queue, Empty

from motion_save_moduel import MotionRecorder
from video_box_detect_module import BoxVideoChecker


# ==========================================================
# Resource Monitor
# ==========================================================

class ResourceMonitor:

    def __init__(
        self,
        recorder,
        get_inference_state,
        sample_interval=0.5
    ):

        self.recorder = recorder

        self.get_inference_state = \
            get_inference_state

        self.sample_interval = \
            sample_interval

        self.running = False

        self.thread = None

        self.stats = {

            "IDLE": {
                "cpu": [],
                "ram": [],
                "gpu": []
            },

            "RECORDING": {
                "cpu": [],
                "ram": [],
                "gpu": []
            },

            "INFERENCE": {
                "cpu": [],
                "ram": [],
                "gpu": []
            }
        }

        self.lock = Lock()

        self.gpu_source_printed = False


    # ======================================================
    # GPU 사용률
    # ======================================================

    def get_gpu_usage(self):

        paths = [

            "/sys/class/devfreq/1f0000.gpu/load",

            "/sys/class/devfreq/1f0000.gpu/utilization"
        ]


        for path in paths:

            try:

                if not os.path.exists(path):

                    continue


                with open(
                    path,
                    "r"
                ) as f:

                    value = f.read().strip()


                match = re.search(
                    r"(\d+(?:\.\d+)?)",
                    value
                )


                if match:

                    gpu = float(
                        match.group(1)
                    )


                    gpu = max(
                        0.0,
                        min(
                            gpu,
                            100.0
                        )
                    )


                    if not self.gpu_source_printed:

                        print(
                            f"[Resource Monitor] "
                            f"GPU source: {path}"
                        )

                        self.gpu_source_printed = True


                    return gpu


            except Exception:

                pass


        return None


    # ======================================================
    # GPU counter
    # ======================================================

    def get_gpu_stats_counter(self):

        paths = [

            "/sys/devices/platform/axi/1002000000.v3d/gpu_stats",

            "/sys/devices/platform/v3dbus/fec00000.v3d/gpu_stats"
        ]


        for path in paths:

            try:

                if not os.path.exists(path):

                    continue


                with open(
                    path,
                    "r"
                ) as f:

                    text = f.read()


                values = re.findall(
                    r"drm-engine-[^:]+:\s*([0-9]+)",
                    text
                )


                if values:

                    total = sum(
                        int(v)
                        for v in values
                    )


                    if not self.gpu_source_printed:

                        print(
                            f"[Resource Monitor] "
                            f"GPU source: {path}"
                        )

                        self.gpu_source_printed = True


                    return total


            except Exception:

                pass


        return None


    # ======================================================
    # GPU 사용률 계산
    # ======================================================

    def calculate_gpu_usage(
        self,
        previous_counter,
        current_counter,
        elapsed_time
    ):

        if (
            previous_counter is None
            or
            current_counter is None
            or
            elapsed_time <= 0
        ):

            return None


        delta = (
            current_counter
            -
            previous_counter
        )


        if delta < 0:

            return None


        elapsed_ns = (
            elapsed_time
            *
            1_000_000_000
        )


        gpu_usage = (
            delta
            /
            elapsed_ns
        ) * 100.0


        return max(
            0.0,
            min(
                gpu_usage,
                100.0
            )
        )


    # ======================================================
    # 현재 phase
    # ======================================================

    def get_phase(self):

        if self.get_inference_state():

            return "INFERENCE"


        if (
            self.recorder.recording
            or
            self.recorder.saving
        ):

            return "RECORDING"


        return "IDLE"


    # ======================================================
    # Monitor Worker
    # ======================================================

    def _monitor_worker(self):

        psutil.cpu_percent(
            interval=None
        )


        previous_gpu_counter = \
            self.get_gpu_stats_counter()


        previous_time = time.time()


        while self.running:

            time.sleep(
                self.sample_interval
            )


            current_time = time.time()


            elapsed = (
                current_time
                -
                previous_time
            )


            previous_time = \
                current_time


            # ==================================================
            # CPU
            # ==================================================

            cpu_usage = psutil.cpu_percent(
                interval=None
            )


            # ==================================================
            # RAM
            # ==================================================

            ram_usage = \
                psutil.virtual_memory().percent


            # ==================================================
            # GPU
            # ==================================================

            current_gpu_counter = \
                self.get_gpu_stats_counter()


            gpu_usage = \
                self.calculate_gpu_usage(
                    previous_gpu_counter,
                    current_gpu_counter,
                    elapsed
                )


            previous_gpu_counter = \
                current_gpu_counter


            # ==================================================
            # Phase
            # ==================================================

            phase = self.get_phase()


            with self.lock:

                self.stats[
                    phase
                ][
                    "cpu"
                ].append(
                    cpu_usage
                )


                self.stats[
                    phase
                ][
                    "ram"
                ].append(
                    ram_usage
                )


                if gpu_usage is not None:

                    self.stats[
                        phase
                    ][
                        "gpu"
                    ].append(
                        gpu_usage
                    )


    # ======================================================
    # Start
    # ======================================================

    def start(self):

        self.running = True

        self.thread = Thread(
            target=self._monitor_worker,
            daemon=True
        )

        self.thread.start()


    # ======================================================
    # Stop
    # ======================================================

    def stop(self):

        self.running = False


        if self.thread is not None:

            self.thread.join(
                timeout=2
            )


    # ======================================================
    # 평균값
    # ======================================================

    def average(
        self,
        values
    ):

        if not values:

            return None


        return (
            sum(values)
            /
            len(values)
        )


    # ======================================================
    # Summary
    # ======================================================

    def print_summary(self):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "        Resource Usage Summary"
        )

        print(
            "=" * 70
        )


        for phase in [
            "IDLE",
            "RECORDING",
            "INFERENCE"
        ]:

            cpu_avg = self.average(
                self.stats[
                    phase
                ][
                    "cpu"
                ]
            )


            ram_avg = self.average(
                self.stats[
                    phase
                ][
                    "ram"
                ]
            )


            gpu_avg = self.average(
                self.stats[
                    phase
                ][
                    "gpu"
                ]
            )


            print(
                f"\n[{phase}]"
            )


            if cpu_avg is not None:

                print(
                    f"  CPU 평균 : "
                    f"{cpu_avg:.1f}%"
                )

            else:

                print(
                    "  CPU 평균 : N/A"
                )


            if ram_avg is not None:

                print(
                    f"  RAM 평균 : "
                    f"{ram_avg:.1f}%"
                )

            else:

                print(
                    "  RAM 평균 : N/A"
                )


            if gpu_avg is not None:

                print(
                    f"  GPU 평균 : "
                    f"{gpu_avg:.1f}%"
                )

            else:

                print(
                    "  GPU 평균 : N/A"
                )


        print(
            "\n"
            + "=" * 70
        )

        print(
            "Resource monitoring finished."
        )

        print(
            "=" * 70
        )


# ==========================================================
# Camera
# ==========================================================

cap = cv2.VideoCapture(
    0
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        'M',
        'J',
        'P',
        'G'
    )
)


cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ==========================================================
# Motion Recorder
# ==========================================================

recorder = MotionRecorder(

    cap=cap,

    camera_width=640,
    camera_height=480,

    fps=15,

    buffer_seconds=20,

    save_dir="/home/willtek/work/video/",

    min_motion_area=2000
)


# ==========================================================
# Queue
# ==========================================================

box_check_queue = Queue()

box_result_queue = Queue()


# ==========================================================
# 현재 화면에 표시할 Box
# ==========================================================

current_boxes = []


# ==========================================================
# 추론 상태
# ==========================================================

inference_active = False


# ==========================================================
# Box Check Worker
# ==========================================================

def box_check_worker():

    global inference_active


    print(
        "[Box Thread] 시작"
    )


    # ======================================================
    # Box Checker
    # ======================================================

    box_checker = BoxVideoChecker(

        box_class_id=0,

        confidence_threshold=0.4,

        check_seconds=3,

        detect_frame_count=5,

        frame_interval=3,

        damage_confidence_threshold=0.5
    )


    while True:

        video_path = \
            box_check_queue.get()


        # ==================================================
        # 종료
        # ==================================================

        if video_path is None:

            print(
                "[Box Thread] 종료"
            )

            box_check_queue.task_done()

            break


        print(
            "\n"
            + "=" * 60
        )

        print(
            "[Box Thread]"
        )

        print(
            "새로운 영상 검사 시작"
        )

        print(
            f"Video : {video_path}"
        )

        print(
            "=" * 60
        )


        # ==================================================
        # 추론 시작
        # ==================================================

        inference_active = True


        inference_start_time = \
            time.time()


        try:

            result = \
                box_checker.check_and_delete(
                    video_path
                )


        except Exception as e:

            print(
                "\n[Box Thread ERROR]"
            )

            print(
                f"{type(e).__name__}: {e}"
            )


            result = {

                "detected": False,

                "boxes": []
            }


        finally:

            inference_end_time = \
                time.time()


            inference_elapsed = (

                inference_end_time
                -
                inference_start_time

            )


            inference_active = False


            print(
                "\n[Inference Timing]"
            )

            print(
                f"Total inference time : "
                f"{inference_elapsed:.3f} sec"
            )


        # ==================================================
        # 최종 결과
        # ==================================================

        if result.get(
            "detected",
            False
        ):

            boxes = result.get(
                "boxes",
                []
            )


            print(
                "\n"
                + "=" * 50
            )

            print(
                f"Final result : "
                f"{len(boxes)} BOX DETECTED"
            )


            # ==================================================
            # 각각의 Box 결과 출력
            # ==================================================

            for box in boxes:

                box_id = box.get(
                    "box_id",
                    0
                )


                print(
                    f"\n[BOX {box_id}]"
                )


                print(
                    f"  BBox : "
                    f"{box.get('bbox')}"
                )


                print(
                    f"  Confidence : "
                    f"{box.get('confidence', 0.0):.3f}"
                )


                print(
                    f"  Damaged : "
                    f"{box.get('damaged', False)}"
                )


                print(
                    f"  Damage confidence : "
                    f"{box.get('damage_confidence', 0.0):.3f}"
                )


                print(
                    f"  Damage class ID : "
                    f"{box.get('damage_class_id', -1)}"
                )


            print(
                "\n"
                + "=" * 50
            )


        else:

            print(
                "\n"
                + "=" * 50
            )

            print(
                "Final result : NO BOX"
            )

            print(
                "=" * 50
            )


        # ==================================================
        # Main Thread로 결과 전달
        # ==================================================

        box_result_queue.put(
            result
        )


        box_check_queue.task_done()


    # ======================================================
    # Damage Timing Summary
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Damage Detection Timing Summary"
    )

    print(
        "=" * 70
    )


    damage_timing = \
        box_checker.get_damage_timing_summary()


    if damage_timing["count"] > 0:

        print(
            f"\n측정 횟수 : "
            f"{damage_timing['count']}"
        )


        print(
            "\n[Damage Model 자체 추론시간]"
        )


        print(
            f"  평균 : "
            f"{damage_timing['average_ms']:.2f} ms"
        )


        print(
            f"  최소 : "
            f"{damage_timing['min_ms']:.2f} ms"
        )


        print(
            f"  최대 : "
            f"{damage_timing['max_ms']:.2f} ms"
        )


        print(
            "\n[Box → Damage 전체 처리시간]"
        )


        print(
            f"  평균 : "
            f"{damage_timing['box_to_damage_average_ms']:.2f} ms"
        )


        print(
            f"  최소 : "
            f"{damage_timing['box_to_damage_min_ms']:.2f} ms"
        )


        print(
            f"  최대 : "
            f"{damage_timing['box_to_damage_max_ms']:.2f} ms"
        )


    else:

        print(
            "\n파손 모델 추론 데이터가 없습니다."
        )


    print(
        "\n"
        + "=" * 70
    )


# ==========================================================
# Box Thread 시작
# ==========================================================

box_thread = Thread(

    target=box_check_worker,

    daemon=True
)


box_thread.start()


# ==========================================================
# Resource Monitor
# ==========================================================

resource_monitor = ResourceMonitor(

    recorder=recorder,

    get_inference_state=lambda:
        inference_active,

    sample_interval=0.5
)


resource_monitor.start()


# ==========================================================
# OpenCV Window
# ==========================================================

cv2.namedWindow(
    "cam",
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    "cam",
    360,
    300
)


# ==========================================================
# 시작
# ==========================================================

print(
    "=" * 60
)

print(
    "Camera started"
)

print(
    "Waiting for motion..."
)

print(
    "Press 's' to stop"
)

print(
    "=" * 60
)


# ==========================================================
# Main Loop
# ==========================================================

try:

    fps_start_time = \
        time.time()


    fps_frame_count = 0

    display_fps = 0.0


    while True:

        # ==================================================
        # FPS
        # ==================================================

        fps_frame_count += 1


        current_time = \
            time.time()


        elapsed = (

            current_time
            -
            fps_start_time

        )


        if elapsed >= 1.0:

            display_fps = (

                fps_frame_count
                /
                elapsed

            )


            fps_frame_count = 0

            fps_start_time = \
                current_time


        # ==================================================
        # Camera Frame
        # ==================================================

        ret, frame = \
            recorder.read()


        if not ret:

            print(
                "카메라 프레임을 읽을 수 없습니다."
            )

            break


        # ==================================================
        # Motion Detection / Recording
        # ==================================================

        video_path = \
            recorder.process(
                frame
            )


        # ==================================================
        # Box 검사 결과
        # ==================================================

        try:

            while True:

                result = \
                    box_result_queue.get_nowait()


                # ==================================================
                # Box 검출
                # ==================================================

                if result.get(
                    "detected",
                    False
                ):

                    current_boxes = \
                        result.get(
                            "boxes",
                            []
                        )


                    print(
                        "\n[Display] "
                        f"{len(current_boxes)}개 Box 결과 수신"
                    )


                # ==================================================
                # Box 미검출
                # ==================================================

                else:

                    current_boxes = []


                box_result_queue.task_done()


        except Empty:

            pass


        # ==================================================
        # 화면에 Box 표시
        # ==================================================

        for box in current_boxes:

            # --------------------------------------------------
            # Box ID
            # --------------------------------------------------

            box_id = box.get(
                "box_id",
                0
            )


            # --------------------------------------------------
            # BBox
            # --------------------------------------------------

            bbox = box.get(
                "bbox"
            )


            if bbox is None:

                continue


            x1, y1, x2, y2 = bbox


            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            confidence = float(
                box.get(
                    "confidence",
                    0.0
                )
            )


            # --------------------------------------------------
            # Damage
            # --------------------------------------------------

            damaged = bool(
                box.get(
                    "damaged",
                    False
                )
            )


            damage_confidence = float(
                box.get(
                    "damage_confidence",
                    0.0
                )
            )


            # ==================================================
            # 색상
            # ==================================================

            if damaged:

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )


            # ==================================================
            # Bounding Box
            # ==================================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                box_color,

                2
            )


            # ==================================================
            # Box Label
            # ==================================================

            box_label = (

                f"BOX {box_id} "

                f"{confidence * 100:.1f}%"
            )


            label_y = max(
                y1 - 8,
                15
            )


            cv2.putText(

                frame,

                box_label,

                (
                    x1,
                    label_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                box_color,

                2
            )


            # ==================================================
            # Damage Label
            # ==================================================

            if damaged:

                damage_label = (

                    f"DAMAGE "

                    f"{damage_confidence * 100:.1f}%"
                )

            else:

                damage_label = (

                    f"NORMAL "

                    f"{damage_confidence * 100:.1f}%"
                )


            # --------------------------------------------------
            # 화면 하단을 넘지 않도록 처리
            # --------------------------------------------------

            frame_height = \
                frame.shape[0]


            damage_y = min(
                y2 + 20,
                frame_height - 10
            )


            cv2.putText(

                frame,

                damage_label,

                (
                    x1,
                    damage_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                box_color,

                2
            )


        # ==================================================
        # Recorder 상태
        # ==================================================

        frame = recorder.draw_status(
            frame
        )


        # ==================================================
        # FPS
        # ==================================================

        cv2.putText(

            frame,

            f"FPS: {display_fps:.1f}",

            (10, 470),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )


        # ==================================================
        # 화면 출력
        # ==================================================

        cv2.imshow(
            "cam",
            frame
        )


        # ==================================================
        # 영상 저장 완료
        # ==================================================

        if video_path is not None:

            print(
                "\n[Main Thread]"
            )

            print(
                "영상 저장 완료"
            )

            print(
                f"Video : {video_path}"
            )

            print(
                "Box 검사 Thread로 전달"
            )


            box_check_queue.put(
                video_path
            )


        # ==================================================
        # 키 입력
        # ==================================================

        key = cv2.waitKey(
            1
        ) & 0xFF


        if key == ord("s"):

            print(
                "\n'S' pressed."
            )

            break


# ==========================================================
# 종료
# ==========================================================

finally:

    print(
        "\n프로그램 종료 중..."
    )


    # ======================================================
    # Recorder
    # ======================================================

    recorder.release()


    # ======================================================
    # Box Thread 종료 요청
    # ======================================================

    box_check_queue.put(
        None
    )


    # ======================================================
    # Box Thread 종료 대기
    # ======================================================

    box_thread.join(
        timeout=5
    )


    # ======================================================
    # Resource Monitor
    # ======================================================

    resource_monitor.stop()


    # ======================================================
    # OpenCV
    # ======================================================

    cv2.destroyAllWindows()


    # ======================================================
    # Resource Summary
    # ======================================================

    resource_monitor.print_summary()


    print(
        "Camera stopped."
    )