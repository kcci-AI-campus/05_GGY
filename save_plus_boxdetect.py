# save_plus_boxdetect

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

        # --------------------------------------------------
        # 구간별 자원 사용량 저장
        # --------------------------------------------------

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

        # --------------------------------------------------
        # GPU 상태 출력 여부
        # --------------------------------------------------

        self.gpu_source_printed = False


    # ======================================================
    # GPU 사용률 읽기
    # ======================================================

    def get_gpu_usage(self):

        # --------------------------------------------------
        # 방법 1
        # Raspberry Pi 5 / 일부 최신 커널
        #
        # devfreq load
        # --------------------------------------------------

        devfreq_paths = [

            "/sys/class/devfreq/1f0000.gpu/load",

            "/sys/class/devfreq/1f0000.gpu/utilization"
        ]

        for path in devfreq_paths:

            try:

                if os.path.exists(path):

                    with open(
                        path,
                        "r"
                    ) as f:

                        value = f.read().strip()

                    # 숫자만 있는 경우
                    match = re.search(
                        r"(\d+(?:\.\d+)?)",
                        value
                    )

                    if match:

                        gpu = float(
                            match.group(1)
                        )

                        # 0~100 범위로 제한
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


        # --------------------------------------------------
        # 방법 2
        # Raspberry Pi GPU stats
        #
        # 최신 Raspberry Pi OS / Kernel
        # --------------------------------------------------

        gpu_stats_paths = [

            "/sys/devices/platform/axi/1002000000.v3d/gpu_stats",

            "/sys/devices/platform/v3dbus/fec00000.v3d/gpu_stats"
        ]

        for path in gpu_stats_paths:

            try:

                if not os.path.exists(path):

                    continue

                with open(
                    path,
                    "r"
                ) as f:

                    text = f.read()

                # --------------------------------------------------
                # gpu_stats는 누적 active time을 제공하는 형태가
                # 있을 수 있으므로 두 번 읽어서 차이를 계산한다.
                #
                # 여기서는 ResourceMonitor에서 별도 처리한다.
                # --------------------------------------------------

                # 이 함수에서는 직접 사용하지 않고
                # 아래 get_gpu_stats_counter()에서 처리한다.

            except Exception:

                pass


        # --------------------------------------------------
        # GPU 사용률을 읽을 수 없는 경우
        # --------------------------------------------------

        return None


    # ======================================================
    # GPU 누적 사용시간 읽기
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

                # --------------------------------------------------
                # drm-engine-xxx: 숫자 형태의 값을 찾는다.
                #
                # 여러 GPU queue의 active time을 합산
                # --------------------------------------------------

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

        # --------------------------------------------------
        # 누적 counter를 사용할 수 없는 경우
        # --------------------------------------------------

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

        # --------------------------------------------------
        # gpu_stats의 active time은 ns 단위이므로
        # 실제 경과시간(ns)과 비교
        # --------------------------------------------------

        elapsed_ns = \
            elapsed_time * 1_000_000_000

        if elapsed_ns <= 0:

            return None

        gpu_usage = (
            delta
            /
            elapsed_ns
        ) * 100.0

        gpu_usage = max(
            0.0,
            min(
                gpu_usage,
                100.0
            )
        )

        return gpu_usage


    # ======================================================
    # 현재 프로그램 구간 확인
    # ======================================================

    def get_phase(self):

        # --------------------------------------------------
        # 1. 추론중
        # --------------------------------------------------

        if self.get_inference_state():

            return "INFERENCE"


        # --------------------------------------------------
        # 2. 녹화중
        # --------------------------------------------------

        if (
            self.recorder.recording
            or
            self.recorder.saving
        ):

            return "RECORDING"


        # --------------------------------------------------
        # 3. 평시
        # --------------------------------------------------

        return "IDLE"


    # ======================================================
    # Resource Monitoring Thread
    # ======================================================

    def _monitor_worker(self):

        # --------------------------------------------------
        # psutil CPU 첫 측정 초기화
        # --------------------------------------------------

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

            previous_time = current_time


            # ==================================================
            # CPU
            # ==================================================

            cpu_usage = psutil.cpu_percent(
                interval=None
            )


            # ==================================================
            # RAM
            # ==================================================

            ram_usage = psutil.virtual_memory().percent


            # ==================================================
            # GPU
            # ==================================================

            # 먼저 devfreq 방식 시도
            gpu_usage = self.get_gpu_usage()


            # --------------------------------------------------
            # devfreq가 없으면 gpu_stats 사용
            # --------------------------------------------------

            if gpu_usage is None:

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
            # 현재 구간
            # ==================================================

            phase = self.get_phase()


            # ==================================================
            # 데이터 저장
            # ==================================================

            with self.lock:

                self.stats[
                    phase
                ]["cpu"].append(
                    cpu_usage
                )

                self.stats[
                    phase
                ]["ram"].append(
                    ram_usage
                )

                if gpu_usage is not None:

                    self.stats[
                        phase
                    ]["gpu"].append(
                        gpu_usage
                    )


    # ======================================================
    # 시작
    # ======================================================

    def start(self):

        if self.running:

            return

        print(
            "\n[Resource Monitor]"
        )

        print(
            "CPU / RAM / GPU monitoring started"
        )

        print(
            f"Sampling interval : "
            f"{self.sample_interval:.1f} sec"
        )

        self.running = True

        self.thread = Thread(
            target=self._monitor_worker,
            daemon=True
        )

        self.thread.start()


    # ======================================================
    # 종료
    # ======================================================

    def stop(self):

        if not self.running:

            return

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=2
            )


    # ======================================================
    # 평균 계산
    # ======================================================

    def average(self, values):

        if not values:

            return None

        return (
            sum(values)
            /
            len(values)
        )


    # ======================================================
    # 결과 출력
    # ======================================================

    def print_summary(self):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "       Raspberry Pi Resource Usage Summary"
        )

        print(
            "=" * 70
        )


        phase_names = {

            "IDLE":
                "평시 (녹화 X / 모션 감지)",

            "RECORDING":
                "녹화중",

            "INFERENCE":
                "추론중"
        }


        for phase in [

            "IDLE",
            "RECORDING",
            "INFERENCE"

        ]:

            with self.lock:

                cpu_values = list(
                    self.stats[
                        phase
                    ]["cpu"]
                )

                ram_values = list(
                    self.stats[
                        phase
                    ]["ram"]
                )

                gpu_values = list(
                    self.stats[
                        phase
                    ]["gpu"]
                )


            cpu_avg = self.average(
                cpu_values
            )

            ram_avg = self.average(
                ram_values
            )

            gpu_avg = self.average(
                gpu_values
            )


            print(
                "\n"
                f"[{phase_names[phase]}]"
            )

            print(
                f"  Samples : "
                f"{len(cpu_values)}"
            )


            # --------------------------------------------------
            # CPU
            # --------------------------------------------------

            if cpu_avg is not None:

                print(
                    f"  CPU 평균 : "
                    f"{cpu_avg:.1f}%"
                )

            else:

                print(
                    "  CPU 평균 : N/A"
                )


            # --------------------------------------------------
            # RAM
            # --------------------------------------------------

            if ram_avg is not None:

                print(
                    f"  RAM 평균 : "
                    f"{ram_avg:.1f}%"
                )

            else:

                print(
                    "  RAM 평균 : N/A"
                )


            # --------------------------------------------------
            # GPU
            # --------------------------------------------------

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
# Camera 설정
# ==========================================================

cap = cv2.VideoCapture(0)

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
# Box Check Queue
# ==========================================================

box_check_queue = Queue()


# ==========================================================
# Box Result Queue
# ==========================================================

box_result_queue = Queue()


# ==========================================================
# 현재 화면에 표시할 Box 정보
# ==========================================================

current_bbox = None

current_confidence = 0.0

current_damaged = False

current_damage_confidence = 0.0


# ==========================================================
# 추론 상태
# ==========================================================
#
# ResourceMonitor에서 사용
#
# True
#   → 현재 BoxVideoChecker가 YOLO / Damage 추론중
#
# False
#   → 추론중이 아님
#
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


    # ------------------------------------------------------
    # BoxVideoChecker 생성
    # ------------------------------------------------------

    box_checker = BoxVideoChecker(

        box_class_id=0,

        confidence_threshold=0.4,

        check_seconds=3,

        # ----------------------------------------------
        # 마지막부터 3프레임마다 YOLO 실행
        # ----------------------------------------------

        detect_frame_count=5,

        frame_interval=3,

        damage_confidence_threshold=0.5
    )


    while True:

        # --------------------------------------------------
        # 저장 완료된 영상 기다림
        # --------------------------------------------------

        video_path = box_check_queue.get()


        # --------------------------------------------------
        # None이면 Thread 종료
        # --------------------------------------------------

        if video_path is None:

            print(
                "[Box Thread] 종료"
            )

            box_check_queue.task_done()

            break


        print(
            "\n[Box Thread]"
        )

        print(
            "새로운 영상 검사 시작"
        )

        print(
            f"Video: {video_path}"
        )


        # ==================================================
        # 추론 시작
        # ==================================================

        inference_active = True


        inference_start_time = \
            time.time()


        try:

            # ==================================================
            # 영상 검사
            # ==================================================

            result = box_checker.check_and_delete(
                video_path
            )


        finally:

            # ==================================================
            # 추론 종료
            # ==================================================

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
        # 결과 출력
        # ==================================================

        if result["detected"]:

            print(
                "\n================================"
            )

            print(
                "Final result: BOX DETECTED"
            )

            print(
                f"BBox: {result['bbox']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']:.2f}"
            )


            # --------------------------------------------------
            # Damage 결과
            # --------------------------------------------------

            print(
                f"Damaged: "
                f"{result['damaged']}"
            )

            print(
                f"Damage confidence: "
                f"{result['damage_confidence']:.2f}"
            )

            print(
                f"Damage class ID: "
                f"{result['damage_class_id']}"
            )


            # --------------------------------------------------
            # 최종 상태
            # --------------------------------------------------

            if result["damaged"]:

                print(
                    "Final status: DAMAGED BOX"
                )

            else:

                print(
                    "Final status: NORMAL BOX"
                )


            print(
                "================================"
            )


        else:

            print(
                "\n================================"
            )

            print(
                "Final result: NO BOX"
            )

            print(
                "================================"
            )


        # ==================================================
        # 결과를 Main Thread로 전달
        # ==================================================

        box_result_queue.put(
            result
        )


        # --------------------------------------------------
        # Queue 작업 완료
        # --------------------------------------------------

        box_check_queue.task_done()


# ==========================================================
# Box Check Thread 시작
# ==========================================================

box_thread = Thread(

    target=box_check_worker,

    daemon=True
)

box_thread.start()


# ==========================================================
# Resource Monitor 시작
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
# 시작 메시지
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

    # ======================================================
    # FPS 측정
    # ======================================================

    fps_start_time = time.time()

    fps_frame_count = 0

    display_fps = 0.0


    while True:

        # --------------------------------------------------
        # FPS 계산
        # --------------------------------------------------

        fps_frame_count += 1

        current_time = time.time()

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

            fps_start_time = current_time


        # ==================================================
        # 1. Camera Frame 읽기
        # ==================================================

        ret, frame = recorder.read()


        if not ret:

            print(
                "카메라 프레임을 읽을 수 없습니다."
            )

            break


        # ==================================================
        # 2. Motion Detection
        # ==================================================

        video_path = recorder.process(
            frame
        )


        # ==================================================
        # 3. Box + Damage 검사 결과 확인
        # ==================================================

        try:

            while True:

                result = \
                    box_result_queue.get_nowait()


                # ------------------------------------------
                # Box가 검출된 경우
                # ------------------------------------------

                if result["detected"]:

                    current_bbox = \
                        result["bbox"]

                    current_confidence = \
                        result["confidence"]

                    current_damaged = \
                        result["damaged"]

                    current_damage_confidence = \
                        result["damage_confidence"]


                # ------------------------------------------
                # Box가 검출되지 않은 경우
                # ------------------------------------------

                else:

                    current_bbox = None

                    current_confidence = 0.0

                    current_damaged = False

                    current_damage_confidence = 0.0


                box_result_queue.task_done()


        except Empty:

            pass


        # ==================================================
        # 4. Main 화면에 BBox 그리기
        # ==================================================

        if current_bbox is not None:

            x1, y1, x2, y2 = \
                current_bbox


            # --------------------------------------------------
            # Bounding Box
            # --------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (255, 0, 0),

                2
            )


            # --------------------------------------------------
            # Box Confidence
            # --------------------------------------------------

            box_label = (

                f"BOX "

                f"{current_confidence * 100:.2f}%"
            )


            cv2.putText(

                frame,

                box_label,

                (
                    x1,
                    max(y1 - 8, 15)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 0, 0),

                2
            )


            # ==================================================
            # Damage 결과 화면 표시
            # ==================================================

            if current_damaged:

                damage_label = (

                    f"DAMAGE "

                    f"{current_damage_confidence * 100:.2f}%"
                )

            else:

                damage_label = (

                    f"NORMAL "

                    f"{current_damage_confidence * 100:.2f}%"
                )


            cv2.putText(

                frame,

                damage_label,

                (
                    x1,
                    min(y2 + 32, 470)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 0, 255)
                if current_damaged
                else (0, 255, 0),

                2
            )


        # ==================================================
        # 5. Recorder 상태 표시
        # ==================================================

        frame = recorder.draw_status(
            frame
        )


        # ==================================================
        # 6. FPS 표시
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
        # 7. 화면 출력
        # ==================================================

        cv2.imshow(
            "cam",
            frame
        )


        # ==================================================
        # 8. 저장 완료된 영상 확인
        # ==================================================

        if video_path is not None:

            print(
                "\n[Main Thread]"
            )

            print(
                "영상 저장 완료"
            )

            print(
                f"Video: {video_path}"
            )

            print(
                "Box 검사 Thread로 전달"
            )


            # --------------------------------------------------
            # 영상 경로를 Box Thread로 전달
            # --------------------------------------------------

            box_check_queue.put(
                video_path
            )


        # ==================================================
        # 9. 키 입력
        # ==================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("s"):

            print(
                "\n'S' pressed."
            )

            break


# ==========================================================
# 프로그램 종료
# ==========================================================

finally:

    print(
        "\n프로그램 종료 중..."
    )


    # ------------------------------------------------------
    # MotionRecorder 종료
    # ------------------------------------------------------

    recorder.release()


    # ------------------------------------------------------
    # Box Thread 종료 요청
    # ------------------------------------------------------

    box_check_queue.put(
        None
    )


    # ------------------------------------------------------
    # Box Thread 종료 대기
    # ------------------------------------------------------

    box_thread.join(
        timeout=5
    )


    # ------------------------------------------------------
    # Resource Monitor 종료
    # ------------------------------------------------------

    resource_monitor.stop()


    # ------------------------------------------------------
    # OpenCV 종료
    # ------------------------------------------------------

    cv2.destroyAllWindows()


    # ======================================================
    # 최종 Resource 사용량 출력
    # ======================================================

    resource_monitor.print_summary()


    print(
        "Camera stopped."
    )

