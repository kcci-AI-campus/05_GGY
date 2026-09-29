import os
import re
import time
import cv2
import psutil

from threading import Thread, Lock
from queue import Queue, Empty

from motion_save_moduel import MotionRecorder
from video_box_detect_module import BoxVideoChecker

#동작하는 동안의 CPU, RAM, GPU 사용량을 측정하고 출력하는 클래스
class ResourceMonitor:
    PHASES = ("IDLE", "RECORDING", "INFERENCE")

    def __init__(self, recorder, get_inference_state, sample_interval=0.5):
        self.recorder = recorder
        self.get_inference_state = get_inference_state
        self.sample_interval = sample_interval
        self.running = False
        self.thread = None
        self.lock = Lock()
        self.gpu_source_printed = False
        self.stats = {
            phase: {"cpu": [], "ram": [], "gpu": [], "cpu_per_core": []}
            for phase in self.PHASES
        }

    def get_gpu_stats_counter(self):
        paths = (
            "/sys/devices/platform/axi/1002000000.v3d/gpu_stats",
            "/sys/devices/platform/v3dbus/fec00000.v3d/gpu_stats",
        )

        for path in paths:
            try:
                if not os.path.exists(path):
                    continue

                with open(path) as file:
                    values = re.findall(
                        r"drm-engine-[^:]+:\s*([0-9]+)",
                        file.read(),
                    )

                if values:
                    if not self.gpu_source_printed:
                        print(f"[Resource Monitor] GPU source: {path}")
                        self.gpu_source_printed = True
                    return sum(map(int, values))
            except OSError:
                pass

        return None

    @staticmethod
    def calculate_gpu_usage(previous, current, elapsed):
        if previous is None or current is None or elapsed <= 0:
            return None

        delta = current - previous
        if delta < 0:
            return None

        return max(0.0, min(delta / (elapsed * 1e9) * 100.0, 100.0))

    def get_phase(self):
        if self.get_inference_state():
            return "INFERENCE"
        if self.recorder.recording or self.recorder.saving:
            return "RECORDING"
        return "IDLE"

    def _worker(self):
        psutil.cpu_percent(None)
        psutil.cpu_percent(None, percpu=True)

        previous_gpu = self.get_gpu_stats_counter()
        previous_time = time.time()

        while self.running:
            time.sleep(self.sample_interval)

            now = time.time()
            elapsed = now - previous_time
            previous_time = now

            cpu = psutil.cpu_percent(None)
            cores = psutil.cpu_percent(None, percpu=True)
            ram = psutil.virtual_memory().percent

            current_gpu = self.get_gpu_stats_counter()
            gpu = self.calculate_gpu_usage(previous_gpu, current_gpu, elapsed)
            previous_gpu = current_gpu

            phase = self.get_phase()
            with self.lock:
                self.stats[phase]["cpu"].append(cpu)
                self.stats[phase]["ram"].append(ram)
                self.stats[phase]["cpu_per_core"].append(cores)
                if gpu is not None:
                    self.stats[phase]["gpu"].append(gpu)

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    @staticmethod
    def average(values):
        return sum(values) / len(values) if values else None

    @staticmethod
    def maximum(values):
        return max(values) if values else None

    def print_summary(self):
        print("\n" + "=" * 70)
        print("        Resource Usage Summary")
        print("=" * 70)

        for phase in self.PHASES:
            data = self.stats[phase]
            cpu_avg = self.average(data["cpu"])
            cpu_max = self.maximum(data["cpu"])
            ram_avg = self.average(data["ram"])
            gpu_avg = self.average(data["gpu"])

            print(f"\n[{phase}]")
            print(
                f"  CPU 평균 : {cpu_avg:.1f}%"
                if cpu_avg is not None else "  CPU 평균 : N/A"
            )
            print(
                f"  CPU 최대 : {cpu_max:.1f}%"
                if cpu_max is not None else "  CPU 최대 : N/A"
            )

            cores = data["cpu_per_core"]
            if cores:
                avg = [
                    self.average([sample[i] for sample in cores])
                    for i in range(len(cores[0]))
                ]
                max_ = [
                    self.maximum([sample[i] for sample in cores])
                    for i in range(len(cores[0]))
                ]
                print("  CPU Core 평균 : " + ", ".join(f"{v:.1f}%" for v in avg))
                print("  CPU Core 최대 : " + ", ".join(f"{v:.1f}%" for v in max_))
            else:
                print("  CPU Core : N/A")

            print(
                f"  RAM 평균 : {ram_avg:.1f}%"
                if ram_avg is not None else "  RAM 평균 : N/A"
            )
            print(
                f"  GPU 평균 : {gpu_avg:.1f}%"
                if gpu_avg is not None else "  GPU 평균 : N/A"
            )

        print("\n" + "=" * 70)
        print("Resource monitoring finished.")
        print("=" * 70)

#캠설정
def create_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# 디버그를 위해 입력받은 RESULT에서 값들을 가져와 터미널의 출력
def print_box_result(result):
    if not result.get("detected"):
        print("\n" + "=" * 50)
        print("Final result : NO BOX")
        print("=" * 50)
        return

    boxes = result.get("boxes", [])
    print("\n" + "=" * 50)
    print(f"Final result : {len(boxes)} BOX DETECTED")

    for box in boxes:
        print(
            f"\n[BOX {box.get('box_id', 0)}]"
            f"\n  BBox : {box.get('bbox')}"
            f"\n  Confidence : {box.get('confidence', 0):.3f}"
            f"\n  Damaged : {box.get('damaged', False)}"
            f"\n  Damage confidence : {box.get('damage_confidence', 0):.3f}"
            f"\n  Damage class ID : {box.get('damage_class_id', -1)}"
        )

    print("\n" + "=" * 50)

# 입력받은 두모델을 통과하고 반환된 boxes 값으로 박스를 프레임에 그리는 함수
def draw_boxes(frame, boxes):
    frame_height = frame.shape[0]

    for box in boxes:
        bbox = box.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        confidence = float(box.get("confidence", 0.0))
        damaged = bool(box.get("damaged", False))
        damage_confidence = float(box.get("damage_confidence", 0.0))
        box_id = box.get("box_id", 0)
        color = (0, 0, 255) if damaged else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"BOX {box_id} {confidence:.1%}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        label = f"{'DAMAGE' if damaged else 'NORMAL'} {damage_confidence:.1%}"
        cv2.putText(
            frame,
            label,
            (x1, min(y2 + 20, frame_height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

# 입력받은 check_queue에서 영상경로를 가져와 BoxVideoChecker를 통해 박스,훼손여부 판별 후 영상 삭제여부를 결정하고 영상 주소를 result_queue에 전달
def box_check_worker(check_queue, result_queue, state):
    checker = BoxVideoChecker(
        box_class_id=0,
        confidence_threshold=0.4,
        check_seconds=1,
        detect_frame_count=5,
        frame_interval=3,
        damage_threshold=0.4,
    )

    print("[Box Thread] 시작")

    while True:
        video_path = check_queue.get()

        try:
            if video_path is None:
                print("[Box Thread] 종료")
                return

            print("\n" + "=" * 60)
            print(f"[Box Thread] 검사 시작\nVideo : {video_path}")
            print("=" * 60)

            state["inference"] = True
            started = time.time()

            try:
                result = checker.check_and_delete(video_path)
                if not isinstance(result, dict):
                    result = {"detected": False, "boxes": []}
            except Exception as e:
                print(f"[Box Thread ERROR] {type(e).__name__}: {e}")
                result = {"detected": False, "boxes": []}
            finally:
                state["inference"] = False
                print(f"[Inference Timing] Total : {time.time() - started:.3f} sec")

            result["_video_path"] = video_path
            print_box_result(result)
            result_queue.put(result)

        finally:
            check_queue.task_done()

    # 여기까지 도달하지 않음 디버그용


    timing = checker.get_damage_timing_summary()
    print("\n" + "=" * 70)
    print("Damage Detection Timing Summary")
    print("=" * 70)
    if timing["count"]:
        print(f"측정 횟수 : {timing['count']}")
        print(f"Damage 평균 : {timing['average_ms']:.2f} ms")
        print(f"Damage 최소 : {timing['min_ms']:.2f} ms")
        print(f"Damage 최대 : {timing['max_ms']:.2f} ms")
        print(f"Box → Damage 평균 : {timing['box_to_damage_average_ms']:.2f} ms")
        print(f"Box → Damage 최소 : {timing['box_to_damage_min_ms']:.2f} ms")
        print(f"Box → Damage 최대 : {timing['box_to_damage_max_ms']:.2f} ms")
    else:
        print("파손 모델 추론 데이터가 없습니다.")
    print("=" * 70)

#위에 함수들을 호출해 연결함
def main():
    cap = create_camera()
    #움직임이 감지되면 녹화하고, 녹화가 완료되면 BoxVideoChecker를 통해 박스,훼손여부 판별 후 영상 삭제여부를 결정하고 결과를 화면에 출력
    recorder = MotionRecorder(
        cap=cap,
        camera_width=640,
        camera_height=480,
        fps=15,
        buffer_seconds=20,
        save_dir="/home/willtek/work/video/",
        min_motion_area=2000,
    )

    check_queue = Queue() # 추론전 영상 주소
    result_queue = Queue() # 추론후 영상 주소,박스,훼손여부 판별결과
    state = {"inference": False}

    current_boxes = []
    latest_video_path = None
    previous_recording = False

    # 추론모델 쓰레드를 통해 동작
    box_thread = Thread(
        target=box_check_worker,
        args=(check_queue, result_queue, state),
        daemon=True,
    )
    box_thread.start()

    # 디버그용 쓰레드 생성후 동작
    resource_monitor = ResourceMonitor(
        recorder=recorder,
        get_inference_state=lambda: state["inference"],
        sample_interval=0.5,
    )
    resource_monitor.start()

    cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cam", 640, 480)

    print("=" * 60)
    print("Camera started")
    print("Waiting for motion...")
    print("Press 's' to stop")
    print("=" * 60)

    fps_start = time.time()
    fps_count = 0
    display_fps = 0.0

    try:
        while True:
            fps_count += 1
            now = time.time()

            if now - fps_start >= 1.0:
                display_fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

            ret, frame = recorder.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.")
                break
            #움직임이 감지되면 영상으로 저장후 경로 저장
            video_path = recorder.process(frame)

            # 새로운 녹화가 시작되면 이전 이벤트 결과를 화면에서 제거
            recording = recorder.recording
            if recording and not previous_recording:
                current_boxes = []
                print("[Display] 새로운 녹화 시작 -> 이전 BBox 초기화")
            previous_recording = recording

            # 완료된 추론 결과 처리
            # 추론이후 저장된 값들을 result에 저장
            while True:
                try:
                    result = result_queue.get_nowait()
                except Empty:
                    break

                try:
                    result_path = result.get("_video_path")
                    if latest_video_path is not None and result_path != latest_video_path:
                        print("[Display] 이전 영상의 결과 무시")
                        continue

                    # 추론이후 값을 저장
                    boxes = result.get("boxes", [])
                    # 탐지된 박스의 bbox값을 저장해 나중에 화면에 그리는데 사용
                    current_boxes = list(boxes) if result.get("detected") and boxes else []

                    if current_boxes:
                        print(f"[Display] {len(current_boxes)}개 Box 결과 수신")
                    else:
                        print("[Display] NO BOX -> 기존 BBox 제거")
                finally:
                    result_queue.task_done()

            # 새 영상이 저장되면 이전 결과를 먼저 지우고 검사 큐에 전달
            if video_path is not None:
                current_boxes = []
                latest_video_path = video_path
                print(f"\n[Main Thread] 영상 저장 완료\nVideo : {video_path}")
                print("기존 BBox 제거 -> Box 검사 Thread로 전달")
                check_queue.put(video_path)

            draw_boxes(frame, current_boxes)
            recorder.draw_status(frame)

            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f}",
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow("cam", frame)

            if cv2.waitKey(1) & 0xFF == ord("s"):
                print("\n'S' pressed.")
                break

    finally:
        print("\n프로그램 종료 중...")
        recorder.release()

        check_queue.put(None)
        box_thread.join(timeout=5)

        resource_monitor.stop()
        cv2.destroyAllWindows()
        resource_monitor.print_summary()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
