import cv2
import time

from threading import Thread
from queue import Queue

from motion_save_moduel import MotionRecorder
from video_box_detect_module import BoxVideoChecker
from box_detect_onboard_module import BoxDetector


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
# Live Box Detector
# ==========================================================

live_box_detector = BoxDetector(
    confidence_threshold=0.4,
    class_id=0
)


# ==========================================================
# Box Check Queue
# ==========================================================
#영상의 주소가 들어감
box_check_queue = Queue()


# ==========================================================
# Live Box Detection
# ==========================================================

def detect_box_on_frame(
    frame,
    detector
):

    results = detector(
        frame
    )

    for detection in results:

        # ----------------------------------------------
        # Detection 정보
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Box class만 표시
        # ----------------------------------------------
        # 한번더 id, TH 확인 하는건가여??
        if class_id != 0:
            continue

        # ----------------------------------------------
        # Confidence 확인
        # ----------------------------------------------

        if confidence < 0.4:
            continue

        # ----------------------------------------------
        # Bounding Box
        # ----------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # ----------------------------------------------
        # Confidence 표시
        # ----------------------------------------------

        label = (
            f"BOX "
            f"{confidence*100:.2f}%"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    return frame


# ==========================================================
# Box Check Worker
# ==========================================================

def box_check_worker():

    print(
        "[Box Thread] 시작"
    )

    # ------------------------------------------------------
    # BoxDetector는 이 Thread에서 생성
    # ------------------------------------------------------

    box_checker = BoxVideoChecker(
        box_class_id=0,
        confidence_threshold=0.4,
        check_seconds=3
    )

    while True:

        # --------------------------------------------------
        # 저장 완료된 영상 기다림
        # --------------------------------------------------

        video_path = box_check_queue.get()

        # None이면 Thread 종료
        if video_path is None:

            print(
                "[Box Thread] 종료"
            )

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

        # --------------------------------------------------
        # YOLO Box 검사
        # --------------------------------------------------

        box_exists = box_checker.check_and_delete(
            video_path
        )

        # --------------------------------------------------
        # 결과
        # --------------------------------------------------

        if box_exists:

            print(
                "\n================================"
            )

            print(
                "Final result: BOX DETECTED"
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

        # Queue 작업 완료
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
    # ==========================================================
    # FPS 측정
    # ==========================================================

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

        # --------------------------------------------------
        # 1. Camera Frame 읽기
        # --------------------------------------------------

        ret, frame = recorder.read()

        if not ret:

            print(
                "카메라 프레임을 읽을 수 없습니다."
            )

            break

        # --------------------------------------------------
        # 2. 실제 Camera는 640×480
        #
        # 따라서 프로그램에서 320×240으로 resize
        # --------------------------------------------------

        # frame = cv2.resize(
        #     frame,
        #     (320, 240)
        # )

        # --------------------------------------------------
        # 3. Motion Detection
        #
        # 여기서 영상 저장은 별도 Thread가 담당
        # --------------------------------------------------

        video_path = recorder.process(
            frame
        )

        # ==================================================
        # Live Box Detection
        # ==================================================

        frame = detect_box_on_frame(
            frame,
            live_box_detector
        )

        # --------------------------------------------------
        # 4. 화면 상태 표시
        # --------------------------------------------------

        frame = recorder.draw_status(
            frame
        )   

        # --------------------------------------------------
        # FPS 표시
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {display_fps:.1f}",
            (10, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "cam",
            frame
        )

        # --------------------------------------------------
        # 6. 저장 완료된 영상 확인
        # --------------------------------------------------

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

            # ----------------------------------------------
            # YOLO 검사를 직접 하지 않는다.
            #
            # Queue에 영상 경로만 전달
            # ----------------------------------------------

            box_check_queue.put(
                video_path
            )

        # --------------------------------------------------
        # 7. 키 입력
        # --------------------------------------------------

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
    # Box Thread 종료
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

    print(
        "Camera stopped."
    )