# save_plus_boxdetect.py

import cv2
import time

from threading import Thread
from queue import Queue, Empty

from motion_save_moduel import MotionRecorder
from video_box_detect_module import BoxVideoChecker


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
#
# 저장된 영상의 경로를 전달
#
# Main
#   ↓
# box_check_queue
#   ↓
# Box Thread
#
# ==========================================================

box_check_queue = Queue()


# ==========================================================
# Box Result Queue
# ==========================================================
#
# Box Thread에서 검사한 결과를
# Main Thread로 전달
#
# Box Thread
#   ↓
# box_result_queue
#   ↓
# Main
#
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
# Box Check Worker
# ==========================================================

def box_check_worker():

    print(
        "[Box Thread] 시작"
    )

    # ------------------------------------------------------
    # BoxVideoChecker 생성
    #
    # YOLO + Damage 모델은 이 Thread 안에서 사용
    # ------------------------------------------------------

    box_checker = BoxVideoChecker(

        box_class_id=0,

        confidence_threshold=0.4,

        check_seconds=3,

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
        # 영상 검사
        # ==================================================

        result = box_checker.check_and_delete(
            video_path
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
        # 검사 결과를 Main Thread로 전달
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
        # 2. 320 × 240으로 resize
        # ==================================================

        # frame = cv2.resize(

        #     frame,

        #     (320, 240)
        # )


        # ==================================================
        # 3. Motion Detection
        # ==================================================

        video_path = recorder.process(
            frame
        )


        # ==================================================
        # 4. Box + Damage 검사 결과 확인
        # ==================================================
        #
        # Box Thread가 비동기로 검사하기 때문에
        # 결과가 있을 때만 가져온다.
        #
        # YOLO / Damage 추론은 여기서 실행하지 않는다.
        #
        # ==================================================

        try:

            while True:

                result = box_result_queue.get_nowait()


                # ------------------------------------------
                # Box가 검출된 경우
                # ------------------------------------------

                if result["detected"]:

                    current_bbox = result["bbox"]

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
        # 5. Main 화면에 BBox 그리기
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


            # --------------------------------------------------
            # Damage 결과 표시
            # --------------------------------------------------

            cv2.putText(

                frame,

                damage_label,

                (
                    x1,
                    min(y2 + 18, 235)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 0, 255)
                if current_damaged
                else (0, 255, 0),

                2
            )


        # ==================================================
        # 6. Recorder 상태 표시
        # ==================================================

        frame = recorder.draw_status(
            frame
        )


        # ==================================================
        # 7. FPS 표시
        # ==================================================

        cv2.putText(

            frame,

            f"FPS: {display_fps:.1f}",

            (10, 225),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )


        # ==================================================
        # 8. 화면 출력
        # ==================================================

        cv2.imshow(
            "cam",
            frame
        )


        # ==================================================
        # 9. 저장 완료된 영상 확인
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
            # 영상 경로만 Box Thread로 전달
            # --------------------------------------------------

            box_check_queue.put(
                video_path
            )


        # ==================================================
        # 10. 키 입력
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


    # ------------------------------------------------------
    # OpenCV 종료
    # ------------------------------------------------------

    cv2.destroyAllWindows()


    print(
        "Camera stopped."
    )
