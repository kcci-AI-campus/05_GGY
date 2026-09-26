# motion_save_moduel.py
import cv2
import os
import time

from collections import deque
from datetime import datetime
from queue import Queue, Empty
from threading import Thread


class MotionRecorder:

    def __init__(
        self,
        cap,
        camera_width=320,
        camera_height=240,
        fps=15,
        buffer_seconds=20,
        save_dir="video",
        min_motion_area=1500
    ):

        self.cap = cap  #프레임 을 cap으로 전달받음

        self.camera_width = camera_width   #마케라 해상도
        self.camera_height = camera_height

        self.fps = fps   #카메라 fps가정

        self.buffer_seconds = buffer_seconds  #녹화할 사건이전시간설정 변수
        self.buffer_size = fps * buffer_seconds #카메라 fps와 사건이전시간을 통해 녹화할 프레임수를 계싼

        self.save_dir = save_dir  # 영상을 저장할 주소
        self.min_motion_area = min_motion_area # 모션디텍트할때의 민감도

        os.makedirs(self.save_dir, exist_ok=True)  #영상을 저장할 폴더를 생성



        # --------------------------------------------------
        # Motion Detection
        # --------------------------------------------------
        '''
         배경차분알고리즘(MOG2)생성, 배경을 학습해 움직이는 물체(전경)을 탐지
         history : 배경을 학습할 프레임의 범위, 프레임이 고정되지않은 500짜리 버퍼 느낌
         varThreshold : 모델이 배경과 차이를 느낄 민감도
         detectShadows : 그림자를 별도로 감지할지 결정 -> 그림자를 움직임으로
         background_subtractor : 모델의 탐지 결과가 저장됨
        '''
        self.background_subtractor = \
            cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=50,
                detectShadows=True
            )

        # --------------------------------------------------
        # 이전 프레임 저장
        # --------------------------------------------------
        # depue(앞뒤에서 입,출력)형식으로 버퍼를 만들어 이전프레임을 저장
        # maxel키워드로 버퍼사이즈를 설정, 이 이상의 갯수가 들어오면 오래된게 사라짐
        self.frame_buffer = deque(
            maxlen=self.buffer_size
        )

        # --------------------------------------------------
        # Recording 상태
        # --------------------------------------------------

        self.recording = False # 현재상태 변수 true이면 녹화중

        # 실제 파일 저장 작업 중인지
        self.saving = False

        self.video_writer = None
        self.save_path = None # write메서드는 none이들어가면 영상 저장을 끝냄
        # 움직임이 끝난 후 추가 녹화할 시간
        self.motion_end_delay = 2.0
        self.motion_active = False

        # 움직임이 마지막으로 감지된 시간
        self.last_motion_time = None

        # --------------------------------------------------
        # Recording Thread용 Queue
        # --------------------------------------------------

        self.record_queue = None

        self.recording_thread = None

        # 저장이 끝난 영상 경로를 전달하는 Queue
        self.finished_video_queue = Queue()

    # ======================================================
    # Camera
    # ======================================================

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return False, None

        return True, frame

    # ======================================================
    # Motion Detection
    # ======================================================

    def detect_motion(self, frame):

        motion_mask = self.background_subtractor.apply(
            frame
        )

        #그림자 제거
        _, motion_mask = cv2.threshold(
            motion_mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # 노이즈 제거
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            None
        )

        #움직임 영역 찾기
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False

        for contour in contours:

            area = cv2.contourArea(contour)

            if area >= self.min_motion_area:

                motion_detected = True

        return motion_detected

    # ======================================================
    # Recording Worker
    # ======================================================

    def _record_worker(
        self,
        prebuffer,
        record_queue,
        save_path
    ):

        print("\n[Recording Thread]")
        print("VideoWriter 시작")
        print(f"Save path : {save_path}")

        #영상인코딩 방식을 카메라와 동일 하게 MJPG로 설정
        fourcc = cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'
        )

        # 영상을 저장할 준비 설정(프레임, 인코딩, 저장 경로,해상도)
        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            self.fps,
            (
                self.camera_width,
                self.camera_height
            )
        )

        # 영상저장 준비가 됐는지 확인해 debig문구 출력
        if not writer.isOpened():

            print("\n[ERROR]")
            print("VideoWriter를 열 수 없습니다.")
            print(f"Save path : {save_path}")

            self.saving = False

            return

        #만들어둔 영상 저장 설정을 인스턴스변수에 저장
        self.video_writer = writer

        # --------------------------------------------------
        # 1. 이전 20초 영상 저장
        # --------------------------------------------------

        print(
            f"Previous buffer : "
            f"{len(prebuffer)} frames"
        )

        for frame in prebuffer:

            writer.write(frame)

        print(
            "Previous buffer 저장 완료"
        )

        # --------------------------------------------------
        # 2. 실시간 프레임 저장
        # --------------------------------------------------

        while True:

            frame = record_queue.get()

            # None = 저장 종료 신호
            if frame is None:
                break

            writer.write(frame)

        # --------------------------------------------------
        # 3. VideoWriter 종료
        # --------------------------------------------------

        writer.release()

        self.video_writer = None

        self.saving = False

        print("\n[Recording Thread]")
        print("VideoWriter 종료")
        print(f"Video saved : {save_path}")

        # 저장 완료된 영상 경로 전달
        self.finished_video_queue.put(
            save_path
        )

    # ======================================================
    # Start Recording
    # ======================================================

    def start_recording(self):

        if self.recording:
            return

        # 이전 영상 저장이 아직 끝나지 않았다면
        if self.saving:
            return

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.save_path = os.path.join(
            self.save_dir,
            f"event_{timestamp}.avi"
        )

        # --------------------------------------------------
        # 현재 frame은 이미 buffer에 들어가 있다.
        #
        # 따라서 마지막 frame은 제외한다.
        # 이후 process()에서 현재 frame을 다시 Queue에 넣는다.
        # --------------------------------------------------

        prebuffer = list(
            self.frame_buffer
        )

        if len(prebuffer) > 0:

            prebuffer = prebuffer[:-1]

        # --------------------------------------------------
        # Recording Queue 생성
        # --------------------------------------------------

        self.record_queue = Queue()

        # 상태 변경
        self.recording = True
        self.saving = True

        # Motion 종료 타이머 초기화
        self.last_motion_time = time.time()

        # --------------------------------------------------
        # Recording Thread 시작
        # --------------------------------------------------

        self.recording_thread = Thread(
            target=self._record_worker,
            args=(
                prebuffer,
                self.record_queue,
                self.save_path
            ),
            daemon=True  # 메인프로그램이 종료되면 작업이 남아있어도 쓰레드를 종료함
        )

        self.recording_thread.start()

        print("\n[Motion detected]")

        print(
            f"Saving previous "
            f"{self.buffer_seconds} seconds..."
        )

        print(
            "Recording thread started"
        )

    # ======================================================
    # Stop Recording
    # ======================================================

    def stop_recording(self):

        if not self.recording:
            return

        print(
            "\n[Motion stopped]"
        )

        print(
            "Recording 종료 요청"
        )

        # --------------------------------------------------
        # recording=False
        #
        # 실제 VideoWriter 종료는 worker가 담당한다.
        # --------------------------------------------------

        self.recording = False

        # Queue에 None을 넣으면
        # Recording Thread가 저장을 종료한다.
        if self.record_queue is not None:

            self.record_queue.put(
                None
            )

    # ======================================================
    # Main Processing
    # ======================================================

    def process(self, frame):

        # --------------------------------------------------
        # 현재 프레임의 원본 보관
        # --------------------------------------------------

        clean_frame = frame.copy()

        # --------------------------------------------------
        # 이전 프레임 Buffer
        # --------------------------------------------------

        self.frame_buffer.append(
            clean_frame
        )

        # --------------------------------------------------
        # Motion Detection
        # --------------------------------------------------

        motion_detected = self.detect_motion(
            frame
        )

        # 현재 시간
        current_time = time.time()

        # --------------------------------------------------
        # Motion 발생
        # --------------------------------------------------

        if motion_detected:

            self.motion_active = True

            self.last_motion_time = current_time # 움직임 탐지 중이라면 계속 갱신되는 값

            if (
                not self.recording
                and
                not self.saving
            ):
                self.start_recording()

        else:

            self.motion_active = False

        # --------------------------------------------------
        # Recording 중
        # --------------------------------------------------

        if self.recording:

            # 현재 프레임 저장
            self.record_queue.put(
                clean_frame
            )

            # --------------------------------------------------
            # Motion이 감지되지 않는 경우
            # --------------------------------------------------

            if not motion_detected:

                # 마지막 Motion 이후 경과 시간
                elapsed_time = (
                    current_time
                    -
                    self.last_motion_time
                )

                # --------------------------------------------------
                # 2초가 지났는지 확인
                # --------------------------------------------------

                if elapsed_time >= self.motion_end_delay:

                    print(
                        "\n[Motion timeout]"
                    )

                    print(
                        f"No motion for "
                        f"{self.motion_end_delay:.1f} seconds."
                    )

                    self.stop_recording()

        # --------------------------------------------------
        # 저장 완료된 영상 확인
        # --------------------------------------------------

        finished_video = self.get_finished_video()

        return finished_video

    # ======================================================
    # 저장 완료 영상 가져오기
    # ======================================================
        # 왜 단순히 인스턴스 변수에 접근하지 않고 함수로 만들어 값을 가져와야할까?
        # .get()메서드로 가져오면 값이 없을때 값이 들어올때까지 대기를함 -> blocking
        # get_nowait()를 사용하면 큐에서 가장오래된 값하나를 꺼내오는데 이러면 큐에는 그값이 사라짐
    def get_finished_video(self):

        try:

            video_path = \
                self.finished_video_queue.get_nowait()

            return video_path

        except Empty:

            return None

    # ======================================================
    # 화면 상태 표시
    # ======================================================

    def draw_status(self, frame):

        if self.recording:

            # ----------------------------------------------
            # 현재 녹화 중
            # ----------------------------------------------

            cv2.putText(
                frame,
                "RECORDING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            # ----------------------------------------------
            # Motion이 사라졌다면
            # 종료까지 남은 시간 표시
            # ----------------------------------------------

            if (
                not self.motion_active
                and
                self.last_motion_time is not None
            ):

                elapsed = (
                    time.time()
                    -
                    self.last_motion_time
                )

                remaining = max(
                    0,
                    self.motion_end_delay - elapsed
                )

                cv2.putText(
                    frame,
                    f"End in: {remaining:.1f}s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        elif self.saving:

            cv2.putText(
                frame,
                "SAVING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "WAITING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        return frame
    # ======================================================
    # 종료
    # ======================================================

    def release(self):

        # --------------------------------------------------
        # Recording 중이라면 종료 요청
        # --------------------------------------------------

        if self.recording:

            self.stop_recording()

        # --------------------------------------------------
        # Recording Thread가 끝날 때까지 잠시 기다림
        # --------------------------------------------------

        if self.recording_thread is not None:

            if self.recording_thread.is_alive():

                print(
                    "Recording Thread 종료 대기..."
                )

                self.recording_thread.join(
                    timeout=5
                )

        # --------------------------------------------------
        # Camera 종료
        # --------------------------------------------------

        if self.cap is not None:

            self.cap.release()

        cv2.destroyAllWindows()