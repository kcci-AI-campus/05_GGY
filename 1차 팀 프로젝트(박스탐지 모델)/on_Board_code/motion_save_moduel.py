import os
import time
import cv2

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
        min_motion_area=1500,
    ):
        self.cap = cap
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.buffer_size = fps * buffer_seconds
        self.save_dir = save_dir
        self.min_motion_area = min_motion_area
        self.motion_end_delay = 2.0

        os.makedirs(save_dir, exist_ok=True) # 영상 저장 주소 확인

        # 모션감지 모델 객채 생성
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True,
        )
        # 이전 시간 저장공간 큐 생성
        self.frame_buffer = deque(maxlen=self.buffer_size)

        self.recording = False
        self.saving = False
        self.motion_active = False
        self.last_motion_time = None
        
        self.record_queue = None
        self.recording_thread = None
        self.video_writer = None
        self.save_path = None
        self.finished_video_queue = Queue()

    def read(self):
        return self.cap.read()

    # 모션감지 객체를 생성하고 TH를 넘은 영역을 return
    def detect_motion(self, frame):
        mask = self.background_subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return any(cv2.contourArea(c) >= self.min_motion_area for c in contours)

    # 이전 화면, 현재 화면을 영상으로 저장
    def _record_worker(self, prebuffer, queue, save_path):
        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            self.fps,
            (self.camera_width, self.camera_height),
        )

        if not writer.isOpened():
            print("[ERROR] VideoWriter를 열 수 없습니다.")
            self.saving = False
            return

        self.video_writer = writer
        print(f"Previous buffer : {len(prebuffer)} frames")

        for frame in prebuffer:
            writer.write(frame)

        while True:
            frame = queue.get()
            if frame is None:
                break
            writer.write(frame)

        writer.release()
        self.video_writer = None
        self.saving = False
        self.finished_video_queue.put(save_path)
        print(f"Video saved : {save_path}")

    #상태 변경, 프레임 화면을 저장공간에 저장하고 record_worker()를 쓰레드로 실행해 영상을 저장함
    def start_recording(self):
        if self.recording or self.saving:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(
            self.save_dir,
            f"event_{timestamp}.avi",
        )

        # 현재 frame은 process()에서 queue에도 들어가므로 중복 저장 방지
        prebuffer = list(self.frame_buffer)
        if prebuffer:
            prebuffer.pop()

        self.record_queue = Queue()
        self.recording = True
        self.saving = True
        self.last_motion_time = time.time()

        self.recording_thread = Thread(
            target=self._record_worker,
            args=(prebuffer, self.record_queue, self.save_path),
            daemon=True,
        )
        self.recording_thread.start()

        print(f"Saving previous {self.buffer_seconds} seconds...")
        print("Recording thread started")

    # 현재 프레임을 저장하는 큐에 None을 입력해 저장을 종료시킴
    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        if self.record_queue is not None:
            self.record_queue.put(None)

    # 위에 함수들을 합침// 프레임 저장 -> 모션감지 -> 영상저장 -> 움직임이 2초동안 멈추면 저장 종료 -> 영상 경로를 return
    def process(self, frame):
        clean_frame = frame.copy()
        self.frame_buffer.append(clean_frame)

        motion = self.detect_motion(frame)
        now = time.time()
        self.motion_active = motion

        if motion:
            self.last_motion_time = now
            if not self.recording and not self.saving:
                self.start_recording()

        if self.recording:
            self.record_queue.put(clean_frame)

            if (
                not motion
                and self.last_motion_time is not None
                and now - self.last_motion_time >= self.motion_end_delay
            ):
                print(
                    f"[Motion timeout] "
                    f"No motion for {self.motion_end_delay:.1f} seconds."
                )
                self.stop_recording()

        return self.get_finished_video()

    #영상 저장 경로를 return
    def get_finished_video(self):
        try:
            return self.finished_video_queue.get_nowait()
        except Empty:
            return None

    #현재 상태에 따라 표시창을 화면에 그려줌
    def draw_status(self, frame):
        if self.recording:
            cv2.putText(
                frame,
                "RECORDING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            #움직임이 멈춘 시간 표시
            if not self.motion_active and self.last_motion_time is not None:
                remaining = max(
                    0,
                    self.motion_end_delay
                    - (time.time() - self.last_motion_time),
                )
                cv2.putText(
                    frame,
                    f"End in: {remaining:.1f}s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        elif self.saving:
            cv2.putText(
                frame,
                "SAVING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "WAITING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        return frame

    #영상 저장 작업을 종료// 프로그램이 종료될때 사용됨
    def release(self):
        if self.recording:
            self.stop_recording()

        if self.recording_thread and self.recording_thread.is_alive():
            print("Recording Thread 종료 대기...")
            self.recording_thread.join(timeout=5)

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
