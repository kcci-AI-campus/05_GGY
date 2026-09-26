def check(self, video_path):

    print("\n[Box Check]")
    print(f"Video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        cap.release()

        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0
        }

    duration = frame_count / fps

    # 마지막 3초 검사
    start_time = max(
        0,
        duration - self.check_seconds
    )

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        start_time * 1000
    )

    # --------------------------------------------------
    # 1초 판단을 위한 변수
    # --------------------------------------------------

    required_seconds = 1.0

    required_frames = int(
        fps * required_seconds
    )

    detected_frames = 0

    last_bbox = None
    last_confidence = 0.0

    # 80% 이상 검출되면 Box 존재
    detection_ratio = 0.8

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = self.model(frame)

            box_detected = False

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

                # Box 검출
                box_detected = True

                last_bbox = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                last_confidence = confidence

                break

            # --------------------------------------------------
            # 현재 프레임에서 Box 검출 여부
            # --------------------------------------------------

            if box_detected:

                detected_frames += 1

            # --------------------------------------------------
            # 1초 분량 프레임이 모였으면 판단
            # --------------------------------------------------

            if detected_frames >= int(
                required_frames * detection_ratio
            ):

                print("Box detected!")

                print(
                    f"Detected frames : "
                    f"{detected_frames}"
                )

                print(
                    f"Required frames : "
                    f"{required_frames}"
                )

                print(
                    f"BBox : {last_bbox}"
                )

                print(
                    f"Confidence : "
                    f"{last_confidence:.2f}"
                )

                return {
                    "detected": True,
                    "bbox": last_bbox,
                    "confidence": last_confidence
                }

        # --------------------------------------------------
        # 검사 종료
        # --------------------------------------------------

        print("Box not detected.")

        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0
        }

    finally:

        cap.release()