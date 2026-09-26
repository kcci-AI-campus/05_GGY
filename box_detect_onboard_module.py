import os

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


# ============================================================
# 모델 설정
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "box_26_320.tflite"
)

# MODEL_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "box_26.tflite"
# )


CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

CLASS_NAMES = ["box"]
BOX_CLASS_ID = 0


# ============================================================
# Letterbox
# ============================================================

def letterbox(
    image,
    size,
    color=(114, 114, 114)
):

    original_height, original_width = image.shape[:2]

    scale = min(
        size / original_width,
        size / original_height
    )

    resized_width = max(
        1,
        int(round(original_width * scale))
    )

    resized_height = max(
        1,
        int(round(original_height * scale))
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height)
    )


    pad_width = size - resized_width
    pad_height = size - resized_height

    left = pad_width // 2
    right = pad_width - left

    top = pad_height // 2
    bottom = pad_height - top


    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )


    return padded, scale, left, top


# ============================================================
# Tensor → float
# ============================================================

def tensor_to_float(
    tensor,
    details
):

    scale, zero_point = details["quantization"]


    if (
        np.issubdtype(
            tensor.dtype,
            np.integer
        )
        and scale
    ):

        return (
            tensor.astype(np.float32)
            - zero_point
        ) * scale


    return tensor.astype(np.float32)


# ============================================================
# float → Tensor
# ============================================================

def float_to_tensor(
    tensor,
    details
):

    dtype = details["dtype"]

    scale, zero_point = details["quantization"]


    if np.issubdtype(
        dtype,
        np.integer
    ):

        if not scale:

            raise ValueError(
                "양자화 입력 텐서의 scale이 0입니다."
            )


        info = np.iinfo(dtype)


        tensor = np.round(
            tensor / scale
            + zero_point
        )


        tensor = np.clip(
            tensor,
            info.min,
            info.max
        )


    return tensor.astype(dtype)


# ============================================================
# BoxDetector
# ============================================================

class BoxDetector:

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        class_id=BOX_CLASS_ID
    ):

        self.model_path = model_path

        self.confidence_threshold = \
            confidence_threshold

        self.iou_threshold = \
            iou_threshold

        self.class_id = class_id


        # ----------------------------------------------------
        # 모델 존재 확인
        # ----------------------------------------------------

        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: "
                f"{self.model_path}"
            )


        # ----------------------------------------------------
        # TFLite Interpreter
        # ----------------------------------------------------

        self.interpreter = \
            tflite.Interpreter(
                model_path=self.model_path
            )

        self.interpreter.allocate_tensors()


        # ----------------------------------------------------
        # Tensor 정보
        # ----------------------------------------------------

        self.input_details = \
            self.interpreter.get_input_details()[0]

        self.output_details = \
            self.interpreter.get_output_details()


        # ----------------------------------------------------
        # 입력 크기
        # ----------------------------------------------------

        input_shape = \
            self.input_details["shape"]


        if len(input_shape) != 4:

            raise ValueError(
                f"지원하지 않는 입력 텐서 형태입니다: "
                f"{input_shape}"
            )


        if input_shape[1] in (1, 3):  # (batch,channel,w,h) 이면 뒤에 2값을 사이즈로 가져옴

            self.input_height = \
                int(input_shape[2])

            self.input_width = \
                int(input_shape[3])

            self.channel_first = True

        else:

            self.input_height = \
                int(input_shape[1])

            self.input_width = \
                int(input_shape[2])

            self.channel_first = False


    # ========================================================
    # YOLO 추론
    # ========================================================

    def detect(
        self,
        frame,
        draw=False
    ):

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Letterbox
        # ----------------------------------------------------

        padded, scale, pad_x, pad_y = \
            letterbox(
                rgb,
                self.input_width
            )


        # ----------------------------------------------------
        # 정규화
        # ----------------------------------------------------

        image = (
            padded.astype(np.float32)
            / 255.0
        )


        # ----------------------------------------------------
        # Channel First
        # ----------------------------------------------------

        if self.channel_first:

            image = np.transpose(
                image,
                (2, 0, 1)
            )


        # ----------------------------------------------------
        # Batch dimension
        # ----------------------------------------------------

        image = np.expand_dims(
            image,
            axis=0
        )


        # ----------------------------------------------------
        # 입력 Tensor
        # ----------------------------------------------------

        self.interpreter.set_tensor(
            self.input_details["index"],
            float_to_tensor(
                image,
                self.input_details
            )
        )


        # ----------------------------------------------------
        # 추론
        # ----------------------------------------------------

        self.interpreter.invoke()


        # ----------------------------------------------------
        # 출력 Tensor
        # ----------------------------------------------------

        raw = tensor_to_float(
            self.interpreter.get_tensor(
                self.output_details[0]["index"]
            ),
            self.output_details[0]
        )


        raw = np.squeeze(raw)


        if raw.ndim != 2:

            raise ValueError(
                f"지원하지 않는 출력 텐서 형태입니다: "
                f"{raw.shape}"
            )


        if raw.shape[0] < raw.shape[1]:

            raw = raw.T  #(nnnn,5) 후보객체,상자좌표+클래스 구조로 shape의 차원을 수정


        if raw.shape[1] < 5:

            raise ValueError(
                f"출력 텐서에 박스 정보가 없습니다: "
                f"{raw.shape}"
            )


        # ====================================================
        # 클래스 점수
        # ====================================================

        scores = np.max(
            raw[:, 4:],
            axis=1
        )


        class_ids = np.argmax(
            raw[:, 4:],
            axis=1
        )


        # ====================================================
        # Confidence + Class 필터
        # ====================================================

        candidates = np.flatnonzero(
            (scores >= self.confidence_threshold)
            &
            (class_ids == self.class_id)
        )


        if candidates.size == 0:

            return []


        # ====================================================
        # Box 생성
        # ====================================================

        boxes = []

        candidate_scores = []

        candidate_classes = []


        for index in candidates:

            cx, cy, width, height = \
                raw[index, :4]


            x = (
                cx
                - width / 2
            ) * self.input_width


            y = (
                cy
                - height / 2
            ) * self.input_height


            boxes.append(
                [
                    x,
                    y,
                    width * self.input_width,
                    height * self.input_height
                ]
            )


            candidate_scores.append(
                float(scores[index])
            )


            candidate_classes.append(
                int(class_ids[index])
            )


        # ====================================================
        # NMS
        # ====================================================

        keep = cv2.dnn.NMSBoxesBatched(
            boxes,
            candidate_scores,
            candidate_classes,
            self.confidence_threshold,
            self.iou_threshold
        )


        results = []


        # ====================================================
        # 최종 결과
        # ====================================================

        for kept_index in np.asarray(
            keep
        ).reshape(-1):

            kept_index = int(
                kept_index
            )


            x, y, width, height = \
                boxes[kept_index]


            x1 = int(
                np.clip(
                    (x - pad_x) / scale,
                    0,
                    frame.shape[1] - 1
                )
            )


            y1 = int(
                np.clip(
                    (y - pad_y) / scale,
                    0,
                    frame.shape[0] - 1
                )
            )


            x2 = int(
                np.clip(
                    (x + width - pad_x) / scale,
                    0,
                    frame.shape[1] - 1
                )
            )


            y2 = int(
                np.clip(
                    (y + height - pad_y) / scale,
                    0,
                    frame.shape[0] - 1
                )
            )


            score = candidate_scores[
                kept_index
            ]


            class_id = candidate_classes[
                kept_index
            ]


            # ------------------------------------------------
            # 결과 저장
            # ------------------------------------------------

            results.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    score,
                    class_id
                ]
            )


            # ------------------------------------------------
            # 필요할 때만 화면에 표시
            # ------------------------------------------------

            if draw:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"box {score:.0%}",
                    (
                        x1,
                        max(
                            20,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )


        return results


    # ========================================================
    # model(frame) 형태로도 사용할 수 있도록 함
    # ========================================================
    #클래스를 객체로 생성해 사용할때 객체이름()으로도 사용가능
    def __call__(
        self,
        frame
    ):

        return self.detect(
            frame,
            draw=False
        )

