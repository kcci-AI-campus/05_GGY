import os

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "box_26_320.tflite",
)

#TH 설정, class index 설정
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
BOX_CLASS_ID = 0


def letterbox(image, size, color=(114, 114, 114)):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(image, (new_width, new_height))

    pad_width = size - new_width
    pad_height = size - new_height
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
        value=color,
    )
    return padded, scale, left, top

# 역양자화
def tensor_to_float(tensor, details):
    scale, zero_point = details["quantization"]

    if np.issubdtype(tensor.dtype, np.integer) and scale:
        return (tensor.astype(np.float32) - zero_point) * scale

    return tensor.astype(np.float32)

#양자화
def float_to_tensor(tensor, details):
    dtype = details["dtype"]
    scale, zero_point = details["quantization"]

    if np.issubdtype(dtype, np.integer):
        if not scale:
            raise ValueError("양자화 입력 텐서의 scale이 0입니다.")

        info = np.iinfo(dtype)
        tensor = np.round(tensor / scale + zero_point)
        tensor = np.clip(tensor, info.min, info.max)

    return tensor.astype(dtype)


class BoxDetector:
    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        class_id=BOX_CLASS_ID,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}"
            )

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_id = class_id

        #모델의 interpreter 객체 생성및 입출력 형태 확인 
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details["shape"]
        if len(shape) != 4:
            raise ValueError(f"지원하지 않는 입력 텐서 형태입니다: {shape}")

        # shape의 W,H값의 위치를 확인후 저장
        self.channel_first = shape[1] in (1, 3)
        if self.channel_first:
            self.input_height, self.input_width = map(int, shape[2:4])
        else:
            self.input_height, self.input_width = map(int, shape[1:3])


    def detect(self, frame, draw=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, scale, pad_x, pad_y = letterbox(rgb, self.input_width)

        image = padded.astype(np.float32) / 255.0
        if self.channel_first:
            image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        self.interpreter.set_tensor(
            self.input_details["index"],
            float_to_tensor(image, self.input_details),
        )
        self.interpreter.invoke()

        raw = tensor_to_float(
            self.interpreter.get_tensor(self.output_details[0]["index"]),
            self.output_details[0],
        )
        # 모델 추론후 결과를 raw에 저장 
        raw = np.squeeze(raw)

        if raw.ndim != 2:
            raise ValueError(f"지원하지 않는 출력 텐서 형태입니다: {raw.shape}")
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if raw.shape[1] < 5:
            raise ValueError(f"출력 텐서에 박스 정보가 없습니다: {raw.shape}")

        # 추론결과를 항목 별로 저장, TH적용및 클래스가 상자인 후보만 저장
        # (scores,class_ids에는 모든 후보가 들어가지만 candodates를 index로 활용해 구분해 활용)
        scores = np.max(raw[:, 4:], axis=1)
        class_ids = np.argmax(raw[:, 4:], axis=1)
        candidates = np.flatnonzero(
            (scores >= self.confidence_threshold)
            & (class_ids == self.class_id)
        )

        if candidates.size == 0:
            return []

        # NMS이전 상자의 값들이 저장될 저장공간
        boxes = []
        candidate_scores = []
        candidate_classes = []
        for index in candidates:
            cx, cy, width, height = raw[index, :4]
            boxes.append([
                (cx - width / 2) * self.input_width,
                (cy - height / 2) * self.input_height,
                width * self.input_width,
                height * self.input_height,
            ])
            candidate_scores.append(float(scores[index]))
            candidate_classes.append(int(class_ids[index]))

        #NMS 적용후 결과를 keep에 저장
        keep = cv2.dnn.NMSBoxesBatched(
            boxes,
            candidate_scores,
            candidate_classes,
            self.confidence_threshold,
            self.iou_threshold,
        )

        #NMS 적용 이후 바운딩박스의 좌표를 실제 좌표 값으로 변환하여 results에 저장
        results = []
        for kept_index in np.asarray(keep).reshape(-1):
            kept_index = int(kept_index)
            x, y, width, height = boxes[kept_index]

            #좌표를 원본 이미지에 맞게 변환
            x1 = int(np.clip((x - pad_x) / scale, 0, frame.shape[1] - 1))
            y1 = int(np.clip((y - pad_y) / scale, 0, frame.shape[0] - 1))
            x2 = int(np.clip((x + width - pad_x) / scale, 0, frame.shape[1] - 1))
            y2 = int(np.clip((y + height - pad_y) / scale, 0, frame.shape[0] - 1))

            score = candidate_scores[kept_index]
            class_id = candidate_classes[kept_index]

            results.append([x1, y1, x2, y2, score, class_id])

            if draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"box {score:.0%}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        return results

    def __call__(self, frame):
        return self.detect(frame, draw=False)
