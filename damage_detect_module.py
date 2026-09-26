# damage_detect_module.py
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class DamageDetector:

    def __init__(
        self,
        model_path="damage_detect_float32.tflite",
        damaged_class_id=1,
        confidence_threshold=0.5
    ):
        self.model_path = model_path
        self.damaged_class_id = damaged_class_id
        self.confidence_threshold = confidence_threshold

        # -----------------------------------------
        # TFLite 모델 로드
        # -----------------------------------------

        self.interpreter = tflite.Interpreter(
            model_path=self.model_path
        )

        self.interpreter.allocate_tensors()

        # 입력 / 출력 정보
        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        # -----------------------------------------
        # 입력 shape 확인
        # -----------------------------------------

        input_shape = self.input_details[0]["shape"]

        # 일반적인 이미지 입력:
        # [1, height, width, channels]
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        self.input_channels = int(input_shape[3])

        print("\n[DamageDetector]")
        print(f"Model       : {self.model_path}")
        print(f"Input shape : {input_shape}")

        for i, output in enumerate(
            self.output_details
        ):
            print(
                f"Output {i} shape : "
                f"{output['shape']}"
            )

        print()

    # -----------------------------------------
    # 입력 이미지 전처리
    # -----------------------------------------

    def preprocess(self, frame):

        # BGR → RGB
        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # 모델 입력 크기로 resize
        image = cv2.resize(
            image,
            (
                self.input_width,
                self.input_height
            )
        )

        # float32 변환
        image = image.astype(
            np.float32
        )

        # -----------------------------------------
        # Float32 모델의 일반적인 정규화
        # -----------------------------------------
        # 0 ~ 255
        # ↓
        # 0.0 ~ 1.0
        image /= 255.0

        # batch dimension 추가
        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    # -----------------------------------------
    # 추론
    # -----------------------------------------

    def predict(self, frame):

        if frame is None:
            return {
                "damaged": False,
                "confidence": 0.0,
                "class_id": None
            }

        # 전처리
        input_data = self.preprocess(
            frame
        )

        # 입력 tensor 설정
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_data
        )

        # 추론
        self.interpreter.invoke()

        # 출력 가져오기
        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # batch 제거
        output = np.squeeze(output)

        # -----------------------------------------
        # 출력 형태에 따른 처리
        # -----------------------------------------

        # 예:
        # [0.9, 0.1]
        # [0.1, 0.9]
        #
        # 와 같은 classification 결과를 가정
        if output.ndim == 1 and len(output) > 1:

            probabilities = output

            class_id = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[class_id]
            )

        # -----------------------------------------
        # binary classification
        #
        # 예:
        # [0.83]
        # -----------------------------------------

        elif output.size == 1:

            value = float(
                output.item()
            )

            # sigmoid 출력이라고 가정
            confidence = value

            if value >= 0.5:
                class_id = 1
            else:
                class_id = 0

        else:

            raise ValueError(
                "지원하지 않는 모델 출력 형태입니다. "
                f"Output shape: {output.shape}"
            )

        # -----------------------------------------
        # 파손 여부
        # -----------------------------------------

        damaged = (
            class_id == self.damaged_class_id
            and confidence >= self.confidence_threshold
        )

        return {
            "damaged": damaged,
            "confidence": confidence,
            "class_id": class_id
        }

    # -----------------------------------------
    # 호출형 인터페이스
    # -----------------------------------------

    def __call__(self, frame):

        return self.predict(frame)
