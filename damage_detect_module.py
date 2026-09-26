# damage_detect_module.py

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class DamageDetector:

    def __init__(
        self,
        model_path="damage_detect_float32.tflite",
        damaged_class_id=0,
        confidence_threshold=0.5
    ):

        self.model_path = model_path

        self.damaged_class_id = \
            damaged_class_id

        self.confidence_threshold = \
            confidence_threshold

        # --------------------------------------------------
        # TFLite Interpreter
        # --------------------------------------------------

        self.interpreter = tflite.Interpreter(
            model_path=self.model_path
        )

        self.interpreter.allocate_tensors()

        # --------------------------------------------------
        # Input / Output 정보
        # --------------------------------------------------

        self.input_details = \
            self.interpreter.get_input_details()

        self.output_details = \
            self.interpreter.get_output_details()

        input_shape = \
            self.input_details[0]["shape"]

        self.input_height = \
            int(input_shape[1])

        self.input_width = \
            int(input_shape[2])

        self.input_channels = \
            int(input_shape[3])

        print("\n[DamageDetector]")

        print(
            f"Model       : "
            f"{self.model_path}"
        )

        print(
            f"Input shape : "
            f"{input_shape}"
        )

        for i, output in enumerate(
            self.output_details
        ):

            print(
                f"Output {i} shape : "
                f"{output['shape']}"
            )

        print()


    # ======================================================
    # 전처리
    # ======================================================

    def preprocess(
        self,
        frame
    ):

        # --------------------------------------------------
        # BGR → RGB
        # --------------------------------------------------

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # 모델 입력 크기로 resize
        # --------------------------------------------------

        image = cv2.resize(
            image,
            (
                self.input_width,
                self.input_height
            )
        )

        # --------------------------------------------------
        # Float32 변환
        # --------------------------------------------------

        image = image.astype(
            np.float32
        )

        # --------------------------------------------------
        # 0 ~ 1 정규화
        # --------------------------------------------------

        image /= 255.0

        # --------------------------------------------------
        # Batch dimension 추가
        # --------------------------------------------------

        image = np.expand_dims(
            image,
            axis=0
        )

        return image


    # ======================================================
    # Softmax
    # ======================================================

    def softmax(
        self,
        values
    ):

        # --------------------------------------------------
        # Overflow 방지를 위해 최댓값을 빼준다.
        # --------------------------------------------------

        values = values - np.max(
            values
        )

        exp_values = np.exp(
            values
        )

        probabilities = (
            exp_values
            /
            np.sum(exp_values)
        )

        return probabilities


    # ======================================================
    # Damage 추론
    # ======================================================

    def predict(
        self,
        frame
    ):

        # --------------------------------------------------
        # 입력 frame 확인
        # --------------------------------------------------

        if frame is None:

            return {
                "damaged": False,
                "confidence": 0.0,
                "class_id": None
            }

        # --------------------------------------------------
        # 전처리
        # --------------------------------------------------

        input_data = self.preprocess(
            frame
        )

        # --------------------------------------------------
        # Input 설정
        # --------------------------------------------------

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_data
        )

        # --------------------------------------------------
        # 추론
        # --------------------------------------------------

        self.interpreter.invoke()

        # --------------------------------------------------
        # Output 가져오기
        # --------------------------------------------------

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        output = np.squeeze(
            output
        )

        # --------------------------------------------------
        # Raw output 확인
        # --------------------------------------------------

        print(
            f"Raw output: {output}"
        )

        print(
            f"Output shape: {output.shape}"
        )

        # ==================================================
        # 2-Class classification
        # ==================================================

        if output.ndim == 1 and len(output) == 2:

            # --------------------------------------------------
            # Logits → Probability
            # --------------------------------------------------

            probabilities = self.softmax(
                output
            )

            # --------------------------------------------------
            # 가장 높은 확률의 class 선택
            # --------------------------------------------------

            class_id = int(
                np.argmax(
                    probabilities
                )
            )

            # --------------------------------------------------
            # 선택된 class의 확률
            # --------------------------------------------------

            confidence = float(
                probabilities[class_id]
            )

        else:

            raise ValueError(
                "지원하지 않는 모델 출력 형태입니다. "
                f"Output shape: {output.shape}"
            )

        # ==================================================
        # Damage 여부
        # ==================================================

        damaged = (

            class_id
            ==
            self.damaged_class_id

            and

            confidence
            >=
            self.confidence_threshold
        )

        # --------------------------------------------------
        # 결과 출력
        # --------------------------------------------------

        print(
            f"Class 0 probability : "
            f"{probabilities[0]:.4f}"
        )

        print(
            f"Class 1 probability : "
            f"{probabilities[1]:.4f}"
        )

        print(
            f"Predicted class     : "
            f"{class_id}"
        )

        print(
            f"Confidence           : "
            f"{confidence:.4f}"
        )

        print(
            f"Damaged              : "
            f"{damaged}"
        )

        return {
            "damaged": damaged,
            "confidence": confidence,
            "class_id": class_id
        }


    # ======================================================
    # __call__
    # ======================================================

    def __call__(
        self,
        frame
    ):

        return self.predict(
            frame
        )


