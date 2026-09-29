# damage_detect_module.py

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class DamageDetector:

    def __init__(
        self,
        model_path="damage_detect_v2_float32.tflite",
        damaged_class_id=0,
        damage_threshold=0.5
    ):

        self.model_path = model_path

        # 현재 모델에서
        # class 0 = damaged
        self.damaged_class_id = damaged_class_id

        # class 0의 확률이 이 값 이상이면
        # DAMAGE로 판정
        self.damage_threshold = damage_threshold

        # ==================================================
        # TFLite Interpreter
        # ==================================================

        self.interpreter = tflite.Interpreter(
            model_path=self.model_path
        )

        self.interpreter.allocate_tensors()

        # ==================================================
        # Input / Output 정보
        # ==================================================

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


        print(
            f"Model            : "
            f"{self.model_path}"
        )

        print(
            f"Input shape      : "
            f"{input_shape}"
        )

        print(
            f"Damaged class ID  : "
            f"{self.damaged_class_id}"
        )

        print(
            f"Damage threshold  : "
            f"{self.damage_threshold:.2f}"
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
        # Overflow 방지
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
                "damage_score": 0.0,
                "normal_score": 0.0,
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
        # Raw output 출력
        # --------------------------------------------------

        print(
            f"Raw output : {output}"
        )

        print(
            f"Output shape : {output.shape}"
        )


        # ==================================================
        # 2-Class classification
        # ==================================================

        if (
            output.ndim == 1
            and
            len(output) == 2
        ):

            # --------------------------------------------------
            # Raw logits → probability
            # --------------------------------------------------

            probabilities = self.softmax(
                output
            )

        else:

            raise ValueError(
                "지원하지 않는 모델 출력 형태입니다. "
                f"Output shape: {output.shape}"
            )


        # ==================================================
        # 각 클래스 확률
        # ==================================================

        damage_score = float(
            probabilities[
                self.damaged_class_id
            ]
        )


        # 현재 모델은
        # class 0 = damage
        # class 1 = normal
        #
        # 따라서 class 1의 점수는 다음과 같이 가져온다.

        normal_class_id = 1

        normal_score = float(
            probabilities[
                normal_class_id
            ]
        )


        # ==================================================
        # Damage Threshold 판정
        # ==================================================

        # 중요:
        #
        # 기존:
        #   가장 높은 class를 선택
        #
        # 변경:
        #   class 0의 점수가 threshold 이상이면
        #   무조건 class 0 = DAMAGE
        #
        #   threshold 미만이면
        #   class 1 = NORMAL

        if damage_score >= self.damage_threshold:

            class_id = self.damaged_class_id

            damaged = True

            confidence = damage_score

        else:

            class_id = normal_class_id

            damaged = False

            confidence = normal_score


        # ==================================================
        # 결과 출력
        # ==================================================

        print(
            f"Class 0 probability : "
            f"{probabilities[0]:.4f}"
        )

        print(
            f"Class 1 probability : "
            f"{probabilities[1]:.4f}"
        )

        print(
            f"Damage threshold    : "
            f"{self.damage_threshold:.4f}"
        )

        print(
            f"Damage score        : "
            f"{damage_score:.4f}"
        )

        print(
            f"Normal score        : "
            f"{normal_score:.4f}"
        )

        print(
            f"Predicted class     : "
            f"{class_id}"
        )

        print(
            f"Confidence          : "
            f"{confidence:.4f}"
        )

        print(
            f"Damaged             : "
            f"{damaged}"
        )


        # ==================================================
        # 결과 반환
        # ==================================================

        return {

            # 최종 DAMAGE 여부
            "damaged":
                damaged,

            # 최종 선택된 클래스의 점수
            "confidence":
                confidence,

            # 중요:
            # class 0의 실제 점수
            "damage_score":
                damage_score,

            # class 1의 점수
            "normal_score":
                normal_score,

            # 최종 판정 class
            "class_id":
                class_id
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