import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class DamageDetector:
    def __init__(
        self,
        model_path="damage_detect_v2_float32.tflite",
        damaged_class_id=0,
        damage_threshold=0.5,
    ):
        self.model_path = model_path
        self.damaged_class_id = damaged_class_id
        self.damage_threshold = damage_threshold

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 모델의 인풋아웃풋 디테일들 확인 후 터미널의 출력
        shape = self.input_details[0]["shape"]
        self.input_height = int(shape[1])
        self.input_width = int(shape[2])

        print(f"Model             : {model_path}")
        print(f"Input shape       : {shape}")
        print(f"Damaged class ID  : {damaged_class_id}")
        print(f"Damage threshold  : {damage_threshold:.2f}")
        for i, output in enumerate(self.output_details):
            print(f"Output {i} shape  : {output['shape']}")
        print()

    #이미지 색,사이즈,데이터타입을 모델에 맞게 변환
    def preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    #훼손여부 판별 모델의 결과가 훼손가능성,노말가능성(logit)으로 출력되는데 이를 소프트맥스함수를 통해 확률값으로 변환
    @staticmethod
    def softmax(values):
        values = values - np.max(values)
        exp_values = np.exp(values)
        return exp_values / np.sum(exp_values)

    #훼손판별모델 객체를 생성해 추론
    def predict(self, frame):
        if frame is None:
            return {
                "damaged": False,
                "confidence": 0.0,
                "damage_score": 0.0,
                "normal_score": 0.0,
                "class_id": None,
            }

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            self.preprocess(frame),
        )
        self.interpreter.invoke()

        output = np.squeeze(
            self.interpreter.get_tensor(self.output_details[0]["index"])
        )

        if output.ndim != 1 or len(output) != 2:
            raise ValueError(
                f"지원하지 않는 모델 출력 형태입니다. Output shape: {output.shape}"
            )

        #결과의 첫번째 값은 훼손가능성, 두번째 값은 노말가능성(logit)으로 출력
        probabilities = self.softmax(output)
        damage_score = float(probabilities[self.damaged_class_id])
        normal_score = float(probabilities[1])

        #훼손가능성 점수가 damage_threshold 이상이면 훼손으로 판단, 아니면 노말로 판단
        damaged = damage_score >= self.damage_threshold
        class_id = self.damaged_class_id if damaged else 1
        confidence = damage_score if damaged else normal_score

        print(
            f"Damage={damage_score:.4f}, "
            f"Normal={normal_score:.4f}, "
            f"Class={class_id}, "
            f"Confidence={confidence:.4f}"
        )

        return {
            "damaged": damaged,
            "confidence": confidence,
            "damage_score": damage_score,
            "normal_score": normal_score,
            "class_id": class_id,
        }

    def __call__(self, frame):
        return self.predict(frame)
