# DB 저장 전용
from .models import ShapResult
from .shap_utils import calculate_shap_values


def save_shap_results(prediction, model_input, explainer=None):
    """
    SHAP 값을 계산해서 DB(ShapResult)에 저장하는 함수

    Args:
        prediction: PredictionData 인스턴스 (DB에 저장된 예측 결과)
        model_input: 전처리 완료된 DataFrame (1행 기준)
        explainer: (선택) bulk 처리 시 재사용할 explainer
    """
    shap_values = calculate_shap_values(model_input, explainer=explainer)

    # model_input이 1행이라는 전제 -> 첫 번째 row만 사용
    row_shap = shap_values[0]

    shap_objects = [
        ShapResult(
            prediction=prediction,
            feature_name=feature_name,
            shap_value=float(shap_value),
        )
        for feature_name, shap_value in zip(model_input.columns, row_shap)
    ]
    ShapResult.objects.bulk_create(shap_objects)