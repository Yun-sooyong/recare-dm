from django.shortcuts import render, get_object_or_404
from django.db.models import F
from django.db.models.functions import Abs

from patients.models import PredictionData


def shap_result(request, prediction_id):
    prediction = get_object_or_404(PredictionData, visit_id=prediction_id)

    # 위험 요인: shap_value > 0, 큰 값부터 상위 5개
    risk_factors = list(
        prediction.shap_results
        .filter(shap_value__gt=0)
        .order_by("-shap_value")[:5]
    )

    # 보호 요인: shap_value < 0, 가장 음수인 값부터 상위 5개
    # abs_value를 같이 계산해서 템플릿에서 |abs 필터 없이 바로 쓸 수 있게 함
    protective_factors = list(
        prediction.shap_results
        .filter(shap_value__lt=0)
        .annotate(abs_value=Abs("shap_value"))
        .order_by("shap_value")[:5]
    )

    # 전체 41개 (표 출력용) - 절대값 기준 정렬
    all_results = (
        prediction.shap_results
        .annotate(abs_shap=Abs("shap_value"))
        .order_by("-abs_shap")
    )

    context = {
        "prediction": prediction,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "all_results": all_results,
    }

    return render(request, "analysis/shap_result.html", context)