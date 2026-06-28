# SHAP 계산 전용

import shap
import pickle
import os
import numpy as np

from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'lgbm_model.pkl')

_model_cache = None
_explainer_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, 'rb') as f:
            _model_cache = pickle.load(f)
    return _model_cache


def get_explainer():
    """
    explainer를 프로세스 내에서 1회만 생성하고 캐싱.
    (Django 워커 프로세스가 여러 개면 워커마다 1회씩 생성됨 - 그래도 N회보단 훨씬 나음)
    """
    global _explainer_cache
    if _explainer_cache is None:
        model = load_model()
        _explainer_cache = shap.TreeExplainer(model)
    return _explainer_cache


def calculate_shap_values(model_input, explainer=None):
    """
    model_input: preprocess_for_lgbm() 이후 DataFrame (1행 또는 N행)
    explainer: 외부에서 미리 생성한 explainer를 넘기면 그걸 재사용.
               안 넘기면 캐싱된 전역 explainer를 사용.

    Returns:
        np.ndarray, shape = (n_samples, n_features)
        => "양성 클래스(1)" 기준 SHAP 값으로 통일해서 리턴
    """
    if explainer is None:
        explainer = get_explainer()

    raw_shap = explainer.shap_values(model_input)

    # SHAP 버전/모델 종류에 따라 리턴 형태가 다름 -> 여기서 한 번에 정규화
    if isinstance(raw_shap, list):
        # [array(클래스0, n_samples, n_features), array(클래스1, ...)]
        shap_values = raw_shap[1]  # 양성 클래스(1) 기준
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        # (n_samples, n_features, n_classes)
        shap_values = raw_shap[:, :, 1]
    else:
        # (n_samples, n_features) - 이미 단일 클래스/회귀 형태
        shap_values = raw_shap

    return shap_values