

# services.py
import logging
import pandas as pd
from django.db import transaction

from .shap_storage import save_shap_results
from .shap_utils import get_explainer
from analysis.preprocessing import preprocess_for_lgbm, save_prediction_result_to_db


logger = logging.getLogger(__name__)



# --------------------- single_predict 흐름 ---------------------
def single_predict(input_df, model, data, risk_label):
    

    # 1. 전처리
    model_input = preprocess_for_lgbm(input_df, model)
   

    # 2. 예측
    probability = model.predict_proba(model_input)[0][1]
    

    with transaction.atomic():
        prediction_data = save_prediction_result_to_db(
            patient_data=data,
            probability_percent=probability,
            risk_label=risk_label,
        )

     
        save_shap_results(
            prediction=prediction_data,
            model_input=model_input,
        )

    
    return prediction_data
# -------------------------- bulk_predict 흐름 ---------------------------
def bulk_predict(dataset, model):
    # 1. 전처리 + 예측은 배치로 한 번에 처리
    df = pd.DataFrame(dataset)
    model_input_batch = preprocess_for_lgbm(df, model)
    probabilities = model.predict_proba(model_input_batch)[:, 1]

    explainer = get_explainer()  # 루프 시작 전 1회만 생성

    results = []
    failed_indices = []

    for i, data in enumerate(dataset):
        try:
            with transaction.atomic():
                probability_percent = round(probabilities[i] * 100, 2)

                if probability_percent >= 70:
                    risk_label = "고위험"
                elif probability_percent >= 40:
                    risk_label = "중위험"
                else:
                    risk_label = "저위험"
                
                prediction_data = save_prediction_result_to_db(
                    patient_data=data,
                    probability_percent=probability_percent,
                    risk_label=risk_label,
                )
            
                save_shap_results(
                    prediction=prediction_data,
                    model_input=model_input_batch.iloc[[i]],
                    explainer=explainer,
                )
            results.append(prediction_data)

        except Exception as e:
            logger.error(f"[bulk_predict] index={i} 처리 실패: {e}", exc_info=True)
            failed_indices.append(i)
            continue

    if failed_indices:
        logger.warning(
            f"[bulk_predict] 총 {len(dataset)}건 중 {len(failed_indices)}건 실패 "
            f"(indices={failed_indices})"
        )

    return results