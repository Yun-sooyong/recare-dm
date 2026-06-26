import os
import pickle
import pandas as pd
import csv

from django.conf import settings
from django.shortcuts import render
from .views import build_patient_data_from_request, save_prediction_result_to_db


# =========================
# 모델 로드
# =========================

MODEL_PATH = os.path.join(settings.BASE_DIR, 'lgbm_model.pkl')


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

# =========================
# 위험 등급 함수
# =========================

def get_risk_label(probability):
    if probability >= 0.7:
        return '고위험'
    elif probability >= 0.4:
        return '중위험'
    else:
        return '저위험'


def get_risk_message(risk_label):
    if risk_label == '고위험':
        return '재입원 가능성이 높은 환자입니다. 퇴원 후 추적 관찰과 추가 상담이 필요합니다.'
    elif risk_label == '중위험':
        return '재입원 가능성이 중간 수준입니다. 주요 위험 요인을 확인하고 관리가 필요합니다.'
    else:
        return '현재 입력 정보 기준으로 재입원 위험이 낮은 편입니다. 기본적인 사후 관리는 유지해야 합니다.'


# =========================
# 모델 입력 전처리
# =========================

NUMERIC_COLUMNS = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
]


def age_to_group(age_value):
    try:
        age = int(age_value)
    except Exception:
        return '[60-70)'

    if age < 10:
        return '[0-10)'
    elif age < 20:
        return '[10-20)'
    elif age < 30:
        return '[20-30)'
    elif age < 40:
        return '[30-40)'
    elif age < 50:
        return '[40-50)'
    elif age < 60:
        return '[50-60)'
    elif age < 70:
        return '[60-70)'
    elif age < 80:
        return '[70-80)'
    elif age < 90:
        return '[80-90)'
    else:
        return '[90-100)'


def make_discharge_group(value):
    try:
        value = int(value)
    except Exception:
        return 'Other/Unknown'

    if value == 1:
        return 'Home'
    elif value == 6:
        return 'Home_Health_Service'
    elif value in [2, 3, 4, 5, 10, 22, 23, 24, 27, 28, 29, 30]:
        return 'Transfer/Facility'
    else:
        return 'Other/Unknown'


def preprocess_for_lgbm(df, model):
    df = df.copy()

    if 'discharge_disposition_id' in df.columns:
        df['discharge_group'] = df['discharge_disposition_id'].apply(make_discharge_group)
    else:
        df['discharge_group'] = 'Other/Unknown'

    if 'number_inpatient' in df.columns:
        df['number_inpatient'] = pd.to_numeric(
            df['number_inpatient'],
            errors='coerce'
        ).fillna(0).astype(int)

        df['inpatient_x_discharge'] = (
            df['number_inpatient'].astype(str)
            + '회_'
            + df['discharge_group'].astype(str)
        )
    else:
        df['inpatient_x_discharge'] = '0회_' + df['discharge_group'].astype(str)

    drop_columns = [
        'name',
        'readmit_30d',
        'discharge_disposition_id',
    ]

    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    feature_names = list(model.feature_name_)

    for col in feature_names:
        if col not in df.columns:
            df[col] = None

    df = df[feature_names]

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    categorical_columns = [col for col in feature_names if col not in NUMERIC_COLUMNS]

    if hasattr(model, 'booster_') and model.booster_.pandas_categorical:
        categorical_values = model.booster_.pandas_categorical

        for col, categories in zip(categorical_columns, categorical_values):
            df[col] = df[col].fillna('missing').astype(str)

            if col in ['max_glu_serum', 'A1Cresult']:
                df[col] = df[col].replace({
                    'missing': 'None',
                    'nan': 'None',
                    'None': 'None',
                })

            elif col in ['race', 'diag_1', 'diag_2', 'diag_3']:
                df[col] = df[col].replace({
                    'nan': 'missing',
                    'None': 'missing',
                })

            else:
                df[col] = df[col].replace({
                    'missing': 'No',
                    'nan': 'No',
                })

            df[col] = pd.Categorical(df[col], categories=categories)

    return df

# =========================
# 단일 환자 예측
# =========================

def single_predict(request):
    if request.method == 'POST':
        try:
            data = build_patient_data_from_request(request)

            input_df = pd.DataFrame([data])

            model = load_model()
            model_input = preprocess_for_lgbm(input_df, model)

            probability = model.predict_proba(model_input)[0][1]
            probability_percent = round(probability * 100, 2)
            risk_label = get_risk_label(probability)
            risk_message = get_risk_message(risk_label)

            prediction_data = save_prediction_result_to_db(
                patient_data=data,
                probability_percent=probability_percent,
                risk_label=risk_label,
            )

            input_summary = {
                '환자명': data.get('name'),
                '인종': data.get('race'),
                '성별': '남성' if data.get('gender') == 'Male' else '여성',
                '나이': data.get('raw_age'),
                '연령대': data.get('age'),
                '입원 기간': data.get('time_in_hospital'),
                '검사 횟수': data.get('num_lab_procedures'),
                '시술 횟수': data.get('num_procedures'),
                '투약 수': data.get('num_medications'),
                '최근 외래 방문 횟수': data.get('number_outpatient'),
                '최근 응급 방문 횟수': data.get('number_emergency'),
                '최근 입원 횟수': data.get('number_inpatient'),
                '진단 개수': data.get('number_diagnoses'),
                '인슐린 처방': data.get('insulin'),
                '당뇨약 처방 여부': data.get('diabetesMed'),
                '당뇨약 변경 여부': data.get('change'),
                '메트포르민 처방': data.get('metformin'),
            }

            risk_factors = []

            if data.get('number_inpatient') >= 1:
                risk_factors.append('최근 입원 이력이 있어 재입원 위험에 영향을 줄 수 있습니다.')

            if data.get('number_emergency') >= 1:
                risk_factors.append('최근 응급실 방문 이력이 있어 환자 상태 악화 가능성을 확인해야 합니다.')

            if data.get('time_in_hospital') >= 7:
                risk_factors.append('입원 기간이 긴 편으로, 환자 상태가 복잡했을 가능성이 있습니다.')

            if data.get('num_medications') >= 20:
                risk_factors.append('투약 수가 많아 복합 질환 또는 약물 관리 부담이 있을 수 있습니다.')

            if data.get('number_diagnoses') >= 7:
                risk_factors.append('진단 개수가 많아 질환 관리 난도가 높을 수 있습니다.')

            if data.get('insulin') != 'No':
                risk_factors.append('인슐린 처방 정보가 있어 당뇨 관리 강도와 환자 상태를 함께 확인해야 합니다.')

            if data.get('diabetesMed') == 'Yes':
                risk_factors.append('당뇨약 처방 이력이 있어 약물 복용 관리 여부를 확인해야 합니다.')

            if data.get('change') == 'Ch':
                risk_factors.append('당뇨약 변경 이력이 있어 퇴원 후 약물 적응 상태 확인이 필요합니다.')

            if not risk_factors:
                risk_factors.append('입력된 주요 항목에서 뚜렷한 위험 상승 요인은 크지 않습니다.')

            context = {
                'patient_name': data.get('name'),
                'probability': probability_percent,
                'risk_label': risk_label,
                'risk_message': risk_message,
                'input_summary': input_summary,
                'risk_factors': risk_factors,
                'prediction_data': prediction_data,
            }       

            return render(request, 'prediction/prediction_result.html', context)

        except Exception as e:
            return render(request, 'prediction/prediction_page.html', {
                'error': f'개인 예측 중 오류가 발생했습니다: {e}'
            })

    return render(request, 'prediction/prediction_page.html')


# =========================
# CSV 대량 예측
# =========================

def bulk_predict(request):
    if request.method == 'POST':
        try:
            csv_file = request.FILES.get('csv_file')

            if not csv_file:
                return render(request, 'prediction/prediction_page.html', {
                    'error': 'CSV 파일을 업로드해주세요.'
                })

            df = pd.read_csv(csv_file)
            display_df = df.copy()

            model = load_model()
            model_input = preprocess_for_lgbm(df, model)

            probabilities = model.predict_proba(model_input)[:, 1]

            display_df['재입원_확률'] = (probabilities * 100).round(2)
            display_df['위험등급'] = display_df['재입원_확률'].apply(
                lambda x: '고위험' if x >= 70 else '중위험' if x >= 40 else '저위험'
            )
            for index, row in display_df.iterrows():
                save_prediction_result_to_db(
                    patient_data=df.iloc[index].to_dict(),
                    probability_percent=row['재입원_확률'],
                    risk_label=row['위험등급'],
                )

            important_columns = [
                'name',
                'gender',
                'age',
                'time_in_hospital',
                'num_lab_procedures',
                'num_medications',
                'number_inpatient',
                'number_emergency',
                'number_diagnoses',
                'insulin',
                '재입원_확률',
                '위험등급',
            ]

            result_columns = [
                col for col in important_columns
                if col in display_df.columns
            ]

            result_df = display_df[result_columns]

            result_df = result_df.sort_values(
                by='재입원_확률',
                ascending=False
            )

            rename_columns = {
                'name': '환자명',
                'gender': '성별',
                'age': '연령대',
                'time_in_hospital': '입원 기간',
                'num_lab_procedures': '검사 횟수',
                'num_medications': '투약 수',
                'number_inpatient': '최근 입원 횟수',
                'number_emergency': '응급 방문 횟수',
                'number_diagnoses': '진단 개수',
                'insulin': '인슐린 처방',
                '재입원_확률': '재입원 확률',
                '위험등급': '위험 등급',
            }

            result_df = result_df.rename(columns=rename_columns)

            table_html = result_df.to_html(
                classes='result-table',
                index=False,
                border=0
            )

            context = {
                'table_html': table_html,
                'total_count': len(result_df),
                'high_count': int((result_df['위험 등급'] == '고위험').sum()),
                'middle_count': int((result_df['위험 등급'] == '중위험').sum()),
                'low_count': int((result_df['위험 등급'] == '저위험').sum()),
            }

            return render(request, 'prediction/bulk_prediction_result.html', context)

        except Exception as e:
            return render(request, 'prediction/prediction_page.html', {
                'error': f'CSV 예측 중 오류가 발생했습니다: {e}'
            })

    return render(request, 'prediction/prediction_page.html')