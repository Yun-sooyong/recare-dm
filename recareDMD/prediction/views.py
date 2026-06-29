import os
import pickle
import pandas as pd
import csv
import math
import numpy as np

from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone

from patients.models import PersonalInfo, PredictionData as PatientPrediction
from analysis.services import single_predict as save_prediction_with_shap
from analysis.services import bulk_predict as save_bulk_prediction_with_shap


MODEL_PATH = os.path.join(settings.BASE_DIR, 'lgbm_model.pkl')


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def get_risk_label(probability):
    if probability >= 0.7:
        return '고위험'
    elif probability >= 0.4:
        return '중위험'
    return '저위험'


def get_risk_message(risk_label):
    if risk_label == '고위험':
        return '재입원 가능성이 높은 환자입니다. 퇴원 후 추적 관찰과 추가 상담이 필요합니다.'
    elif risk_label == '중위험':
        return '재입원 가능성이 중간 수준입니다. 주요 위험 요인을 확인하고 관리가 필요합니다.'
    return '현재 입력 정보 기준으로 재입원 위험이 낮은 편입니다. 기본적인 사후 관리는 유지해야 합니다.'


def get_prediction_url_id(prediction):
    if not prediction:
        return None

    for attr in ['prediction_id', 'id', 'pk']:
        value = getattr(prediction, attr, None)
        if value:
            return value

    return None


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
        'patient_nbr',
        'readmit_30d',
        'discharge_disposition_id',
        '재입원_확률',
        '위험등급',
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


def clean_json_value(value):
    if isinstance(value, dict):
        return {key: clean_json_value(val) for key, val in value.items()}

    if isinstance(value, list):
        return [clean_json_value(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if math.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def clean_json_data(data):
    return {key: clean_json_value(value) for key, value in data.items()}


def save_prediction_result_to_db(patient_data, probability_percent, risk_label):
    clean = clean_json_data(patient_data)

    def safe_value(key, default=''):
        value = clean.get(key)
        if value is None or value == '':
            return default
        return value

    def safe_int(key, default=0):
        value = clean.get(key)
        if value is None or value == '':
            return default
        try:
            return int(value)
        except Exception:
            return default

    nbr = clean.get('patient_nbr')

    if nbr:
        patient, _ = PersonalInfo.objects.get_or_create(patient_nbr=int(nbr))
    else:
        max_nbr = (
            PersonalInfo.objects
            .order_by('-patient_nbr')
            .values_list('patient_nbr', flat=True)
            .first()
            or 10000
        )
        patient = PersonalInfo.objects.create(patient_nbr=max_nbr + 1)

    max_eid = (
        PatientPrediction.objects
        .order_by('-encounter_id')
        .values_list('encounter_id', flat=True)
        .first()
        or 0
    )

    prediction_data = PatientPrediction.objects.create(
        patient=patient,
        encounter_id=max_eid + 1,
        visit_date=timezone.now().date(),
        is_inpatient=True,
        race=safe_value('race', 'Unknown'),
        gender=safe_value('gender', 'Unknown'),
        age=safe_value('age', safe_value('raw_age', 'Unknown')),
        time_in_hospital=safe_int('time_in_hospital'),
        num_lab_procedures=safe_int('num_lab_procedures'),
        num_procedures=safe_int('num_procedures'),
        num_medications=safe_int('num_medications'),
        number_outpatient=safe_int('number_outpatient'),
        number_emergency=safe_int('number_emergency'),
        number_inpatient=safe_int('number_inpatient'),
        number_diagnoses=safe_int('number_diagnoses'),
        diag_1=safe_value('diag_1', 'Unknown'),
        diag_2=safe_value('diag_2', 'Unknown'),
        diag_3=safe_value('diag_3', 'Unknown'),
        max_glu_serum=safe_value('max_glu_serum', 'None'),
        A1Cresult=safe_value('A1Cresult', 'None'),
        discharge_disposition_id=safe_int('discharge_disposition_id', 1),
        metformin=safe_value('metformin', 'No'),
        insulin=safe_value('insulin', 'No'),
        change=safe_value('change', 'No'),
        diabetesMed=safe_value('diabetesMed', 'No'),
        readmit_proba=float(probability_percent),
        risk_level=risk_label,
        predicted_at=timezone.now(),
    )

    return prediction_data


def prediction_page(request):
    patient_cart = request.session.get('patient_cart', [])

    return render(request, 'prediction/prediction_page.html', {
        'patient_cart': patient_cart,
        'cart_count': len(patient_cart),
    })


def build_patient_data_from_request(request):
    patient_name = request.POST.get('patient_name') or '이름 없음'
    raw_age = request.POST.get('age') or '65'
    gender = request.POST.get('gender') or 'Female'
    race = request.POST.get('race') or 'Caucasian'
    insulin = request.POST.get('insulin') or 'No'
    diabetes_med = request.POST.get('diabetesMed') or 'Yes'
    change = request.POST.get('change') or 'No'
    metformin = request.POST.get('metformin') or 'No'

    return {
        'name': patient_name,
        'race': race,
        'gender': gender,
        'age': age_to_group(raw_age),
        'raw_age': raw_age,
        'discharge_disposition_id': 1,
        'time_in_hospital': int(request.POST.get('time_in_hospital') or 0),
        'num_lab_procedures': int(request.POST.get('num_lab_procedures') or 0),
        'num_procedures': int(request.POST.get('num_procedures') or 0),
        'num_medications': int(request.POST.get('num_medications') or 0),
        'number_outpatient': int(request.POST.get('number_outpatient') or 0),
        'number_emergency': int(request.POST.get('number_emergency') or 0),
        'number_inpatient': int(request.POST.get('number_inpatient') or 0),
        'diag_1': 'Circulatory',
        'diag_2': 'Diabetes',
        'diag_3': 'Other',
        'number_diagnoses': int(request.POST.get('number_diagnoses') or 0),
        'max_glu_serum': 'None',
        'A1Cresult': 'None',
        'metformin': metformin,
        'repaglinide': 'No',
        'nateglinide': 'No',
        'chlorpropamide': 'No',
        'glimepiride': 'No',
        'acetohexamide': 'No',
        'glipizide': 'No',
        'glyburide': 'No',
        'tolbutamide': 'No',
        'pioglitazone': 'No',
        'rosiglitazone': 'No',
        'acarbose': 'No',
        'miglitol': 'No',
        'troglitazone': 'No',
        'tolazamide': 'No',
        'insulin': insulin,
        'glyburide-metformin': 'No',
        'glipizide-metformin': 'No',
        'glimepiride-pioglitazone': 'No',
        'metformin-rosiglitazone': 'No',
        'metformin-pioglitazone': 'No',
        'change': change,
        'diabetesMed': diabetes_med,
    }


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

            save_prediction_result_to_db(
                patient_data=data,
                probability_percent=probability_percent,
                risk_label=risk_label,
            )

            prediction_data = save_prediction_with_shap(
                input_df=input_df,
                model=model,
                data=data,
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

            return render(request, 'prediction/prediction_result.html', {
                'patient_name': data.get('name'),
                'probability': probability_percent,
                'risk_label': risk_label,
                'risk_message': risk_message,
                'input_summary': input_summary,
                'risk_factors': risk_factors,
                'prediction_data': prediction_data,
                'prediction_url_id': get_prediction_url_id(prediction_data),
            })

        except Exception as e:
            return render(request, 'prediction/prediction_page.html', {
                'error': f'개인 예측 중 오류가 발생했습니다: {e}'
            })

    return render(request, 'prediction/prediction_page.html')


def make_bulk_result_context(display_df, first_prediction_id=None):
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

    result_df = display_df[result_columns].sort_values(
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

    return {
        'table_html': table_html,
        'total_count': len(result_df),
        'high_count': int((result_df['위험 등급'] == '고위험').sum()),
        'middle_count': int((result_df['위험 등급'] == '중위험').sum()),
        'low_count': int((result_df['위험 등급'] == '저위험').sum()),
        'first_prediction_id': first_prediction_id,
    }


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
                row_data = row.to_dict()

                save_prediction_result_to_db(
                    patient_data=row_data,
                    probability_percent=row_data.get('재입원_확률', 0),
                    risk_label=row_data.get('위험등급', '저위험'),
                )

            saved_shap_predictions = save_bulk_prediction_with_shap(
                dataset=df.to_dict(orient='records'),
                model=model,
            )

            first_prediction_id = None

            if saved_shap_predictions:
                first_prediction_id = get_prediction_url_id(saved_shap_predictions[0])

            context = make_bulk_result_context(
                display_df=display_df,
                first_prediction_id=first_prediction_id,
            )

            return render(request, 'prediction/bulk_prediction_result.html', context)

        except Exception as e:
            return render(request, 'prediction/prediction_page.html', {
                'error': f'CSV 예측 중 오류가 발생했습니다: {e}'
            })

    return render(request, 'prediction/prediction_page.html')


def download_sample_csv(request):
    response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
    response['Content-Disposition'] = 'attachment; filename="sample_prediction_input.csv"'

    writer = csv.writer(response)

    columns = [
        'patient_nbr', 'name', 'race', 'gender', 'age', 'discharge_disposition_id',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
        'number_diagnoses', 'max_glu_serum', 'A1Cresult',
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change', 'diabetesMed', 'readmit_30d'
    ]

    sample_rows = [
        [
            10001, '김도윤', 'Caucasian', 'Male', '[30-40)', 1,
            4, 40, 1, 13, 0, 0, 5,
            'Circulatory', 'Diabetes', 'Other', 6,
            'None', 'None',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'Down',
            'No', 'No', 'No', 'No',
            'No', 'Ch', 'Yes', 0
        ],
        [
            10002, '정유진', 'Asian', 'Female', '[50-60)', 1,
            9, 62, 2, 26, 0, 1, 3,
            'Diabetes', 'Circulatory', 'Respiratory', 8,
            'None', '>8',
            'Steady', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'Ch', 'Yes', 1
        ],
    ]

    writer.writerow(columns)
    writer.writerows(sample_rows)

    return response


def download_sample_pdf(request):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
    except ImportError:
        return HttpResponse('PDF 생성을 위해 reportlab 패키지를 설치해주세요: pip install reportlab')

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="sample_csv_guide.pdf"'

    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    y = height - 50

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, "CSV Upload Guide")
    y -= 35

    p.setFont("Helvetica", 10)
    lines = [
        "Required CSV columns:",
        "patient_nbr, name, race, gender, age, discharge_disposition_id,",
        "time_in_hospital, num_lab_procedures, num_procedures, num_medications,",
        "number_outpatient, number_emergency, number_inpatient,",
        "diag_1, diag_2, diag_3, number_diagnoses,",
        "max_glu_serum, A1Cresult, metformin, insulin, change, diabetesMed",
        "",
        "Note:",
        "patient_nbr is used to connect prediction results to existing patients.",
        "readmit_30d can exist in the CSV, but it will be excluded from prediction input."
    ]

    for line in lines:
        p.drawString(50, y, line)
        y -= 18

    p.showPage()
    p.save()

    return response


def add_patient_cart(request):
    if request.method == 'POST':
        try:
            patient_data = build_patient_data_from_request(request)

            patient_cart = request.session.get('patient_cart', [])
            patient_cart.append(patient_data)

            request.session['patient_cart'] = patient_cart
            request.session.modified = True

            return redirect('prediction:prediction_page')

        except Exception as e:
            return render(request, 'prediction/prediction_page.html', {
                'error': f'환자 추가 중 오류가 발생했습니다: {e}',
                'patient_cart': request.session.get('patient_cart', []),
                'cart_count': len(request.session.get('patient_cart', [])),
            })

    return redirect('prediction:prediction_page')


def clear_patient_cart(request):
    request.session['patient_cart'] = []
    request.session.modified = True

    return redirect('prediction:prediction_page')


def predict_patient_cart(request):
    try:
        patient_cart = request.session.get('patient_cart', [])

        if not patient_cart:
            return render(request, 'prediction/prediction_page.html', {
                'error': '예측할 환자가 없습니다. 먼저 환자를 추가해주세요.',
                'patient_cart': [],
                'cart_count': 0,
            })

        df = pd.DataFrame(patient_cart)
        display_df = df.copy()

        model = load_model()
        model_input = preprocess_for_lgbm(df, model)

        probabilities = model.predict_proba(model_input)[:, 1]

        display_df['재입원_확률'] = (probabilities * 100).round(2)
        display_df['위험등급'] = display_df['재입원_확률'].apply(
            lambda x: '고위험' if x >= 70 else '중위험' if x >= 40 else '저위험'
        )

        for index, row in display_df.iterrows():
            row_data = row.to_dict()

            save_prediction_result_to_db(
                patient_data=row_data,
                probability_percent=row_data.get('재입원_확률', 0),
                risk_label=row_data.get('위험등급', '저위험'),
            )

        saved_shap_predictions = save_bulk_prediction_with_shap(
            dataset=df.to_dict(orient='records'),
            model=model,
        )

        first_prediction_id = None

        if saved_shap_predictions:
            first_prediction_id = get_prediction_url_id(saved_shap_predictions[0])

        context = make_bulk_result_context(
            display_df=display_df,
            first_prediction_id=first_prediction_id,
        )

        request.session['patient_cart'] = []
        request.session.modified = True

        return render(request, 'prediction/bulk_prediction_result.html', context)

    except Exception as e:
        return render(request, 'prediction/prediction_page.html', {
            'error': f'장바구니 예측 중 오류가 발생했습니다: {e}',
            'patient_cart': request.session.get('patient_cart', []),
            'cart_count': len(request.session.get('patient_cart', [])),
        })