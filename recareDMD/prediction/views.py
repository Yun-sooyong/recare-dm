import os
import pickle
import pandas as pd
import csv
import math
import numpy as np
from django.shortcuts import render

from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse

from django.utils import timezone
from patients.models import PersonalInfo, PredictionData as PatientPrediction
from .models import PredictionData
from analysis.services import single_predict as save_prediction_with_shap
from analysis.services import bulk_predict as save_bulk_prediction_with_shap
from analysis.preprocessing import (
    preprocess_for_lgbm,
    age_to_group,
    get_risk_label,
    get_risk_message,
    load_model,
    save_prediction_result_to_db,
    clean_json_data,
)


MODEL_PATH = os.path.join(settings.BASE_DIR, 'lgbm_model.pkl')




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

    age_group = age_to_group(raw_age)

    data = {
        'name': patient_name,
        'race': race,
        'gender': gender,
        'age': age_group,
        'raw_age': raw_age,
        'discharge_disposition_id': 1,

        'time_in_hospital': int(request.POST.get('time_in_hospital')),
        'num_lab_procedures': int(request.POST.get('num_lab_procedures')),
        'num_procedures': int(request.POST.get('num_procedures')),
        'num_medications': int(request.POST.get('num_medications')),
        'number_outpatient': int(request.POST.get('number_outpatient')),
        'number_emergency': int(request.POST.get('number_emergency')),
        'number_inpatient': int(request.POST.get('number_inpatient')),

        'diag_1': 'Circulatory',
        'diag_2': 'Diabetes',
        'diag_3': 'Other',
        'number_diagnoses': int(request.POST.get('number_diagnoses')),

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

    return data

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
            dataset = df.to_dict(orient='records')

            save_bulk_prediction_with_shap(
                dataset=dataset,
                model=model,
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

#=========================
# csv 샘플 다운로드 
#=========================
def download_sample_csv(request):
    response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
    response['Content-Disposition'] = 'attanchment; filename="sample_prediction_input.csv'
    
    writer = csv.writer(response)
    
    columns = [
        'name', 'race', 'gender', 'age', 'discharge_disposition_id',
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
            '곽춘봉', 'Caucasian', 'Male', '[60-70)', 1,
            7, 45, 1, 20, 2, 0, 1,
            'Circulatory', 'Diabetes', 'Other', 6,
            'None', 'None',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'Steady',
            'No', 'No', 'No', 'No',
            'No', 'Ch', 'Yes', 0
        ],
        [
            '김미자', 'Asian', 'Female', '[70-80)', 6,
            4, 32, 0, 15, 0, 1, 2,
            'Diabetes', 'Circulatory', 'Respiratory', 8,
            'None', '>8',
            'Steady', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'Up',
            'No', 'No', 'No', 'No',
            'No', 'Ch', 'Yes', 1
        ],
    ]

    writer.writerow(columns)
    writer.writerows(sample_rows)

    return response

#=========================
# pdf 샘플 다운로드 
#=========================
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
        "name, race, gender, age, discharge_disposition_id,",
        "time_in_hospital, num_lab_procedures, num_procedures, num_medications,",
        "number_outpatient, number_emergency, number_inpatient,",
        "diag_1, diag_2, diag_3, number_diagnoses,",
        "max_glu_serum, A1Cresult, metformin, insulin, change, diabetesMed",
        "",
        "Example row:",
        "Kim, Asian, Female, [60-70), 1, 7, 45, 1, 20, 2, 0, 1,",
        "Circulatory, Diabetes, Other, 6, None, None, No, Steady, Ch, Yes",
        "",
        "Note:",
        "readmit_30d can exist in the CSV, but it will be excluded from prediction input."
    ]

    for line in lines:
        p.drawString(50, y, line)
        y -= 18

    p.showPage()
    p.save()

    return response



#=========================
# 환자 장바구니 기능
#=========================
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

        dataset = df.to_dict(orient="records")

        save_bulk_prediction_with_shap(
            dataset=dataset,
            model=model,
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

        request.session['patient_cart'] = []
        request.session.modified = True

        return render(request, 'prediction/bulk_prediction_result.html', context)

    except Exception as e:
        return render(request, 'prediction/prediction_page.html', {
            'error': f'장바구니 예측 중 오류가 발생했습니다: {e}',
            'patient_cart': request.session.get('patient_cart', []),
            'cart_count': len(request.session.get('patient_cart', [])),
        })
        


