import pandas as pd

from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from patients.models import PersonalInfo, PredictionData

from analysis.preprocessing import (
    age_to_group,
    get_risk_label,
    get_risk_message,
    load_model,
    preprocess_for_lgbm,
)

from .forms import DischargeRecordForm
from .models import DischargeRecord


# 예측 데이터로부터 퇴원 준비도 5개 항목 점수 자동 계산 (각 20점, 총 100점)
def compute_readiness(prediction):
    if not prediction:
        return None

    # 1) 재입원 위험도: 확률이 낮을수록 높은 점수
    proba = prediction.readmit_proba or 0
    if proba < 30:
        risk_score = 20
    elif proba < 50:
        risk_score = 14
    elif proba < 70:
        risk_score = 8
    else:
        risk_score = 4

    # 2) 이전 입원 이력
    inp = prediction.number_inpatient or 0
    if inp == 0:
        inp_score = 20
    elif inp <= 1:
        inp_score = 14
    elif inp <= 3:
        inp_score = 8
    else:
        inp_score = 4

    # 3) 응급실 방문 이력
    er = prediction.number_emergency or 0
    if er == 0:
        er_score = 20
    elif er <= 1:
        er_score = 14
    elif er <= 3:
        er_score = 8
    else:
        er_score = 4

    # 4) 약물 안정성: 처방 변경 없으면 안정
    med_score = 10 if prediction.change == 'Ch' else 20

    # 5) 퇴원 경로 적절성
    disp = prediction.discharge_disposition_id or 1
    if disp == 1:
        path_score = 20
    elif disp == 6:
        path_score = 16
    else:
        path_score = 10

    items = [
        {
            'icon': '🔄', 'name': '재입원 위험도',
            'score': risk_score,
            'desc': f'위험도 {proba:.1f}% → {"집중 추적 관찰 필요" if proba >= 50 else "양호"}',
        },
        {
            'icon': '🏥', 'name': '이전 입원 이력',
            'score': inp_score,
            'desc': f'이전 입원 {inp}회 → {"지속 관리 프로그램 연계" if inp >= 2 else "양호"}',
        },
        {
            'icon': '⚡', 'name': '응급실 방문 이력',
            'score': er_score,
            'desc': f'응급실 방문 {er}회 → {"응급 상황 대응 계획 필요" if er >= 2 else "양호"}',
        },
        {
            'icon': '💊', 'name': '약물 안정성',
            'score': med_score,
            'desc': f'처방 약물 변경 {"있음 → 복약 적응 확인 필요" if prediction.change == "Ch" else "없음 → 안정"}',
        },
        {
            'icon': '🏠', 'name': '퇴원 경로 적절성',
            'score': path_score,
            'desc': '자택 귀가 가능 → 독립적 퇴원' if disp == 1 else '가정간호 연계' if disp == 6 else '시설 이송 필요',
        },
    ]

    total = sum(i['score'] for i in items)

    if total >= 80:
        label = '양호'
    elif total >= 60:
        label = '주의 필요'
    else:
        label = '집중 지원 필요'

    # 위험 요인 TOP 3: 점수가 낮은 순으로 정렬
    risk_factors = sorted(
        [i for i in items if i['score'] < 20],
        key=lambda x: x['score'],
    )[:3]
    for rf in risk_factors:
        rf['impact'] = round((20 - rf['score']) / 20, 3)
        rf['bar_pct'] = int(rf['impact'] * 100)

    # 게이지 SVG용 dasharray 값 (반원 둘레 ≈ 220px)
    gauge_fill = int(total * 220 / 100)

    return {
        'items': items, 'total': total, 'label': label,
        'risk_factors': risk_factors, 'gauge_fill': gauge_fill,
    }


# 퇴원 안내서: 퇴원 준비도 점수 + 체크리스트
def discharge_guide(request, patient_id):
    patient = get_object_or_404(PersonalInfo, pk=patient_id)
    latest_prediction = patient.predictions.order_by('-visit_date', '-encounter_id').first()
    record = DischargeRecord.objects.filter(uid=patient).first()
    readiness = compute_readiness(latest_prediction)

    if request.method == 'POST':
        form = DischargeRecordForm(request.POST, instance=record)
        if form.is_valid():
            saved = form.save(commit=False)
            saved.uid = patient
            if readiness:
                saved.discharge_score = readiness['total']
            saved.save()
            return redirect('discharge:discharge_guide', patient_id=patient.pk)
    else:
        form = DischargeRecordForm(instance=record)

    context = {
        'patient': patient,
        'prediction': latest_prediction,
        'record': record,
        'form': form,
        'readiness': readiness,
    }
    return render(request, 'discharge/discharge_guide.html', context)


# 퇴원 안내서 인쇄용 페이지
def discharge_print(request, patient_id):
    patient = get_object_or_404(PersonalInfo, pk=patient_id)
    record = DischargeRecord.objects.filter(uid=patient).first()
    latest_prediction = patient.predictions.order_by('-visit_date', '-encounter_id').first()
    readiness = compute_readiness(latest_prediction)

    context = {
        'patient': patient,
        'record': record,
        'prediction': latest_prediction,
        'readiness': readiness,
    }
    return render(request, 'discharge/discharge_print.html', context)


# 재방문 등록: 이전 방문 데이터를 불러와 수정 후 재예측
def revisit(request, patient_id):
    patient = get_object_or_404(PersonalInfo, pk=patient_id)
    # 가장 최근 방문 기록을 가져와서 폼 초기값으로 사용
    prev = patient.predictions.order_by('-visit_date', '-encounter_id').first()

    if request.method == 'POST':
        # 폼에서 입력된 새 방문 데이터 수집
        raw_age = request.POST.get('age', '')
        data = {
            'race': request.POST.get('race', ''),
            'gender': request.POST.get('gender', ''),
            'age': age_to_group(raw_age),
            'discharge_disposition_id': int(request.POST.get('discharge_disposition_id', 1)),
            'time_in_hospital': int(request.POST.get('time_in_hospital', 0)),
            'num_lab_procedures': int(request.POST.get('num_lab_procedures', 0)),
            'num_procedures': int(request.POST.get('num_procedures', 0)),
            'num_medications': int(request.POST.get('num_medications', 0)),
            'number_outpatient': int(request.POST.get('number_outpatient', 0)),
            'number_emergency': int(request.POST.get('number_emergency', 0)),
            'number_inpatient': int(request.POST.get('number_inpatient', 0)),
            'number_diagnoses': int(request.POST.get('number_diagnoses', 0)),
            'diag_1': request.POST.get('diag_1', ''),
            'diag_2': request.POST.get('diag_2', ''),
            'diag_3': request.POST.get('diag_3', ''),
            'max_glu_serum': request.POST.get('max_glu_serum', 'None'),
            'A1Cresult': request.POST.get('A1Cresult', 'None'),
            'metformin': request.POST.get('metformin', 'No'),
            'repaglinide': request.POST.get('repaglinide', 'No'),
            'nateglinide': request.POST.get('nateglinide', 'No'),
            'chlorpropamide': request.POST.get('chlorpropamide', 'No'),
            'glimepiride': request.POST.get('glimepiride', 'No'),
            'acetohexamide': request.POST.get('acetohexamide', 'No'),
            'glipizide': request.POST.get('glipizide', 'No'),
            'glyburide': request.POST.get('glyburide', 'No'),
            'tolbutamide': request.POST.get('tolbutamide', 'No'),
            'pioglitazone': request.POST.get('pioglitazone', 'No'),
            'rosiglitazone': request.POST.get('rosiglitazone', 'No'),
            'acarbose': request.POST.get('acarbose', 'No'),
            'miglitol': request.POST.get('miglitol', 'No'),
            'troglitazone': request.POST.get('troglitazone', 'No'),
            'tolazamide': request.POST.get('tolazamide', 'No'),
            'insulin': request.POST.get('insulin', 'No'),
            'glyburide-metformin': request.POST.get('glyburide_metformin', 'No'),
            'glipizide-metformin': request.POST.get('glipizide_metformin', 'No'),
            'glimepiride-pioglitazone': request.POST.get('glimepiride_pioglitazone', 'No'),
            'metformin-rosiglitazone': request.POST.get('metformin_rosiglitazone', 'No'),
            'metformin-pioglitazone': request.POST.get('metformin_pioglitazone', 'No'),
            'change': request.POST.get('change', 'No'),
            'diabetesMed': request.POST.get('diabetesMed', 'No'),
        }

        # ML 모델 로드 및 예측 실행
        model = load_model()
        input_df = pd.DataFrame([data])
        model_input = preprocess_for_lgbm(input_df, model)
        probability = model.predict_proba(model_input)[0][1]
        probability_percent = round(probability * 100, 2)
        risk_label = get_risk_label(probability)
        risk_message = get_risk_message(risk_label)

        # 새 방문 기록을 DB에 저장
        new_encounter_id = (PredictionData.objects.order_by('-encounter_id')
                           .values_list('encounter_id', flat=True).first() or 0) + 1
        prediction = PredictionData.objects.create(
            patient=patient,
            encounter_id=new_encounter_id,
            visit_date=timezone.now().date(),
            is_inpatient=True,
            race=data['race'],
            gender=data['gender'],
            age=data['age'],
            time_in_hospital=data['time_in_hospital'],
            num_lab_procedures=data['num_lab_procedures'],
            num_procedures=data['num_procedures'],
            num_medications=data['num_medications'],
            number_outpatient=data['number_outpatient'],
            number_emergency=data['number_emergency'],
            number_inpatient=data['number_inpatient'],
            number_diagnoses=data['number_diagnoses'],
            diag_1=data['diag_1'],
            diag_2=data['diag_2'],
            diag_3=data['diag_3'],
            max_glu_serum=data['max_glu_serum'],
            A1Cresult=data['A1Cresult'],
            metformin=data['metformin'],
            insulin=data['insulin'],
            change=data['change'],
            diabetesMed=data['diabetesMed'],
            discharge_disposition_id=data['discharge_disposition_id'],
            readmit_proba=probability_percent,
            risk_level=risk_label,
            predicted_at=timezone.now(),
        )

        # 예측 결과 페이지로 렌더링
        context = {
            'patient': patient,
            'prediction': prediction,
            'probability': probability_percent,
            'risk_label': risk_label,
            'risk_message': risk_message,
        }
        return render(request, 'discharge/revisit_result.html', context)

    # GET: 이전 방문 데이터를 초기값으로 전달
    context = {
        'patient': patient,
        'prev': prev,
    }
    return render(request, 'discharge/revisit.html', context)
