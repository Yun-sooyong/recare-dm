from analysis.models import ShapResult
from analysis.preprocessing import (
    encounter_to_model_input,
    get_risk_label,
    group_by_icd9,
    load_model,
)
from django.core.paginator import Paginator
from django.db.models import Count, Max, OuterRef, Q, Subquery
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from patients.models import MEDICATION_FIELDS, PersonalInfo, PredictionData

from .forms import PatientForm, PredictionDataEditForm

PAGE_SIZE = 20


def patient_list(request):
    q = request.GET.get("q", "").strip()

    latest = PredictionData.objects.filter(patient=OuterRef("pk")).order_by(
        "-visit_date", "-encounter_id"
    )
    patients = PersonalInfo.objects.annotate(
        n_visits=Count("predictions", distinct=True),
        last_visit=Max("predictions__visit_date"),
        latest_age=Subquery(latest.values("age")[:1]),
        latest_gender=Subquery(latest.values("gender")[:1]),
        risk_level=Subquery(latest.values('risk_level')[:1]),
        # latest_name=Subquery(latest.values('patient_name')[:1]),
    ).order_by("patient_nbr")

    if q:
        patients = patients.filter(Q(patient_nbr__icontains=q) | Q(patient_name__icontains=q))
        

    paginator = Paginator(patients, PAGE_SIZE)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(
        request,
        "patients/patient_list.html",
        {"page_obj": page_obj, "q": q, "total": paginator.count},
    )


def patient_create(request):
    form = PatientForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        return redirect("patients:patient_list")
    return render(
        request,
        "patients/patient_form.html",
        {"form": form, "mode": "create"},
    )


def patient_update(request, pk):
    patient = get_object_or_404(PersonalInfo, pk=pk)
    form = PatientForm(request.POST or None, instance=patient)
    if request.method == "POST" and form.is_valid():
        form.save()
        return redirect("patients:patient_list")
    return render(
        request,
        "patients/patient_form.html",
        {"form": form, "mode": "update", "patient": patient},
    )


def patient_delete(request, pk):
    patient = get_object_or_404(PersonalInfo, pk=pk)
    if request.method == "POST":
        patient.delete()
        return redirect("patients:patient_list")
    return render(
        request,
        "patients/patient_confirm_delete.html",
        {"patient": patient, "n_visits": patient.predictions.count()},
    )


# ---------------------------------------------------------------------------
# 내원(PredictionData) 임상변수 수정 + 재예측
# ---------------------------------------------------------------------------

def encounter_update(request, visit_id):
    encounter = get_object_or_404(PredictionData, visit_id=visit_id)
    patient = encounter.patient
    form = PredictionDataEditForm(request.POST or None, instance=encounter)

    if request.method == "POST" and form.is_valid():
        # 폼에 있는 필드만 갱신. 폼에 없는 19개 약물 등 기존 값은 instance 에서 보존.
        obj = form.save(commit=False)
        obj.save()

        # 갱신된 row 로 재예측 (patients 전처리 기준 — 학습 그룹핑과 일치).
        model = load_model()
        model_input = encounter_to_model_input(obj, model)
        probability = float(model.predict_proba(model_input)[0][1])

        # readmit_proba 는 기존 코드(revisit/퇴원 준비도) 관행대로 퍼센트(0~100)로 저장.
        obj.readmit_proba = round(probability * 100, 2)
        obj.risk_level = get_risk_label(probability)
        obj.predicted_at = timezone.now()
        obj.save(update_fields=["readmit_proba", "risk_level", "predicted_at"])

        return redirect("patients:patient_detail", pk=patient.pk)

    return render(
        request,
        "patients/encounter_form.html",
        {"form": form, "patient": patient, "encounter": encounter},
    )


# ---------------------------------------------------------------------------
# 환자 상세
# ---------------------------------------------------------------------------

def get_risk(encounter):
    if encounter.readmit_proba is not None:
        return round(encounter.readmit_proba, 1)
    return None


def _diag_view(code):
    # 원본 ICD9 코드 + group_by_icd9 카테고리. 빈 값/? 은 표시 제외.
    if not code or str(code).strip() in ("", "?", "nan"):
        return None
    return {"code": code, "group": group_by_icd9(code)}


def _active_meds(encounter):
    # 21개 약물 중 "No"가 아닌 것만 (change/diabetesMed 는 카드에서 별도 표시).
    meds = []
    for field in MEDICATION_FIELDS:
        value = getattr(encounter, field)
        if value and value != "No":
            meds.append({"name": field.replace("_", "-"), "status": value})
    return meds


def _build_chart(timeline):
    # 순수 SVG Line Chart 좌표 계산(템플릿 단순화 위해 뷰에서 산출).
    width, height = 720, 240
    pad_l, pad_r, pad_t, pad_b = 48, 24, 24, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(timeline)

    points = []
    for i, t in enumerate(timeline):
        x = pad_l + plot_w / 2 if n == 1 else pad_l + plot_w * i / (n - 1)
        y = pad_t + plot_h * (1 - t["risk"] / 100)
        points.append(
            {
                "x": round(x, 1),
                "y": round(y, 1),
                "risk": t["risk"],
                "date": t["date"],
                "seq": t["seq"],
            }
        )

    grid = []
    for val in (0, 25, 50, 75, 100):
        gy = pad_t + plot_h * (1 - val / 100)
        grid.append({"val": val, "y": round(gy, 1)})

    return {
        "w": width,
        "h": height,
        "pad_l": pad_l,
        "axis_y": round(pad_t + plot_h, 1),
        "points": points,
        "polyline": " ".join(f'{p["x"]},{p["y"]}' for p in points),
        "grid": grid,
    }


def _get_shap_top3(prediction_data):
    shaps = ShapResult.objects.filter(prediction=prediction_data).order_by('-shap_value')[:3]
    return [{"name": s.feature_name, "value": round(s.shap_value, 3)} for s in shaps]


def patient_detail(request, pk):
    patient = get_object_or_404(PersonalInfo, pk=pk)
    encounters = list(patient.predictions.order_by("visit_date", "encounter_id"))

    rows = []
    timeline = []
    latest_age = latest_gender = ""
    latest_shap = []
    for seq, e in enumerate(encounters, start=1):
        risk = get_risk(e)
        rows.append(
            {
                "e": e,
                "seq": seq,
                "risk": risk,
                "risk_level": e.risk_level,
                "diags": [
                    d
                    for d in (
                        _diag_view(e.diag_1),
                        _diag_view(e.diag_2),
                        _diag_view(e.diag_3),
                    )
                    if d
                ],
                "meds": _active_meds(e),
            }
        )
        if risk is not None:
            timeline.append({"seq": seq, "date": e.visit_date, "risk": risk})
        latest_age, latest_gender = e.age, e.gender

    # 최근 내원의 SHAP TOP3
    if encounters:
        latest_shap = _get_shap_top3(encounters[-1])

    return render(
        request,
        "patients/patient_detail.html",
        {
            "patient": patient,
            "rows": rows,
            "visit_count": len(encounters),
            "latest_age": latest_age,
            "latest_gender": latest_gender,
            "chart": _build_chart(timeline) if len(timeline) >= 2 else None,
            "single_visit": len(timeline) == 1,
            "shap_top3": latest_shap,
        },
    )
