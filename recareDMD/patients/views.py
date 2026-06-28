import hashlib

from django.core.paginator import Paginator
from django.db.models import Count, Max, OuterRef, Subquery
from django.shortcuts import get_object_or_404, redirect, render

from .forms import PatientForm
from .models import MEDICATION_FIELDS, PersonalInfo, PredictionData
from .preprocessing import group_by_icd9

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
    ).order_by("patient_nbr")

    if q:
        patients = patients.filter(patient_nbr__icontains=q)

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
# 환자 상세
# ---------------------------------------------------------------------------

def get_demo_risk(encounter):
    """30일 재입원 위험도(%) — 데모 값.

    # DEMO: 예측 모델 미연결. prediction 앱 연동 시 이 함수만 실제 예측
    # (preprocessing.encounter_to_features → 모델 추론)으로 교체하면 됨.
    # encounter_id 기반 고정 시드 → 같은 내원은 매번 같은 값(5~85%).
    """
    digest = hashlib.md5(str(encounter.encounter_id).encode()).hexdigest()
    return int(digest, 16) % 81 + 5


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


def patient_detail(request, pk):
    patient = get_object_or_404(PersonalInfo, pk=pk)
    # 시간순 1회 조회. visit_seq property 반복 호출(N+1) 회피 위해 enumerate 로 차수 부여.
    encounters = list(patient.predictions.order_by("visit_date", "encounter_id"))

    rows = []
    timeline = []
    latest_age = latest_gender = ""
    for seq, e in enumerate(encounters, start=1):
        risk = get_demo_risk(e)
        rows.append(
            {
                "e": e,
                "seq": seq,
                "risk": risk,
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
        timeline.append({"seq": seq, "date": e.visit_date, "risk": risk})
        latest_age, latest_gender = e.age, e.gender  # 시간순 마지막 = 최근 내원

    # 카드/차트 모두 시간순(1차 → N차)으로 일관 표시.
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
        },
    )
