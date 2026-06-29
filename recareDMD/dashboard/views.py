from django.db.models import OuterRef, Subquery
from django.shortcuts import render
from patients.models import PersonalInfo, PredictionData

def landing(request):
    return render(request, "dashboard/landing.html")

def home(request):
    total = PersonalInfo.objects.count()

    latest_pk = Subquery(
        PredictionData.objects.filter(
            patient=OuterRef('patient')
        ).order_by('-predicted_at', '-visit_id').values('visit_id')[:1]
    )
    latest_preds = PredictionData.objects.filter(visit_id=latest_pk)

    high = latest_preds.filter(risk_level='고위험').count()
    caution = latest_preds.filter(risk_level='중위험').count()
    normal = latest_preds.filter(risk_level='저위험').count()

    high_patients = (
        latest_preds
        .filter(risk_level='고위험')
        .select_related('patient')
        .order_by('-predicted_at')[:10]
    )

    recent = (
        latest_preds
        .exclude(predicted_at=None)
        .select_related('patient')
        .order_by('-predicted_at')[:10]
    )

    return render(request, "dashboard/home.html", {
        "active_page": "dashboard",
        "total": total,
        "high": high,
        "caution": caution,
        "normal": normal,
        "high_patients": high_patients,
        "recent": recent,
        "risk_counts": [normal, caution, high],
    })