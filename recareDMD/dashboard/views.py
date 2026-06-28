from django.shortcuts import render
from patients.models import PersonalInfo, PredictionData

def home(request):
    total = PersonalInfo.objects.count()
    high = PredictionData.objects.filter(risk_level='high').values('patient').distinct().count()
    caution = PredictionData.objects.filter(risk_level='medium').values('patient').distinct().count()
    normal = PredictionData.objects.filter(risk_level='low').values('patient').distinct().count()

    high_patients = (
        PredictionData.objects
        .filter(risk_level='high')
        .select_related('patient')
        .order_by('-predicted_at')[:10]
    )

    recent = (
        PredictionData.objects
        .exclude(predicted_at=None)
        .select_related('patient')
        .order_by('-predicted_at')[:10]
    )

    risk_counts = [
        PredictionData.objects.filter(risk_level='low').values('patient').distinct().count(),
        PredictionData.objects.filter(risk_level='medium').values('patient').distinct().count(),
        PredictionData.objects.filter(risk_level='high').values('patient').distinct().count(),
    ]

    return render(request, "dashboard/home.html", {
        "active_page": "dashboard",
        "total": total,
        "high": high,
        "caution": caution,
        "normal": normal,
        "high_patients": high_patients,
        "recent": recent,
        "risk_counts": risk_counts,
    })