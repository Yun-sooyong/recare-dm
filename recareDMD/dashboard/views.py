from django.shortcuts import render   # ← 이거 추가
import json
from patients.models import PatientRecord

def home(request):
    qs = PatientRecord.objects.filter(is_deleted=False)
    
    high_risk_list  = qs.filter(risk_level="고위험").order_by("-risk_probability")[:8]
    recent_records  = qs.order_by("-created_at")[:5]
    
    risk_chart = json.dumps({
        "labels": ["정상", "주의", "고위험"],
        "values": [qs.filter(risk_level=l).count() for l in ["정상","주의","고위험"]],
        "colors": ["#059669", "#D97706", "#DC2626"],
    })
    
    return render(request, "dashboard/home.html", {
        "total_count":    qs.count(),
        "high_count":     qs.filter(risk_level="고위험").count(),
        "caution_count":  qs.filter(risk_level="주의").count(),
        "normal_count":   qs.filter(risk_level="정상").count(),
        "high_risk_list": high_risk_list,
        "recent_records": recent_records,
        "risk_chart":     risk_chart,
        "active_page":    "dashboard",
    })