from django.shortcuts import render

def home(request):
    return render(request, "dashboard/home.html", {
        "active_page": "dashboard",
        # 나중에 모델 생기면 쿼리 추가
    })