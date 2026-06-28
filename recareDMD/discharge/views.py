from django.shortcuts import render

def discharge_page(request):
    return render(request, 'discharge/discharge_page.html')