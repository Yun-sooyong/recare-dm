from django.shortcuts import render

def analysis_page(request):
    return render(request, 'analysis/analysis_page.html')