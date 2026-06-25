from django.shortcuts import render

def patient_list(request):
    return render(request, 'patients/patient_list.html')