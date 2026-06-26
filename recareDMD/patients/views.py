from django.shortcuts import render
from django.shortcuts import redirect

def patient_list(request):
    return render(request, 'patients/patient_list.html')

def save_bulk_patients(request):
    return redirect('patients:home')