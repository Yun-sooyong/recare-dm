from django import forms

from .models import Patient


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ["patient_nbr"]
        labels = {"patient_nbr": "환자코드"}
        widgets = {
            "patient_nbr": forms.NumberInput(
                attrs={"class": "form-input", "placeholder": "환자코드 입력"}
            )
        }
