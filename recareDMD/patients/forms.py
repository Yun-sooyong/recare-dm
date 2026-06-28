from django import forms

from .models import PersonalInfo


class PatientForm(forms.ModelForm):
    class Meta:
        model = PersonalInfo
        fields = ["patient_nbr"]
        labels = {"patient_nbr": "환자코드"}
        widgets = {
            "patient_nbr": forms.NumberInput(
                attrs={"class": "form-input", "placeholder": "환자코드 입력"}
            )
        }
