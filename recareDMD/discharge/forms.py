from django import forms

from .models import DischargeRecord


# 퇴원 기록 입력 폼 (퇴원 준비도 점수 + 퇴원일)
class DischargeRecordForm(forms.ModelForm):
    class Meta:
        model = DischargeRecord
        fields = ['discharge_score', 'discharge_date']
        widgets = {
            'discharge_score': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 0,
                'placeholder': '퇴원 준비도 점수',
            }),
            'discharge_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
            }),
        }
        labels = {
            'discharge_score': '퇴원 준비도 점수',
            'discharge_date': '퇴원 예정일',
        }
