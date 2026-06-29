from django import forms

from .models import PersonalInfo, PredictionData


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


# 모델이 기대하는 연령대(그룹) 라벨. DB·학습 모두 그룹 문자열로 저장/사용한다.
AGE_GROUP_CHOICES = [("", "—")] + [
    (g, g)
    for g in [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
    ]
]
GENDER_CHOICES = [("", "—"), ("Female", "Female"), ("Male", "Male")]
CHANGE_CHOICES = [("", "—"), ("No", "No"), ("Ch", "Ch")]
DIABETESMED_CHOICES = [("", "—"), ("No", "No"), ("Yes", "Yes")]


class PredictionDataEditForm(forms.ModelForm):
    """기존 내원(PredictionData) 임상변수 수정 폼.

    약물은 화면에 insulin / metformin 2개만 노출한다. 나머지 19개 약물은 폼에
    없으므로 instance(기존 DB row) 값이 그대로 보존된다(절대 No 로 덮어쓰지 않음).
    age 는 모델이 그룹 카테고리를 기대하므로 그룹 드롭다운으로 받는다.
    """

    age = forms.ChoiceField(
        choices=AGE_GROUP_CHOICES, required=False, label="연령대",
        widget=forms.Select(attrs={"class": "form-input"}),
    )
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES, required=False, label="성별",
        widget=forms.Select(attrs={"class": "form-input"}),
    )
    change = forms.ChoiceField(
        choices=CHANGE_CHOICES, required=False, label="약물 변경(change)",
        widget=forms.Select(attrs={"class": "form-input"}),
    )
    diabetesMed = forms.ChoiceField(
        choices=DIABETESMED_CHOICES, required=False, label="당뇨약 처방(diabetesMed)",
        widget=forms.Select(attrs={"class": "form-input"}),
    )
    # diag 는 원본 ICD 코드(예 428) 입력. 일부 기존 행은 그룹명('Circulatory', 11자)이
    # 저장돼 있어 모델 max_length=10 으로는 편집이 막힘 -> 폼에서만 길이 완화.
    diag_1 = forms.CharField(
        max_length=20, required=False, label="진단1 (ICD 코드)",
        widget=forms.TextInput(attrs={"class": "form-input", "placeholder": "예: 428"}),
    )
    diag_2 = forms.CharField(
        max_length=20, required=False, label="진단2 (ICD 코드)",
        widget=forms.TextInput(attrs={"class": "form-input", "placeholder": "예: 250.83"}),
    )
    diag_3 = forms.CharField(
        max_length=20, required=False, label="진단3 (ICD 코드)",
        widget=forms.TextInput(attrs={"class": "form-input", "placeholder": "원본 ICD 코드"}),
    )

    class Meta:
        model = PredictionData
        fields = [
            "visit_date", "is_inpatient",
            "race", "gender", "age",
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_diagnoses",
            "number_outpatient", "number_emergency", "number_inpatient",
            "diag_1", "diag_2", "diag_3",
            "max_glu_serum", "A1Cresult",
            "discharge_disposition_id",
            "insulin", "metformin",
            "change", "diabetesMed",
            "readmitted",
        ]
        labels = {
            "visit_date": "내원일",
            "is_inpatient": "입원 여부",
            "race": "인종",
            "time_in_hospital": "재원일수",
            "num_lab_procedures": "검사 수",
            "num_procedures": "처치 수",
            "num_medications": "약물 수",
            "number_diagnoses": "진단 수",
            "number_outpatient": "외래 방문 횟수",
            "number_emergency": "응급 방문 횟수",
            "number_inpatient": "입원 횟수",
            "diag_1": "진단1 (ICD 코드)",
            "diag_2": "진단2 (ICD 코드)",
            "diag_3": "진단3 (ICD 코드)",
            "max_glu_serum": "혈당검사(max_glu_serum)",
            "A1Cresult": "당화혈색소(A1Cresult)",
            "discharge_disposition_id": "퇴원 경로",
            "insulin": "인슐린",
            "metformin": "메트포르민",
            "readmitted": "재입원 라벨(readmitted)",
        }
        widgets = {
            "visit_date": forms.DateInput(
                attrs={"type": "date", "class": "form-input"}, format="%Y-%m-%d"
            ),
            "race": forms.TextInput(attrs={"class": "form-input"}),
            "time_in_hospital": forms.NumberInput(attrs={"class": "form-input"}),
            "num_lab_procedures": forms.NumberInput(attrs={"class": "form-input"}),
            "num_procedures": forms.NumberInput(attrs={"class": "form-input"}),
            "num_medications": forms.NumberInput(attrs={"class": "form-input"}),
            "number_diagnoses": forms.NumberInput(attrs={"class": "form-input"}),
            "number_outpatient": forms.NumberInput(attrs={"class": "form-input"}),
            "number_emergency": forms.NumberInput(attrs={"class": "form-input"}),
            "number_inpatient": forms.NumberInput(attrs={"class": "form-input"}),
            "max_glu_serum": forms.TextInput(
                attrs={"class": "form-input", "placeholder": "None / Norm / >200 / >300"}
            ),
            "A1Cresult": forms.TextInput(
                attrs={"class": "form-input", "placeholder": "None / Norm / >7 / >8"}
            ),
            "discharge_disposition_id": forms.NumberInput(
                attrs={"class": "form-input", "placeholder": "퇴원 경로 코드 (예: 1=자택)"}
            ),
            "insulin": forms.Select(attrs={"class": "form-input"}),
            "metformin": forms.Select(attrs={"class": "form-input"}),
            "readmitted": forms.TextInput(
                attrs={"class": "form-input", "placeholder": "NO / <30 / >30"}
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # is_inpatient 은 BooleanField -> 체크박스. 정수형 카운트는 음수 방지.
        for name in (
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_diagnoses", "number_outpatient",
            "number_emergency", "number_inpatient",
        ):
            self.fields[name].widget.attrs["min"] = 0
