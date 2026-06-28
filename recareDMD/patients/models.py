from django.db import models

MEDICATION_FIELDS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide_metformin",
    "glipizide_metformin",
    "glimepiride_pioglitazone",
    "metformin_rosiglitazone",
    "metformin_pioglitazone",
]

MEDICATION_CHOICES = [
    ("No", "No"),
    ("Steady", "Steady"),
    ("Up", "Up"),
    ("Down", "Down"),
]


class PersonalInfo(models.Model):
    patient_nbr = models.BigIntegerField(unique=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["patient_nbr"]

    def __str__(self):
        return f"PersonalInfo({self.patient_nbr})"

    @property
    def visit_count(self):
        # 화면 표시용 총 내원 횟수. 모델 입력 number_* 와 무관.
        return self.predictions.count()


class PredictionData(models.Model):
    # 문서상 VisitHistory/PredictionData 통합. 한 행 = 한 내원 + (있으면) 예측 결과.
    visit_id = models.AutoField(primary_key=True)
    patient = models.ForeignKey(
        PersonalInfo, on_delete=models.CASCADE, related_name="predictions"
    )
    encounter_id = models.BigIntegerField(unique=True)

    visit_date = models.DateField()
    is_inpatient = models.BooleanField(default=False)

    race = models.CharField(max_length=20, blank=True)
    gender = models.CharField(max_length=10, blank=True)
    age = models.CharField(max_length=10, blank=True)

    time_in_hospital = models.IntegerField(null=True, blank=True)
    num_lab_procedures = models.IntegerField(null=True, blank=True)
    num_procedures = models.IntegerField(null=True, blank=True)
    num_medications = models.IntegerField(null=True, blank=True)
    number_diagnoses = models.IntegerField(null=True, blank=True)

    # 누수주의 3종. 원본 csv 값 그대로 저장. DB row 카운트로 만들지 말 것.
    number_outpatient = models.IntegerField(null=True, blank=True)
    number_emergency = models.IntegerField(null=True, blank=True)
    number_inpatient = models.IntegerField(null=True, blank=True)

    diag_1 = models.CharField(max_length=10, blank=True)
    diag_2 = models.CharField(max_length=10, blank=True)
    diag_3 = models.CharField(max_length=10, blank=True)

    max_glu_serum = models.CharField(max_length=10, blank=True)
    A1Cresult = models.CharField(max_length=10, blank=True)

    # discharge_group, inpatient_x_discharge 파생 재생성의 소스. 파생값 자체는 저장 안 함.
    discharge_disposition_id = models.IntegerField(null=True, blank=True)

    metformin = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    repaglinide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    nateglinide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    chlorpropamide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    glimepiride = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    acetohexamide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    glipizide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    glyburide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    tolbutamide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    pioglitazone = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    rosiglitazone = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    acarbose = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    miglitol = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    troglitazone = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    tolazamide = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    insulin = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    glyburide_metformin = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    glipizide_metformin = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    glimepiride_pioglitazone = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    metformin_rosiglitazone = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)
    metformin_pioglitazone = models.CharField(max_length=10, default="No", choices=MEDICATION_CHOICES)

    change = models.CharField(max_length=10, blank=True)
    diabetesMed = models.CharField(max_length=10, blank=True)

    # 실제 재입원 라벨(타깃). 원본값 NO/<30/>30 그대로 저장. 모델 입력 아님(누수 아님).
    readmitted = models.CharField(max_length=10, blank=True)

    # ── 예측 결과 (PredictionData 통합. 한 내원당 1건) ──
    # 아직 예측 안 된 내원은 null. prediction(정은)이 채운다.
    readmit_proba = models.FloatField(null=True, blank=True,
        help_text="30일 재입원 위험도 (0~1). 미예측 시 null")
    risk_level = models.CharField(max_length=10, blank=True,
        help_text="위험 등급 (high/medium/low 등). 미예측 시 빈 문자열")
    predicted_at = models.DateTimeField(null=True, blank=True,
        help_text="예측 수행 시각. 미예측 시 null")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["patient_id", "visit_date", "encounter_id"]

    def __str__(self):
        return f"PredictionData({self.encounter_id}) of PersonalInfo({self.patient.patient_nbr})"

    @property
    def readmit_30d(self):
        # 30일 내 재입원 여부. readmitted 원본 라벨에서 유도.
        return self.readmitted == "<30"

    @property
    def visit_seq(self):
        # 내원 차수. 저장 안 하고 계산. visit_date 오름차순, 동일자는 encounter_id tie-break.
        earlier = self.patient.predictions.filter(
            models.Q(visit_date__lt=self.visit_date)
            | models.Q(visit_date=self.visit_date, encounter_id__lte=self.encounter_id)
        )
        return earlier.count()
