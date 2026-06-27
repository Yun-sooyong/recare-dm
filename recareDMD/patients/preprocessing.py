"""원본 컬럼 -> 모델 입력(41 피처) 변환 공용 모듈.

prediction(정은)이 예측 직전에 import 해서 쓴다. 파생 3종(discharge_group,
inpatient_x_discharge, diag 그룹핑)은 학습 노트북 전처리와 동일하다.
"""

MODEL_FEATURES = [
    "race", "gender", "age", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed", "discharge_group",
    "inpatient_x_discharge",
]

# DB 약물 필드명(snake_case) -> 모델 피처명. 추측 변환 금지, 명시적 매핑.
MEDICATION_DB_TO_FEATURE = {
    "metformin": "metformin",
    "repaglinide": "repaglinide",
    "nateglinide": "nateglinide",
    "chlorpropamide": "chlorpropamide",
    "glimepiride": "glimepiride",
    "acetohexamide": "acetohexamide",
    "glipizide": "glipizide",
    "glyburide": "glyburide",
    "tolbutamide": "tolbutamide",
    "pioglitazone": "pioglitazone",
    "rosiglitazone": "rosiglitazone",
    "acarbose": "acarbose",
    "miglitol": "miglitol",
    "troglitazone": "troglitazone",
    "tolazamide": "tolazamide",
    "insulin": "insulin",
    "glyburide_metformin": "glyburide-metformin",
    "glipizide_metformin": "glipizide-metformin",
    "glimepiride_pioglitazone": "glimepiride-pioglitazone",
    "metformin_rosiglitazone": "metformin-rosiglitazone",
    "metformin_pioglitazone": "metformin-pioglitazone",
}


def group_discharge_id(val):
    if val is None:
        return "Other/Unknown"
    val = int(val)
    if val == 1:
        return "Home"
    elif val in [6, 8]:
        return "Home_Health_Service"
    elif val in [2, 3, 4, 5, 22, 23, 24, 27, 28, 29, 30]:
        return "Transfer/Facility"
    else:
        return "Other/Unknown"


def make_inpatient_x_discharge(number_inpatient, discharge_group_value):
    return f"{number_inpatient}회_{discharge_group_value}"


def group_by_icd9(code):
    s = str(code).strip()
    if code is None or s in ("?", "", " ", "nan"):
        return "missing"
    if s.startswith("V") or s.startswith("E"):
        return "Other"
    try:
        n = float(s)
    except ValueError:
        return "Other"
    i = int(n)
    if 390 <= i <= 459 or i == 785:
        return "Circulatory"
    if 460 <= i <= 519 or i == 786:
        return "Respiratory"
    if 520 <= i <= 579 or i == 787:
        return "Digestive"
    if 250 <= n < 251:
        return "Diabetes"
    if 800 <= i <= 999:
        return "Injury"
    if 710 <= i <= 739:
        return "Musculoskeletal"
    if 580 <= i <= 629 or i == 788:
        return "Genitourinary"
    if 140 <= i <= 239:
        return "Neoplasms"
    return "Other"


def encounter_to_features(encounter):
    # Encounter 인스턴스 -> 모델 입력 dict(MODEL_FEATURES 순서). 파생 3종 포함.
    f = {
        "race": encounter.race,
        "gender": encounter.gender,
        "age": encounter.age,
        "time_in_hospital": encounter.time_in_hospital,
        "num_lab_procedures": encounter.num_lab_procedures,
        "num_procedures": encounter.num_procedures,
        "num_medications": encounter.num_medications,
        "number_outpatient": encounter.number_outpatient,
        "number_emergency": encounter.number_emergency,
        "number_inpatient": encounter.number_inpatient,
        "number_diagnoses": encounter.number_diagnoses,
        "max_glu_serum": encounter.max_glu_serum,
        "A1Cresult": encounter.A1Cresult,
        "change": encounter.change,
        "diabetesMed": encounter.diabetesMed,
    }

    # diag 그룹핑 (원본 코드 아닌 카테고리)
    f["diag_1"] = group_by_icd9(encounter.diag_1)
    f["diag_2"] = group_by_icd9(encounter.diag_2)
    f["diag_3"] = group_by_icd9(encounter.diag_3)

    # 약물 21종: DB 필드 -> 모델 피처명
    for db_field, feature in MEDICATION_DB_TO_FEATURE.items():
        f[feature] = getattr(encounter, db_field)

    # 파생: discharge_group, inpatient_x_discharge
    dg = group_discharge_id(encounter.discharge_disposition_id)
    f["discharge_group"] = dg
    f["inpatient_x_discharge"] = make_inpatient_x_discharge(encounter.number_inpatient, dg)

    return {k: f[k] for k in MODEL_FEATURES}
