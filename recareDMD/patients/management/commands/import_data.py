import random
from datetime import date, timedelta

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from patients.models import Patient, PredictionData, MEDICATION_FIELDS
from patients.preprocessing import MEDICATION_DB_TO_FEATURE

SEED = 42
BASE_DATE = date(2023, 1, 1)
EXCLUDE_DISCHARGE = {11, 13, 14, 19, 20, 21}  # 사망 / 호스피스

INT_FIELDS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_diagnoses", "number_outpatient", "number_emergency", "number_inpatient",
    "discharge_disposition_id",
]
CHAR_FIELDS = [
    "race", "gender", "age", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed", "readmitted",
]


def _to_int(v):
    s = str(v).strip()
    if s in ("", "?"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _to_char(v):
    s = str(v).strip()
    return "" if s == "?" else s


class Command(BaseCommand):
    help = "diabetic_data.csv 정제·샘플링 후 Patient/PredictionData 적재. number_* 원본값 보존."

    def add_arguments(self, parser):
        parser.add_argument("csv_path", help="원본 diabetic_data.csv 경로")
        parser.add_argument("--patients", type=int, default=250, help="목표 환자 수")
        parser.add_argument("--encounters", type=int, default=500, help="목표 내원 수")
        parser.add_argument("--clear", action="store_true", help="적재 전 Patient/PredictionData 전체 삭제")

    def handle(self, *args, **opts):
        path = opts["csv_path"]
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except FileNotFoundError:
            raise CommandError(f"csv 없음: {path}")

        df = self._clean(df)
        picked = self._sample(df, opts["patients"], opts["encounters"])
        sample = df[df["patient_nbr"].isin(picked)].copy()
        visit_dates = self._assign_visit_dates(sample)

        if opts["clear"]:
            Patient.objects.all().delete()

        self._load(sample, visit_dates)
        self._report(sample)

    def _clean(self, df):
        dd = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
        df = df[~dd.isin(EXCLUDE_DISCHARGE)]
        df = df[df["gender"] != "Unknown/Invalid"]
        df = df.copy()
        df["race"] = df["race"].replace("?", "missing")
        df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)
        return df.reset_index(drop=True)

    def _sample(self, df, target_patients, target_encounters):
        rng = random.Random(SEED)
        counts = df["patient_nbr"].value_counts()
        multi = [p for p in counts.index if counts[p] >= 2]
        single = [p for p in counts.index if counts[p] == 1]
        rng.shuffle(multi)

        readmit_by_patient = (
            df[df["patient_nbr"].isin(single)]
            .set_index("patient_nbr")["readmit_30d"]
            .to_dict()
        )
        pos = [p for p in single if readmit_by_patient.get(p) == 1]
        neg = [p for p in single if readmit_by_patient.get(p) == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        single_pref = pos + neg

        picked = []
        enc = 0
        multi_budget = int(target_encounters * 0.65)
        for p in multi:
            if enc >= multi_budget or len(picked) >= target_patients:
                break
            picked.append(p)
            enc += int(counts[p])
        for p in single_pref:
            if len(picked) >= target_patients:
                break
            picked.append(p)
            enc += 1
        return set(picked)

    def _assign_visit_dates(self, sample):
        rng = random.Random(SEED + 1)
        sample = sample.copy()
        sample["_e"] = sample["encounter_id"].astype(int)
        dates = {}
        for _, grp in sample.sort_values(["patient_nbr", "_e"]).groupby("patient_nbr", sort=False):
            cur = BASE_DATE + timedelta(days=rng.randint(0, 364))
            first = True
            for _, row in grp.iterrows():
                if not first:
                    cur = cur + timedelta(days=rng.randint(30, 120))
                first = False
                dates[int(row["encounter_id"])] = cur
        return dates

    def _load(self, sample, visit_dates):
        with transaction.atomic():
            for _, row in sample.iterrows():
                patient, _new = Patient.objects.get_or_create(
                    patient_nbr=int(row["patient_nbr"])
                )
                enc_id = int(row["encounter_id"])
                if PredictionData.objects.filter(encounter_id=enc_id).exists():
                    continue

                tih = _to_int(row.get("time_in_hospital"))
                fields = {
                    "patient": patient,
                    "encounter_id": enc_id,
                    "visit_date": visit_dates[enc_id],
                    "is_inpatient": bool(tih and tih >= 1),
                }
                for f in INT_FIELDS:
                    fields[f] = _to_int(row.get(f))
                for f in CHAR_FIELDS:
                    fields[f] = _to_char(row.get(f))
                for db_field in MEDICATION_FIELDS:
                    csv_col = MEDICATION_DB_TO_FEATURE[db_field]
                    fields[db_field] = _to_char(row.get(csv_col)) or "No"

                PredictionData.objects.create(**fields)

    def _report(self, sample):
        total_p = Patient.objects.count()
        total_e = PredictionData.objects.count()
        from django.db.models import Count

        multi_p = (
            Patient.objects.annotate(n=Count("predictions"))
            .filter(n__gte=2)
            .count()
        )
        readmit_pos = PredictionData.objects.filter(readmitted="<30").count()
        inpatient_gt0 = PredictionData.objects.filter(number_inpatient__gt=0).count()

        self.stdout.write(self.style.SUCCESS("=== 적재 완료 ==="))
        self.stdout.write(f"Patient 수: {total_p}")
        self.stdout.write(f"PredictionData(내원) 수: {total_e}")
        self.stdout.write(f"내원 2회 이상 환자 수: {multi_p}")
        if total_e:
            self.stdout.write(
                f"readmit_30d=1 비율: {readmit_pos}/{total_e} = {readmit_pos / total_e:.1%}"
            )
        self.stdout.write(f"number_inpatient>0 내원 수(누수방지 확인): {inpatient_gt0}")
