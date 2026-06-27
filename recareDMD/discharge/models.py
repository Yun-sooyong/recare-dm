from django.db import models
from patients.models import Patient


class DischargeRecord(models.Model) :
    uid = models.ForeignKey(Patient, on_delete=models.CASCADE)
    discharge_score = models.PositiveSmallIntegerField()
    discharge_date = models.DateField() 