from django.db import models


class DischargeRecord(models.Model) :
    uid = models.ForeignKey(PersonalInfo, on_delete=models.CASCADE)
    discharge_score = models.PositiveSmallIntegerField()
    discharge_date = models.DateField() 