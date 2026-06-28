from django.db import models


class PredictionData(models.Model):
    patient_name = models.CharField(max_length=100, blank=True, null=True)
    gender = models.CharField(max_length=20, blank=True, null=True)
    age = models.CharField(max_length=20, blank=True, null=True)

    features = models.JSONField()
    prediction_result = models.FloatField()
    risk_level = models.CharField(max_length=20)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.patient_name} - {self.risk_level}"
