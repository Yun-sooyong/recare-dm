from django.db import models
from patients.models import PredictionData


class ShapResult(models.Model):
    prediction = models.ForeignKey(
        PredictionData,
        on_delete=models.CASCADE,
        related_name='shap_results'
    )

    feature_name = models.CharField(max_length=100)
    shap_value = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.feature_name}: {self.shap_value}"

