from django.urls import path
from . import views

app_name = "analysis"

urlpatterns = [
    path("<int:prediction_id>/", views.shap_result, name="shap_result"),
]