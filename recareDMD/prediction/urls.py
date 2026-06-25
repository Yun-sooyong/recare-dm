from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    path('', views.prediction_page, name='prediction_page'),
    path('single/', views.single_predict, name='single_predict'),
    path('bulk/', views.bulk_predict, name='bulk_predict'),
]