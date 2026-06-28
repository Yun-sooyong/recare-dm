from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    path('', views.prediction_page, name='prediction_page'),

    path('single/', views.single_predict, name='single_predict'),
    path('bulk/', views.bulk_predict, name='bulk_predict'),

    path('download/sample-csv/', views.download_sample_csv, name='download_sample_csv'),
    path('download/sample-pdf/', views.download_sample_pdf, name='download_sample_pdf'),

    path('cart/add/', views.add_patient_cart, name='add_patient_cart'),
    path('cart/predict/', views.predict_patient_cart, name='predict_patient_cart'),
    path('cart/clear/', views.clear_patient_cart, name='clear_patient_cart'),
]