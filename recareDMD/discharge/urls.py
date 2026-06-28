from django.urls import path
from . import views

app_name = 'discharge'

urlpatterns = [
    path('', views.discharge_page, name='discharge_page'),
]