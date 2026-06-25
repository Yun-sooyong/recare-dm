from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    path('', views.analysis_page, name='analysis_page'),
]