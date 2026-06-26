from django.urls import path 
from discharge import views

urlpatterns = [
    path('', view=views.DischargeGuide.as_vieW(),)
]