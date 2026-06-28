from django.urls import path

from discharge import views

urlpatterns = [
    path('guide/<int:patient_id>', views.discharge_guide,name='discharge_guide'),
    path('print/<int:patient_id>', views.discharge_print,name='discharge_print'),
    path('revisit/<int:patient_id>', views.revisit ,name='discharge_revisit'),
]