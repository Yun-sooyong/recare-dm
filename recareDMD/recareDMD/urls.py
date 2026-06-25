# recareDMD/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('dashboard.urls')),
    path('patients/', include('patients.urls')),
    path('discharge/', include('discharge.urls')),
    path('prediction/', include('prediction.urls')),
    path('analysis/', include('analysis.urls')),
]