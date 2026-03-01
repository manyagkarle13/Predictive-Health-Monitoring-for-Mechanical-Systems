from django.urls import path
from .views import predict_rul

urlpatterns = [
    path('predict/', predict_rul),
]