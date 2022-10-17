from django.urls import path

from . import views

app_name='appUpload'
urlpatterns = [
    path('',views.Home.as_view(),name='index'),
    path('generateDesc/', views.upload, name='generateDesc'),
]

