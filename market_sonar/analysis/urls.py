from django.urls import path
from . import views

app_name='stock'

urlpatterns = [
    path('news',views.News,name="news"),
    path('stock',views.stock_prediction,name="graph"),
    path('socio',views.socio,name="socio"),
    path('feedback',views.feedback,name="feedback"),

    ]