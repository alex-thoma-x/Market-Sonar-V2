from django.urls import path
from . import views

app_name='home'

urlpatterns = [
    path('',views.Landingpage,name="landing"),
    path('register',views.Register,name="Register"),
    path('login',views.Login,name="Login"),
    path('logout',views.Logout,name='logout')
]
