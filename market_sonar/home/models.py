from django.db import models

# Create your models here.


# Create your models here.
class Client(models.Model):
    Active=1
    InActive=0
    user_status=[
        (Active,'active'),
        (InActive,'inactive'),
    ]
    user=1
    Expert=0
    user_role=[
        (user,"USER"),
        (Expert,'EXPERT')
    ]
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=35)
    email=models.CharField(max_length=50)
    mobile=models.CharField(max_length=12)
    uname=models.CharField(max_length=15)
    password=models.CharField(max_length=30)
    role=models.PositiveSmallIntegerField(default=1,choices=user_role)
    status=models.PositiveSmallIntegerField(default=1,choices=user_status)