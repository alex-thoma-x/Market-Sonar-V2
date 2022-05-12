from django.db import models
from home.models import Client

# Create your models here.
class stockdata(models.Model):        
    symbol=models.CharField(max_length=12)
    date=models.DateField(null=False)

class feedback(models.Model):
    user=models.ForeignKey(Client, on_delete=models.CASCADE)
    feedback=models.TextField(null=False,blank=False)