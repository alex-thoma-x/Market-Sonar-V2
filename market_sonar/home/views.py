from django.shortcuts import render
from django.contrib import messages
from django.http import *
from django.shortcuts import render
from  .models import Client
from django.shortcuts import redirect
# Create your views here.

def Landingpage(req):
    return render(req,'home/landing.html')

#-------------------------------------------------------

def Login(req):
    #cred=users.objects.all()
    if req.method == 'POST':
        username=req.POST['uname']
        password=req.POST['password']
        cred=Client.objects.filter(uname=username,password=password,status=1)
        if cred.count()==1:
            for i in cred:
                id=i.id
            req.session['id']=id 
            return redirect("stock:news")       
        else:
            messages.error(req, f"Login failed")
            return HttpResponse(f"Login Failed Try again")
        
    return render(req,'home/login.html')


    #------------------------------------------------------------------


def Register(req):
    duplication_set=set()
    user_details=Client.objects.all()
    for individual in user_details:
        duplication_set.add(individual.uname)
        duplication_set.add(individual.email)
        duplication_set.add(individual.mobile)

    if req.method == 'POST':
        mob=req.POST['mob']
        email=req.POST['email']
        uname=req.POST['uname']
        password=req.POST['pass']
        pass1=req.POST['pass1']
        usr=Client()
        if uname=='' or email=='' or mob==''  or password=='' or pass1=='':
            messages.info(req,'please fill all fields')
        elif mob not in duplication_set and email not in duplication_set and uname not in duplication_set:

            usr.name= req.POST['name']
            usr.email= req.POST['email']
            usr.mobile= req.POST['mob']
            usr.uname= req.POST['uname']
            usr.password= req.POST['pass']
            
            if password != pass1:
                messages.error(req, f"passwords doesnt match")
        
            else:
                usr.save()
                return render(req,'home/login.html')
        else:
            if mob in duplication_set:
                messages.error(req, f"Mobile already registered")
            elif email in duplication_set:
                messages.error(req, f"email already registered")
            else:
                messages.error(req, f"username already registered")
        print(duplication_set)

    return render(req,'home/register.html')

    #-------------------------------------------------------------------------

def Logout(req):
    req.session.flush()
    return redirect("home:Login")