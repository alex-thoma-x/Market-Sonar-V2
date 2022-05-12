from django.contrib import admin
from django.contrib.auth.models import Group
from .models import Client

admin.site.site_header="Market Sonar"

class usr(admin.ModelAdmin):
    
    list_display=('uname','id','email','status','role')
    list_filter=('role','status')    
    search_fields = ['uname', 'email','id', ]
    list_editable = ['status']
    
admin.site.register(Client,usr)
admin.site.unregister(Group)