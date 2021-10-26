from django.contrib import admin
from .models import *
# Register your models here.


admin.site.register(Profile)
admin.site.register(UserUploadedFile)
admin.site.register(ResultFile)
admin.site.register(File)