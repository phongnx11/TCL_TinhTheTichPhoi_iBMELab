
from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User , on_delete=models.CASCADE)
    auth_token = models.CharField(max_length=100 )
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username

class UserUploadedFile(models.Model):
    user = models.ForeignKey(Profile,on_delete=models.CASCADE,null=True)
    f_name = models.CharField(max_length=255)
    myfiles = models.FileField(upload_to="")
    def __str__(self):
        return self.f_name