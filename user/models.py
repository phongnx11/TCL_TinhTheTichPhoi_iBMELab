
from django.db import models
from django.contrib.auth.models import User


class Profile(models.Model):
    user = models.OneToOneField(User , on_delete=models.CASCADE)
    auth_token = models.CharField(max_length=100 )
    is_verified = models.BooleanField(default=False)
    is_admin=models.BooleanField(default=False)
    is_normal_user=models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    time=models.IntegerField(default=50)
    is_active=models.BooleanField(default=True)
    phone=models.CharField(max_length=10)
    def __str__(self):
        return self.user.username


class UserUploadedFile(models.Model):
    user = models.ForeignKey(Profile,on_delete=models.CASCADE,null=True)
    id_file=models.CharField(max_length=9, null=True)
    drive_id=models.CharField(max_length=100)
    create_at=models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return str(self.user)



class File(models.Model):
    user = models.ForeignKey(Profile,on_delete=models.CASCADE,null=True)
    f_name = models.CharField(max_length=255)
    myfiles = models.FileField(upload_to="")
    def __str__(self):
        return str(self.user)



class ResultFile(models.Model):
    upload_file=models.ForeignKey(UserUploadedFile, on_delete=models.CASCADE, null=False)
    right_lung=models.CharField(max_length=20)
    left_lung=models.CharField(max_length=20)
    lung_volume=models.CharField(max_length=20)
    create_at=models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return str(self.upload_file)