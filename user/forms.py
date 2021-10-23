from django.contrib.auth.forms import forms,PasswordResetForm,PasswordChangeForm
from django.contrib.auth.models import User
class FileFieldForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs=
        {'multiple': True, 'webkitdirectory': True, 'directory': True}))


class PasswordsChangingForm(PasswordChangeForm):
    old_password =  forms.CharField(widget=forms.PasswordInput(attrs={'class':'form-control','type':'password'}))
    new_password1 = forms.CharField(max_length=100,widget=forms.PasswordInput(attrs={'class':'form-control','type':'password'}))
    new_password2 = forms.CharField(max_length=100,widget=forms.PasswordInput(attrs={'class':'form-control','type':'password'}))

    class Meta:
        model = User
        fields=('old_password','new_password1','new_password2')

class Path(forms.Form):
    duongdan = forms.CharField(widget=forms.Textarea(attrs={'class': 'thanhlam', 'id': 'noidung'}))