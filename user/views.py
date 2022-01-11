from django.shortcuts import redirect, render
from django.contrib import messages
from .models import *
from .decorators import admin_only, user_only
import uuid
from django.conf import settings
from django.core.mail import send_mail
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import PasswordChangeView,PasswordResetView,PasswordResetConfirmView
from django.contrib.auth.forms import PasswordResetForm,SetPasswordForm
from django.urls import reverse_lazy
from .forms import PasswordsChangingForm
import numpy as np
from  django.contrib.auth.models import User
from glob import glob
from datetime import datetime
import shutil,os
from user.models import Profile
from pydrive.auth import GoogleAuth
from .utils import load_data
GoogleAuth.DEFAULT_SETTINGS['client_config_file'] ="user/client_secrets.json"

def check():
    user=Profile.objects.all()
    for i in user:
        if i.time==0:
            i.is_active=False



class PasswordsChangeView(PasswordChangeView):
    form_class = PasswordsChangingForm
    success_url = reverse_lazy('password_change_done')




def PasswordChangeDoneView(request):
    return render(request,'password_change_done.html')





class PasswordsResetView(PasswordResetView):
    form_class = PasswordResetForm
    success_url = reverse_lazy('password_reset_done')



def PasswordResetDoneView(request):
    return render(request,'password_reset_done.html')




class PasswordsResetConfirmView(PasswordResetConfirmView):
    form_class = SetPasswordForm
    success_url = reverse_lazy('password_reset_complete')




def login_attempt(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user_obj = User.objects.filter(username=username).first()
        if user_obj is None:
            messages.warning(request, 'Tên người dùng không tồn tại')
            return redirect('/accounts/login')

        profile_obj = Profile.objects.filter(user=user_obj).first()

        if not profile_obj.is_verified:
            messages.warning(request, 'Tài khoản chưa xác thực vui lòng kiểm tra email')
            return redirect('/accounts/login')

        user = authenticate(username=username, password=password)
        if user is None:
            messages.error(request, 'Sai mật khẩu')
            return redirect('/accounts/login')

        login(request, user)
        ur=Profile.objects.get(user=user)
        if ur.is_admin==True:
            return redirect('admin_statistical')
        if ur.is_normal_user==True:
            return redirect('/')
    return render(request, 'dangnhap.html')




@user_only
@login_required(login_url="login")
def home(request):
    return render(request, 'home.html')





def register_attempt(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            if User.objects.filter(username=username).first():
                messages.warning(request, 'Tên người dùng để trống hoặc đã tồn tại')
                return redirect('/register')

            if User.objects.filter(email=email).first():
                messages.warning(request, 'Email đã tồn tại')
                return redirect('/register')

            user_obj = User(username=username, email=email)
            user_obj.set_password(password)
            user_obj.save()
            auth_token = str(uuid.uuid4())
            profile_obj = Profile.objects.create(user=user_obj, auth_token=auth_token)
            profile_obj.save()
            send_mail_after_registration(email, auth_token)
            return redirect('/token')
        except Exception as e:
            print(e)
    return render(request, 'dangki.html')





def success(request):
    return render(request, 'success.html')




def token_send(request):
    return render(request, 'token_send.html')




def verify(request, auth_token):
    try:
        profile_obj = Profile.objects.filter(auth_token=auth_token).first()
        if profile_obj:
            if profile_obj.is_verified:
                messages.success(request, 'Bạn đã xác thực tài khoản này rồi')
                return redirect('/accounts/login')
            profile_obj.is_verified = True
            profile_obj.save()
            messages.success(request, 'Tài khoản của bạn đã xác thực.')
            return redirect('/accounts/login')
        else:
            return redirect('/error')
    except Exception as e:
        print(e)
        return redirect('/')


def error_404_view(request,exception):
    return render(request,'error404.html')



def error_page(request):
    return render(request, 'error.html')




def log_out(request):
    logout(request)
    messages.success(request,'Đăng xuất thành công')
    return redirect('/accounts/login')




def send_mail_after_registration(email, auth_token):
    subject = 'Tài khoản của bạn cần xác thực'
    message = f'Nhấn vào đường link này để xác thực: https://demothetich2000.herokuapp.com/verify/{auth_token}'
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email]
    send_mail(subject, message, email_from, recipient_list)

# https://accounts.google.com/DisplayUnlockCaptcha
# https://myaccount.google.com/lesssecureapps?pli=1&rapt=AEjHL4MKvsNPAkDdvjmxFXA3SG6EQk2YWe6JqfK4FVVjvrc3yuVMPlsEqpRqP2bDFwR-vAtiiUVL4sT3xFhWMCMsptJ1EZ_xJw



@login_required
def display_file(request):
    request_user = Profile.objects.get(user__id=request.user.id)
    context = {
    "request_user": request_user
    }
    return render(request, "display_file.html", context)


@user_only
@login_required
def user_statistical(request):
    id=request.user.id
    pr=Profile.objects.get(user__id=id)
    file=UserUploadedFile.objects.filter(user__id=pr.id)
    return render(request,"user_statistical.html",{'file':file})

@user_only
@login_required
def result(request, id):
    result=ResultFile.objects.get(upload_file__id=id)
    url=result.url
    x = np.load('./media/' + url + '/x.npy').tolist()
    y = np.load('./media/' + url + '/y.npy').tolist()
    z = np.load('./media/' + url + '/z.npy').tolist()
    d = np.load('./media/' + url + '/d.npy').tolist()
    e = np.load('./media/' + url + '/e.npy').tolist()
    f = np.load('./media/' + url + '/f.npy').tolist()
    context={'x':x, 'y': y, 'z':z , 'd':d, 'e':e, 'f':f,
             'right_lung': result.right_lung,
             'left_lung': result.left_lung,
             'lung_volume': result.lung_volume,
             }
    return render(request,'result.html', context)



@admin_only
def admin_statistical(request):
    users = Profile.objects.filter(is_admin=False)
    turns = []
    for user in users:
        upload_file = UserUploadedFile.objects.filter(user=user)
        statistical_results = ResultFile.objects.filter(upload_file__in=upload_file)
        count = statistical_results.count()
        turns.append(count)
    
    context = {
        'users':users,
        'turns':turns,
    }
    return render(request,"admin_statistical.html",context)

@admin_only
def user_setrole(request,id):
    user = Profile.objects.get(pk = id)
    user.is_verified = False
    user.save()
    return redirect('/admin_statistical')
@admin_only
def user_turn_active(request,id):
    user = Profile.objects.get(pk=id)
    user.is_verified = True
    user.save()
    return redirect('/admin_statistical')

@admin_only
def detele_user(request,id):
    user = User.objects.get(pk=id)
    user.delete()
    
    return redirect('/admin_statistical')


@login_required
@user_only
def upload_file(request):
    request_user = Profile.objects.get(user__id=request.user.id)
    id=request.user.id
    time1=datetime.now().minute
    print(time1)
    if request.method == "POST":
        name = request.POST.get("filename")
        uploaded_files = request.FILES.getlist("uploadfiles")
        urlk= str(datetime.today().year)+ str(datetime.today().month)+ str(datetime.today().day)+str(datetime.now().hour)+str(datetime.now().minute)+str(datetime.now().second)+ str(id)
        print(urlk)
        Folder='./media/'+urlk
        os.makedirs(Folder)
        # gauth = GoogleAuth()
        # drive = GoogleDrive(gauth)
        for uploaded_file in uploaded_files:
            File(f_name=name, myfiles=uploaded_file,user=request_user).save()
        for uploaded_file in uploaded_files:
            uploaded_file_name =str(uploaded_file)
            global server_store_path
            uploaded_file_path ='./media/' + uploaded_file_name
            server_store_path = './media/' + urlk
            shutil.move(uploaded_file_path, server_store_path)
        # entries = os.scandir(Folder)
        # upload_files = []
        # for entry in entries:
        #     print(entry.name)
        #     k = str(entry.name)
        #     upload_files.append(os.path.join(Folder, k))
        # folder_name = urlk
        # folder = drive.CreateFile({'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
        # folder.Upload()
        # print('File ID: %s' % folder.get('id'))
        # folder_id = str(folder.get('id'))
        # for upload_file in upload_files:
        #     gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
        #     gfile.SetContentFile(upload_file)
        #     gfile.Upload()  # Upload the file.
        #     print('success')
        # folder_contents = UserUploadedFile.objects.all()
        # folder_contents.delete()
        data_path = server_store_path
        # export result to other folder
        # open dicom files
        g = glob(data_path + '/*.dcm')
        output_path = working_path = 'test'
        # loop over the image files and store everything into a list
        right_mask, left_mask, volume, z=load_data(Folder, request_user, urlk)
        k1 = len(z)
        print(k1)
        h1= 3 * k1
        x1 = z[:, :, 0].reshape(-1)
        y1 = z[:, :, 1].reshape(-1)
        z1 = z[:, :, 2].reshape(-1)
        np.save('./media/'+urlk+'/x.npy', x1)
        np.save('./media/'+urlk+'/y.npy', y1)
        np.save('./media/'+urlk+'/z.npy', z1)
        p1 = x1.tolist()
        p2 = y1.tolist()
        p3 = z1.tolist()
        d1 = np.arange(0, h1, 3)
        e1 = np.arange(1, h1, 3)
        f1 = np.arange(2, h1, 3)
        np.save('./media/'+urlk+'/d.npy', d1)
        np.save('./media/'+urlk+'/e.npy', e1)
        np.save('./media/'+urlk+'/f.npy', f1)
        p4 = x1.tolist()
        p5 = y1.tolist()
        p6 = z1.tolist()

        context ={
            'right_lung': right_mask,
            'left_lung':left_mask,
            'lung_volume':volume,
            'x': p1, 'y': p2, 'z': p3, 'd': p4, 'e': p5, 'f': p6
        }
        time2=datetime.now().minute
        print(time2)
        return render(request, 'result.html', context)
    return render(request,'display_file.html')



def getUser(request):
    user=Profile.objects.all()
    return render(request, 'all_user.html', {'us': user})





