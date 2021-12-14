from os import name
from django.contrib import admin
from django.urls import path
from .views import *
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', home, name="home"),
    path('register', register_attempt, name="register"),
    path('accounts/login/', login_attempt, name="login"),
    path('token', token_send, name="token_send"),
    path('success', success, name='success'),
    path('verify/<auth_token>', verify, name="verify"),
    path('error', error_page, name="error"),
    path('logout/',log_out,name='logout'),
    path('password_change/',PasswordsChangeView.as_view(template_name='./password_change.html'), name='password_change'),
    path('password_change/done',PasswordChangeDoneView, name='password_change_done'),
    path('password_reset/',PasswordsResetView.as_view(template_name='./forgot-password.html'),name='password_reset'),
    path('password_reset_done/',PasswordResetDoneView,name='password_reset_done'),
    path('reset/<uidb64>/<token>/',PasswordsResetConfirmView.as_view(template_name='./password_reset_confirm.html'),name='password_reset_confirm'),
    path('reset/done/',auth_views.PasswordResetCompleteView.as_view(template_name='./password_reset_complete.html'),name='password_reset_complete'),
    path("myfile/", display_file,name='display'),
    path('upload/', upload_file, name="upload"),
    path('test/', test,name='test'),
    path('result/',result,name="result"),
    path('statistical/',user_statistical,name="user_statistical"),
    path('admin_statistical/',admin_statistical,name="admin_statistical"),
    path('delete_user/<int:id>/',detele_user,name='delete_user'),
    path('user_setrole/<int:id>/',user_setrole,name='user_setrole'),
    path('user_turn_active/<int:id>/',user_turn_active,name='user_turn_active'),
    path('test_graph/',test_graph,name='graph'),
    # path('process/', process, name='pro'),


]
