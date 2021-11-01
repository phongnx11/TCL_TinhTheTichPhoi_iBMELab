from django.shortcuts import redirect, render
from django.contrib import messages
from .models import *
import uuid
from django.conf import settings
from django.core.mail import send_mail
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import PasswordChangeView,PasswordResetView,PasswordResetConfirmView
from django.contrib.auth.forms import PasswordResetForm,SetPasswordForm
from django.urls import reverse_lazy
from .forms import FileFieldForm,PasswordsChangingForm,Path
import numpy as np
import pydicom
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
import scipy.ndimage
from sklearn.cluster import KMeans
# import os
# import matplotlib.pyplot as plt
# import cv2 as cv
# from glob import glob
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from scipy import ndimage
# from skimage.morphology import watershed
# from skimage.feature import peak_local_max
# from skimage import segmentation
# from skimage import morphology
# from skimage import measure
# from skimage.transform import resize
# from plotly import __version__
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# from plotly.tools import FigureFactory as FF
# from plotly.graph_objs import *
# from django.views.generic.edit import FormView
# from django.http import HttpResponse
# from django.views import View
# import math
# from skimage.measure import label,regionprops, perimeter
# from skimage.morphology import binary_dilation, binary_opening
# from skimage.filters import roberts, sobel
from skimage import measure, feature
# from skimage.segmentation import clear_border, mark_boundaries
# from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from glob import glob
# from skimage.io import imread
from datetime import datetime
import shutil,os
from user.models import Profile
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
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
        return redirect('/')

    return render(request, 'dangnhap.html')





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


def result(request):
    return render(request,'result.html')


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




@login_required
def upload_file(request):
    request_user = Profile.objects.get(user__id=request.user.id)
    user_uploaded_files = UserUploadedFile.objects.all()
    id=request.user.id
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
        def load_scan(path):
            # os.listdir(path) = name of all files in path
            # dicom_read_file() get the dataset of ct image in string type
            slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
            # order the list by the increasement of instancenumber
            # [(0020,0013),(0020,0011),(0020,0015)]->[(0020,0011),(0020,0013),(0020,0015)]
            slices.sort(key=lambda x: int(x.InstanceNumber))
            # if 'try' fail or error, 'except' code will be operated
            try:
                # Position coordinate [x,y,z] , ImagePositionPatient[2] get the z coordinate
                # abs absolute value
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            for s in slices:
                s.SliceThickness = slice_thickness
            return slices

        # convert raw values of voxel in the images to HU
        def get_pixels_hu(scans):
            # s.pixel_array get ct image's pixel data in matrix
            # np.stack join many array(matrix) with the same dimension into new sequence
            image = np.stack([s.pixel_array for s in scans])
            # should be possible as values should always be low enough (<32k)
            # convert all dataframe columns into dtype int16
            image = image.astype(np.int16)
            # Set outside-of-scan pixels (HU = -2000) to 0
            # The intercept(threshold) is usually -1024, so air is approximately 0
            image[image == -2000] = 0
            # Convert to Hounsfield units (HU=pixel_value*slope+intercept)
            intercept = scans[0].RescaleIntercept if 'RescaleIntercept' in scans[0] else -1024
            slope = scans[0].RescaleSlope if 'RescaleSlope' in scans[0] else 1
            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(np.int16)
            image += np.int16(intercept)
            return np.array(image, dtype=np.int16)

        def resample(image, scan, new_spacing=[1, 1, 1]):
            # Determine current pixel spacing
            # map(function,iterable) -> return a list,tuple to inform function
            # get scan to be in form of float number
            spacing = [scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]]
            # change list to array
            spacing = np.array(spacing)

            resize_factor = spacing / new_spacing
            new_real_shape = image.shape * resize_factor
            # round after comma ','
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / image.shape
            new_spacing = spacing / real_resize_factor
            # change the size of image with a factor
            image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

            return image, new_spacing

        def make_lungmask(img, display=False):
            row_size = img.shape[0]
            col_size = img.shape[1]
            # compute average value
            mean = np.mean(img)
            # compute standard deviation
            std = np.std(img)
            img = img - mean
            img = img / std
            # Find the average pixel value near the lungs to renormalize washed out images
            middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
            mean = np.mean(middle)
            max = np.max(img)
            min = np.min(img)
            # To improve threshold finding, moving the underflow and overflow on the pixel spectrum
            img[img == max] = mean
            img[img == min] = mean

            # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air) -> cluster = 2
            # np.reshape(reshaped array, newshape (row,col))
            # np.prod() return number of elements in array
            # np.reshape(middle,[np.prod(middle.shape),1] change the array into [elementsx1] size
            kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
            # cluster_centers_: array(n.o cluster, n.o features) Coordinates of cluster centers.
            # .flatten() return array to one dimension
            # sorted() sort array from 1->n, alphabet a,b,c,d,...
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centers)
            # np.where(condition, true, else) if img<threshold, img =1, else = 0
            thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
            # Different labels are displayed in different colors
            labels = measure.label(thresh_img)
            # np.unique() sorted unique elements of array
            label_vals = np.unique(labels)
            # Measure properties of labeled image regions.
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                # bbbox(bounding box):tuple (min_row, min_col, max_row, max_col)
                B = prop.bbox
                C = prop.area
                if B[0] > 0 and B[1] > 0 and C > 500 and B[2] < img.shape[0] and B[3] < img.shape[1] - 10:
                    # label: int the label in labeled input img
                    good_labels.append(prop.label)
            mask = np.ndarray([row_size, col_size], dtype=np.int8)
            mask[:] = 0

            #  After just the lungs are left, we do another large dilation in order to fill in and out the lung mask
            for N in good_labels:
                mask = mask + np.where(labels == N, 1, 0)
            selem = disk(5)
            final_mask = ndi.binary_fill_holes(mask)
            original_mask = img * thresh_img * mask
            original_mask = np.where(original_mask != 0, 1, 0)
            new_label = measure.label(original_mask)
            # mask = morphology.dilation(mask,np.ones(3,3])) # one last dilation
            if (display):
                fig, ax = plt.subplots(3, 2, figsize=[12, 12])
                ax[0, 0].set_title("Original")
                ax[0, 0].imshow(img, cmap='gray')
                ax[0, 0].axis('off')
                ax[0, 1].set_title("Threshold")
                ax[0, 1].imshow(thresh_img, cmap='gray')
                ax[0, 1].axis('off')
                ax[1, 0].set_title("Color Labels")
                ax[1, 0].imshow(labels)
                ax[1, 0].axis('off')
                ax[1, 1].set_title("Good Labels")
                ax[1, 1].imshow(mask, cmap='gray')
                ax[1, 1].axis('off')
                ax[2, 0].set_title("Final Mask")
                ax[2, 0].imshow(final_mask, cmap='gray')
                ax[2, 0].axis('off')
                ax[2, 1].set_title("Apply Mask on Original")
                ax[2, 1].imshow(new_label, cmap='gray')
                ax[2, 1].axis('off')
                plt.show()
            return final_mask
            # return mask

        def shallowest_lung(lung_mask, min_row, min_col, max_row, max_col):
            col_size = []
            dis = int((max_col - min_col) / 10)
            for i in range(min_col + dis, max_col - dis):
                a = 0
                for j in range(min_row, max_row + 1):
                    if lung_mask[j][i] == 1:
                        a += 1
                col_size.append(a)
            mincol_index = min_col + dis + col_size.index(min(col_size))
            for k in range(min_row, max_row + 1):
                if lung_mask[k][mincol_index] == 1:
                    break
            minrow_index = k
            return mincol_index, minrow_index

        # for connected mask only
        def connected_region(img_connected, img_original):
            row = img_connected.shape[0]
            col = img_connected.shape[1]
            label_img = measure.label(img_connected)
            label_vals = np.unique(label_img)
            regions = measure.regionprops(label_img)
            for prop in regions:
                box = prop.bbox
                lung_area = prop.area
            right_mask = np.ndarray([row, col], dtype=np.int8)
            right_mask[:] = 0
            good_labels = []
            col_min, row_min = shallowest_lung(img_connected, box[0], box[1], box[2], box[3])
            max_row = min([int(row_min + row / 10), row])
            min_col = max([int(col_min - col / 15), 0])
            max_col = min([int(col_min + col / 10), col])
            new_mask = np.zeros((row, col))
            for j in range(box[0], max_row):
                for k in range(min_col, max_col):
                    if img_connected[j][k] == 1:
                        new_mask[j][k] = 1
            new_img = img_original * new_mask
            number = []
            for i in new_img:
                for j in i:
                    if j != 0:
                        number.append(j)
            mean = np.mean(number)
            if mean == 0 or lung_area < 6000:
                if box[1] < col / 2 and box[3] < col * 8 / 10:
                    good_labels.append(prop.label)
                for N in good_labels:
                    right_mask = right_mask + np.where(label_img == N, 1, 0)
                right_mask = np.where(right_mask != 0, 1, 0)
                new_img = img_connected
            else:
                i = 0
                new_label = np.ndarray([row, col], dtype=np.int8)
                while len(label_vals) == 2 and i < 20:
                    new_img = np.where(new_img < mean, 1, 0)
                    diff = (new_img > 0) ^ (new_mask > 0)
                    diff = np.where(diff == 1, 0, 1)
                    selem = disk(10)
                    new_img = img_connected * diff
                    new_img = ndi.binary_fill_holes(new_img)
                    test_label = measure.label(new_img)
                    label_vals = np.unique(test_label)
                    if len(label_vals) != 2:
                        region = measure.regionprops(test_label)
                        for prop in region:
                            A = prop.area
                            if A < 100:
                                new_label = np.where(test_label == prop.label, 0, test_label)
                        label_vals = np.unique(new_label)
                    mean -= 10
                    i += 1
                if len(label_vals) == 2:
                    region = measure.regionprops(test_label)
                    for prop in region:
                        B = prop.bbox
                    mincol, minrow = shallowest_lung(new_img, B[0], B[1], B[2], B[3])
                    for i in range(0, mincol):
                        for k in range(0, row):
                            if new_img[k][i] == 1:
                                right_mask[k][i] = 1
                else:
                    region = measure.regionprops(test_label)
                    for prop in region:
                        B = prop.bbox
                        if B[1] < col / 2 and B[3] < col * 7 / 10:
                            good_labels.append(prop.label)
                    for N in good_labels:
                        right_mask = right_mask + np.where(test_label == N, 1, 0)
                    right_mask = np.where(right_mask != 0, 1, 0)
            return right_mask, new_img

        def divide_lung(img_original, display=False):
            img = make_lungmask(img_original)
            row = img.shape[0]
            col = img.shape[1]
            label_img = measure.label(img)
            label_vals = np.unique(label_img)
            regions = measure.regionprops(label_img)
            good_labels = []
            right_mask = np.ndarray([row, col], dtype=np.int8)
            right_mask[:] = 0
            left_mask = np.ndarray([row, col], dtype=np.int8)
            left_mask[:] = 0
            box = []
            label_part = []
            if len(label_vals) == 3:
                for prop in regions:
                    box.append(prop.bbox)
                    label_part.append(prop.label)
                if box[0][1] < box[1][1]:
                    good_labels.append(label_part[0])
                else:
                    good_labels.append(label_part[1])
                for N in good_labels:
                    right_mask = right_mask + np.where(label_img == N, 1, 0)
                    right_mask = np.where(right_mask != 0, 1, 0)
                left_mask = img - right_mask

            if len(label_vals) > 3:
                for prop in regions:
                    B = prop.bbox
                    if B[0] > 0 and B[1] > 0 and B[2] < row and B[1] < col * 4 / 10 and B[3] < col * 7 / 10:
                        # label: int the label in labeled input img
                        good_labels.append(prop.label)
                    for N in good_labels:
                        right_mask = right_mask + np.where(label_img == N, 1, 0)
                        right_mask = np.where(right_mask != 0, 1, 0)
                    left_mask = img - right_mask

            if len(label_vals) == 2:
                right_mask, img = connected_region(img, img_original)
                left_mask = img - right_mask
            if (display):
                fig, ax = plt.subplots(3, 2, figsize=[12, 12])
                ax[0, 0].set_title("Original")
                ax[0, 0].imshow(img_original, cmap='gray')
                ax[0, 0].axis('off')
                ax[0, 1].set_title("Label")
                ax[0, 1].imshow(img, cmap='gray')
                ax[0, 1].axis('off')
                ax[1, 0].set_title("Distance")
                ax[1, 0].imshow(label_img)
                ax[1, 0].axis('off')
                ax[1, 1].set_title("Label")
                ax[1, 1].imshow(img, cmap='gray')
                ax[1, 1].axis('off')
                ax[2, 0].set_title("Right mask")
                ax[2, 0].imshow(right_mask, cmap='gray')
                ax[2, 0].axis('off')
                ax[2, 1].set_title("Left mask")
                ax[2, 1].imshow(left_mask, cmap='gray')
                ax[2, 1].axis('off')
                plt.show()
            return img, right_mask, left_mask

        def measure_lung(img, px=0):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] == 1:
                        px += 1
            return px

        def sample_stack(stack, rows=10, cols=10, start_with=0, show_every=2):
            # get image size 12x12 and set the coordinate
            fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
            for i in range(rows * cols):
                ind = start_with + i * show_every
                ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
                ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
                ax[int(i / rows), int(i % rows)].axis('off')
            plt.show()

        id = 1
        patient = load_scan(data_path)
        imgs = get_pixels_hu(patient)
        # save an array to a numpy file (.npy) format
        np.save(output_path + "fullimages_%d.npy" % (id), imgs)
        file_used = output_path + "fullimages_%d.npy" % id
        imgs_to_process = np.load(file_used).astype(np.float64)
        # each slice is resampled in 1x1x1 mm pixels and slices.
        imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1, 1, 1])
        new_size = imgs_after_resamp.shape

        mask_stack = np.zeros((imgs_after_resamp.shape[0], imgs_after_resamp.shape[1], imgs_after_resamp.shape[2]))
        leftmask_stack = np.zeros((imgs_after_resamp.shape[0], imgs_after_resamp.shape[1], imgs_after_resamp.shape[2]))
        rightmask_stack = np.zeros((imgs_after_resamp.shape[0], imgs_after_resamp.shape[1], imgs_after_resamp.shape[2]))
        i = 0
        volume = 0
        right_mask = 0
        left_mask = 0
        for img in imgs_after_resamp:
            mask_stack[i], rightmask_stack[i], leftmask_stack[i] = divide_lung(img)
            mask_stack[i] = np.where(mask_stack[i] != 0, 1.0, 0.0)
            rightmask_stack[i] = np.where(rightmask_stack[i] != 0, 1.0, 0.0)
            leftmask_stack[i] = np.where(leftmask_stack[i] != 0, 1.0, 0.0)
            slice_volume = measure_lung(mask_stack[i], 0)
            right_volume = measure_lung(rightmask_stack[i], 0)
            left_volume = measure_lung(leftmask_stack[i], 0)
            right_mask += right_volume
            left_mask += left_volume
            volume += slice_volume
            i += 1

        patient_lung = "patient lung volume is " + str(volume) + " mm^3"
        right_lung = "\nright lung volume is " + str(right_mask) + " mm^3"
        left_lung = "\nleft lung volume is " + str(left_mask) + " mm^3"
        print(patient_lung)
        print(right_lung)
        print(left_lung)
        UserFile=UserUploadedFile.objects.create(user=request_user, drive_id=0)
        UserFile.save()
        Result=ResultFile.objects.create(file=UserFile, right_lung=right_mask, left_lung=left_mask, lung_volume=volume)
        Result.save()
        context ={
            'right_lung':make_lungmask,
            'left_lung':left_mask,
            'lung_volume':patient_lung
        }
        redirect('/result')
    return render(request,'display_file.html',context)




def test(request):
    b = Path()
    return render(request, 'test.html',{'f':b})


