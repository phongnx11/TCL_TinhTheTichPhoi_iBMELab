import math
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2 as cv
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border, mark_boundaries
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from glob import glob
from skimage.io import imread
from io import BytesIO
import base64
#link to the ct images folder
data_path = 'D:/Dicom_Data/Dicom_TRAN VAN VUONG_207501098_520/1.3.12.2.1107.5.1.4.83610.30000020120218321811600103629'
#export result to other folder
output_path = working_path ='D:/Django_Project/TLC_TinhTheTichPhoi_iBMELab/media/'
#open dicom files
g = glob(data_path+'/*.dcm')
#loop over the image files and store everything into a list
def load_scan(path):
  #os.listdir(path) = name of all files in path
  #dicom_read_file() get the dataset of ct image in string type
  slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
  #order the list by the increasement of instancenumber
  #[(0020,0013),(0020,0011),(0020,0015)]->[(0020,0011),(0020,0013),(0020,0015)]
  slices.sort(key = lambda x: int(x.InstanceNumber))
  #if 'try' fail or error, 'except' code will be operated
  try:
    #Position coordinate [x,y,z] , ImagePositionPatient[2] get the z coordinate
    #abs absolute value
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
  except:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)    
  for s in slices:
    s.SliceThickness = slice_thickness       
  return slices

#convert raw values of voxel in the images to HU
def get_pixels_hu(scans):
  #s.pixel_array get ct image's pixel data in matrix 
  #np.stack join many array(matrix) with the same dimension into new sequence
  image = np.stack([s.pixel_array for s in scans])
  # should be possible as values should always be low enough (<32k)
  #convert all dataframe columns into dtype int16
  image = image.astype(np.int16) 
  # Set outside-of-scan pixels (HU = -2000) to 0
  # The intercept(threshold) is usually -1024, so air is approximately 0
  image[image == -2000] = 0    
  # Convert to Hounsfield units (HU=pixel_value*slope+intercept)
  intercept = scans[0].RescaleIntercept
  slope = scans[0].RescaleSlope   
  if slope != 1:
    image = slope * image.astype(np.float64)
    image = image.astype(np.int16)        
  image += np.int16(intercept)   
  return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    #map(function,iterable) -> return a list,tuple to inform function
    #get scan to be in form of float number
    spacing = [scan[0].SliceThickness,scan[0].PixelSpacing[0],scan[0].PixelSpacing[1]] 
    #change list to array
    spacing = np.array(spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    #round after comma ','
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    #change the size of image with a factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]  
    #compute average value  
    mean = np.mean(img)
    #compute standard deviation
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, moving the underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air) -> cluster = 2
    #np.reshape(reshaped array, newshape (row,col))
    #np.prod() return number of elements in array
    #np.reshape(middle,[np.prod(middle.shape),1] change the array into [elementsx1] size
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    #cluster_centers_: array(n.o cluster, n.o features) Coordinates of cluster centers.
    #.flatten() return array to one dimension 
    #sorted() sort array from 1->n, alphabet a,b,c,d,...
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    #np.where(condition, true, else) if img<threshold, img =1, else = 0
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    # Different labels are displayed in different colors
    labels = measure.label(thresh_img) 
    #np.unique() sorted unique elements of array
    label_vals = np.unique(labels)
    #Measure properties of labeled image regions.
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
      #bbbox(bounding box):tuple (min_row, min_col, max_row, max_col)
        B = prop.bbox
        C = prop.area
        if B[0]>0 and B[1]>img.shape[1]*2/10 and 100< C < 800 and B[3]<img.shape[1]*8/10 and B[2]<img.shape[0]*7/10 and B[3]-B[1]<img.shape[1]*1/10:
            #label: int the label in labeled input img 
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    selem = disk(5)
    final_mask = ndi.binary_fill_holes(mask)
    original_mask = img*thresh_img*mask
    original_mask = np.where(original_mask!=0,1,0)
    new_label = measure.label(original_mask)
    #mask = morphology.dilation(mask,np.ones(3,3])) # one last dilation
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
    #return mask

def sample_stack(stack, rows=10, cols=10, start_with=0, show_every=2):
    #get image size 12x12 and set the coordinate
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every        
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()
#3D plotting
def make_mesh(original_img, mask_img, threshold=-300, step_size=1):
    print("Transposing surface")
    #reorder matrix (0,1,3) -> transpose(2,1,0) -> (3,1,0)
    image = original_img*mask_img
    #p = image.transpose(2,1,0)    
    print("Calculating surface")
    #get the coordinate of each surface(slices) in 3D volume: vertice, face(triangle x,y,z)
    #all in (x,3) dimension
    #The threshold of -300 HU is fine for visualizing chest CT scans
    verts, faces, norm, val = measure.marching_cubes_lewiner(image, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces
def plt_3d(verts, faces):
    print("Drawing")
    x,y,z = zip(*verts) 
    #get the image size 10x10 inches
    fig = plt.figure(figsize=(10, 10))
    #create 3D axes
    ax = fig.add_subplot(111, projection='3d')
    f=verts[faces]
    j=0
    print(f[1500])
    print(f[1502])

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    print(mesh)
    #get color of appearent image (bone)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    
    ax.add_collection3d(mesh)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    
    plt.show()
id=1
patient = load_scan(data_path)
print("slice thickness: %f" %patient[0].SliceThickness)

print("pixel spacing (row,col): (%f, %f)" %(patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))
imgs = get_pixels_hu(patient)
#save an array to a numpy file (.npy) format
np.save(output_path + "fullimages_%d.npy" %(id), imgs)
file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64)
# each slice is resampled in 1x1x1 mm pixels and slices.
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
new_size = imgs_after_resamp.shape
print('image orginal:',imgs_to_process.shape)
print('image after resamp:',imgs_after_resamp.shape)
print('voxel value:',spacing)

mask_stack = np.zeros((imgs_after_resamp.shape[0],imgs_after_resamp.shape[1],imgs_after_resamp.shape[2]))
leftmask_stack = np.zeros((imgs_after_resamp.shape[0],imgs_after_resamp.shape[1],imgs_after_resamp.shape[2]))
rightmask_stack = np.zeros((imgs_after_resamp.shape[0],imgs_after_resamp.shape[1],imgs_after_resamp.shape[2]))
i = 0 
for img in imgs_after_resamp:
    mask_stack[i] = make_lungmask(img)
    mask_stack[i] = np.where(mask_stack[i]!=0,1.0,0.0)
    rightmask_stack[i] = np.where(rightmask_stack[i]!=0,1.0,0.0)
    leftmask_stack[i] = np.where(leftmask_stack[i]!=0,1.0,0.0)
    i+=1

step = int(mask_stack.shape[0]/100)
# sample_stack(imgs_after_resamp, start_with = 0, show_every = step)
# sample_stack(mask_stack, start_with = 0, show_every = step)

def make_mesh(original_img, mask_img, threshold=-300, step_size=1):
    print("Transposing surface")
    #reorder matrix (0,1,3) -> transpose(2,1,0) -> (3,1,0)
    image = original_img*mask_img
    image = image[::-1]
    p = image.transpose(2,1,0)    
    
    print("Calculating surface")
    #get the coordinate of each surface(slices) in 3D volume: vertice, face(triangle x,y,z)
    #all in (x,3) dimension
    #The threshold of -300 HU is fine for visualizing chest CT scans
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def getgraph():
    buffer = BytesIO()
    plt.savefig(buffer,format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def plot_graph():
    v, f = make_mesh(imgs_after_resamp,mask_stack, -1, 1)
    print(type(v))
    print(type(f))
    # for i in v:
    #   print(i)
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))

    plt_3d(v, f)
    graph = getgraph()
    return graph 