import pickle
import cv2
#for SSIM
from skimage.metrics import structural_similarity 
from skimage import img_as_float
#for CPL
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.losses import mean_squared_error as mse
import numpy as np
#for CS
from sklearn.metrics.pairwise import cosine_similarity

                
                
def svm_classifier(org_img_path, gen_img_path):
    # load the model
    with open('svm95.pkl', 'rb') as f:
        clf = pickle.load(f)

    #PSNR
    img_org = cv2.imread(org_img_path)
    img_gen = cv2.imread(gen_img_path)
    psnr = cv2.PSNR(img_org, img_gen)

    #SSIM
    img_org = img_as_float(cv2.imread(org_img_path, cv2.IMREAD_GRAYSCALE))
    img_gen = img_as_float(cv2.imread(gen_img_path, cv2.IMREAD_GRAYSCALE))
    ssim = structural_similarity (img_org, img_gen, data_range=img_gen.max()-img_gen.min())

    #CPL
    img_shape = (512, 320)
    img_gen = cv2.resize(cv2.imread(gen_img_path), img_shape).astype(float)
    img_org = cv2.resize(cv2.imread(org_img_path), img_shape).astype(float)
    model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_shape[::-1], 3))
    x_org = np.array([preprocess_input(img_org)])
    x_gen = np.array([preprocess_input(img_gen)])
    x_org = model.predict(x_org)
    x_gen = model.predict(x_gen)
    cpl = np.array(mse(x_org.reshape(1, -1), x_gen.reshape(1, -1)))[0]

    #CS
    cs = cosine_similarity(x_org.reshape(1, -1), x_gen.reshape(1, -1))[0, 0]
    x = [psnr, ssim, cpl, cs]
    return clf.predict([x])


def preprocessing(org_img_path, gen_img_path):
    #PSNR
    img_org = cv2.imread(org_img_path)
    img_gen = cv2.imread(gen_img_path)
    psnr = cv2.PSNR(img_org, img_gen)

    #SSIM
    img_org = img_as_float(cv2.imread(org_img_path, cv2.IMREAD_GRAYSCALE))
    img_gen = img_as_float(cv2.imread(gen_img_path, cv2.IMREAD_GRAYSCALE))
    ssim = structural_similarity (img_org, img_gen, data_range=img_gen.max()-img_gen.min())

    #CPL
    img_shape = (512, 320)
    img_gen = cv2.resize(cv2.imread(gen_img_path), img_shape).astype(float)
    img_org = cv2.resize(cv2.imread(org_img_path), img_shape).astype(float)
    model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_shape[::-1], 3))
    x_org = np.array([preprocess_input(img_org)])
    x_gen = np.array([preprocess_input(img_gen)])
    x_org = model.predict(x_org, verbose = 0)
    x_gen = model.predict(x_gen, verbose = 0)
    cpl = np.array(mse(x_org.reshape(1, -1), x_gen.reshape(1, -1)))[0]

    #CS
    cs = cosine_similarity(x_org.reshape(1, -1), x_gen.reshape(1, -1))[0, 0]
    
    return psnr, ssim, cpl, cs


#___________________________________________________________________________________________________
if __name__ == "__main__":
    org_img = ""
    gen_img = ""
    print(svm_classifier(org_img, gen_img))
