import pandas as pd
from numpy import matlib
import Obj_Seg
from DenseUnet import Model_DenseUNet
from Model_DETR import Model_DETR
from Model_EADSPP_YoloV9 import Model_EADSPP_YoloV9
from Model_TransYOLO import Model_TransYolo
from Model_ViT_Deformable_DETR import Model_ViT_Deformable_DETR
from PSO import PSO
from ResUnet import Model_ResUnet
from Trans_Unet import Model_Trans_Unet
from Unet import Model_Unet
from WOA import WOA
from SCO import SCO
from Global_Vars import Global_vars
from GKSO import GKSO
from PROPOSED import PROPOSED
from Spatial_ViT_ResUnetpp import  Model_Spatial_ViT_ResUnet
from plot_results import *


def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df

def Read_image(path):
    img = cv.imread(path)
    img = cv.resize(img, (512,512))
    img = img.astype('uint8')
    return img


# Read the Dataset
an = 0
if an == 1:
    Datasets = './Dataset/archive/metadata.csv'
    df = pd.read_csv(Datasets)
    last_two_columns = df.iloc[:, -2:]
    images=[]
    target=[]
    path = './Dataset/archive/Preprocessed_512x512/'
    last_two_columns = np.asarray(last_two_columns)
    for i in range(5000):  # last_two_columns.shape[0]
        print(i,last_two_columns.shape[0])
        ori_img_path = path + last_two_columns[i ,1]
        tar_img_path = path + last_two_columns[i, 0]
        Original = Read_image(ori_img_path)
        Target = Read_image(tar_img_path)
        images.append(Original)
        target.append(Target)
    Original_ = np.asarray(images)
    target_ = np.asarray(target)
    np.save('Images.npy', Original_)
    np.save('GT_Img.npy', target_)

# Image Reconstruction
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Ground_truth = np.load('GT_Img.npy', allow_pickle=True)
    Unet, Eval_1  = Model_Unet(Images, Ground_truth)
    ResUnet, Eval_2 = Model_ResUnet(Images, Ground_truth)
    DenseUnet, Eval_3 = Model_DenseUNet(Images, Ground_truth)
    TransUnet, Eval_4 = Model_Trans_Unet(Images, Ground_truth)
    Proposed, Eval_5 = Model_Spatial_ViT_ResUnet(Images, Ground_truth)
    Seg = [Unet, ResUnet, DenseUnet, TransUnet, Proposed]
    np.save('Reconstructed_Img.npy', Proposed)
    np.save('Reconstructed_Image_CMP.npy',Seg)

# Optimization for Detection
an = 0
if an == 1:
    Feat = np.load('Reconstructed_Img.npy', allow_pickle=True)
    Target = np.load('Images.npy', allow_pickle=True)
    Global_vars.Feat = Feat
    Global_vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden neuron count in YoloV9 , No of Epochs in YoloV9 , Activation Function in YoloV9
    xmin = matlib.repmat(np.asarray([5, 5, 1]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 5]), Npop, 1)
    fname = Obj_Seg
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("PSO...")
    [bestfit1, fitness1, bestsol1, time1] = PSO(initsol, fname, xmin, xmax, Max_iter)

    print("WOA...")
    [bestfit2, fitness2, bestsol2, time2] = WOA(initsol, fname, xmin, xmax, Max_iter)

    print("SCO...")
    [bestfit3, fitness3, bestsol3, time3] = SCO(initsol, fname, xmin, xmax, Max_iter)

    print("GKSO...")
    [bestfit4, fitness4, bestsol4, time4] = GKSO(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol.npy', np.asarray(BestSol_CLS))

# Detection
an = 0
if an == 1:
    Images = np.load('Reconstructed_Img.npy', allow_pickle=True)
    Ground_truth = np.load('GT_Img.npy', allow_pickle=True)
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    Eval_1, Method_1 = Model_DETR(Images, Ground_truth)
    Eval_2, Method_2 = Model_TransYolo(Images, Ground_truth)
    Eval_3, Method_3 = Model_ViT_Deformable_DETR(Images, Ground_truth)
    Eval_4, Method_4 = Model_EADSPP_YoloV9(Images, Ground_truth)
    Eval_5, Proposed = Model_EADSPP_YoloV9(Images, Ground_truth, sol=BestSol[-1, :])
    Seg = [Method_1, Method_2, Method_3, Method_4, Proposed]
    np.save('Segmented_image.npy', Proposed)
    np.save('Seg_img.npy',Seg)


Plot_Conv()
Segmentation_Images()
Plot_Seg_Results()
gradcam_images()
Reconstructed_Images()
Plot_KFold_Reconstruction()
plot_Seg_Loss()
plot_SPE_tables()