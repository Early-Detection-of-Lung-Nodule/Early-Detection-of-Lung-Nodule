import numpy as np
from sklearn.metrics import mean_squared_error
from math import log10, sqrt
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
import cv2 as cv
from math import log10, sqrt
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2yiq
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim

# https://en.wikipedia.org/wiki/Confusion_matrix
def preValidation(actual, predict):
    if actual.shape != predict.shape:
        raise Exception("Actual and Predicted array shape must be equal")
    Max = 1
    if len(np.unique(actual)) == 2 and np.prod(np.unique(actual) == np.array([0, 1])):
        Max = np.unique(actual)[-1]
    elif not np.prod(np.unique(actual) == np.array([0, 1])):
        raise Exception("Actual Values are must be 0 and 1")
    if not np.prod(np.unique(predict) == np.array([0, 1])):
        raise Exception("Predicted Values are must be 0 and 1")
    return Max


def findConfusionMatrix(actual, predict, Max):
    act_one = np.where(actual == Max)
    act_zero = np.where(actual == 0)
    pred_one = np.where(predict == Max)
    pred_zero = np.where(predict == 0)

    '''Find Shape of the Each Dimension for Single Array Conversion'''
    array = [actual.shape[i] for i in range(len(actual.shape))]

    Act_One = np.zeros(shape=act_one[0].shape[0], dtype=np.int32)
    Act_Zero = np.zeros(shape=act_zero[0].shape[0], dtype=np.int32)
    Pred_One = np.zeros(shape=pred_one[0].shape[0], dtype=np.int32)
    Pred_Zero = np.zeros(shape=pred_zero[0].shape[0], dtype=np.int32)

    '''Convert Single Array for Easy Intersection'''
    for iter in range(len(act_one) - 1):
        Act_One += act_one[iter] * np.prod(array[iter + 1:])
        Act_Zero += act_zero[iter] * np.prod(array[iter + 1:])
        Pred_One += pred_one[iter] * np.prod(array[iter + 1:])
        Pred_Zero += pred_zero[iter] * np.prod(array[iter + 1:])
    Act_One += act_one[len(act_one) - 1]
    Act_Zero += act_zero[len(act_zero) - 1]
    Pred_One += pred_one[len(pred_one) - 1]
    Pred_Zero += pred_zero[len(pred_zero) - 1]

    '''Find Confusion Matrix'''
    # 1 ---> TP (True Positive) ------> If Actual = 1 and Predicted = 1
    TP = len(np.intersect1d(Act_One, Pred_One))
    # 2 ---> TN (True Negative) ------> If Actual = 0 and Predicted = 0
    TN = len(np.intersect1d(Act_Zero, Pred_Zero))
    # 3 ---> FP (False Positive) -----> If Actual = 0 and Predicted = 1
    FP = len(np.intersect1d(Act_Zero, Pred_One))
    # 4 ---> FN (False Negative) -----> If Actual = 1 and Predicted = 0
    FN = len(np.intersect1d(Act_One, Pred_Zero))

    # Positive P = TP + FN
    # Negative N = FP + TN
    # Predicted Positive PP = TP + FP
    # Predicted Negative PN = TN + FN
    # Total Population = P + N (or) PP + PN

    return array, [TP, TN, FP, FN]


def Accuracy(TP, TN, FP, FN):
    # Overall Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy * 100  # for Percentage


def Sensitivity(TP, FN):
    # Sensitivity, Hitrate, Recall, or True Positive Rate (TPR) = 1 - FNR
    # sensitivity = TP / P
    sensitivity = TP / (TP + FN)
    return sensitivity * 100  # for Percentage


def Specificity(TN, FP):
    # Specificity or True Negative Rate (TNR) = 1 - FPR
    # specificity = TN / N
    specificity = TN / (FP + TN)
    return specificity * 100  # for Percentage


def Precision(TP, FP):
    # Precision or Positive Predictive Value (PPV) = 1 - FDR
    # Precision = TP / PP
    precision = TP / (TP + FP)
    return precision * 100  # for Percentage


def FPR(TN, FP):
    # Fall out or False Positive Rate (FPR) = 1 - TNR
    # FPR = FP / N
    fpr = FP / (FP + TN)
    return fpr * 100  # for Percentage


def FNR(TP, FN):
    # False Negative Rate = 1 - TPR
    # FNR = FN / P
    fnr = FN / (TP + FN)
    return fnr * 100  # for Percentage


def NPV(TN, FN):
    # Negative Predictive Value (NPV) = 1- FOR
    # NPV = TN + PN
    npv = TN / (FN + TN)
    return npv * 100  # for Percentage


def FDR(TP, FP):
    # False Discovery Rate (FDR) = 1 - PPV
    # FDR = FP / PP
    fdr = FP / (TP + FP)
    return fdr * 100  # for Percentage


def F1SCORE(TP, FP, FN):
    # F1 score is the harmonic mean of Precision and Sensitivity
    # F1SCORE = (2 * PPV * TPR) / (PPV + TPR)
    f1score = (2 * TP) / (2 * TP + FP + FN)
    return f1score * 100  # for Percentage


def MCC(TP, TN, FP, FN):
    # Matthews Correlation Coefficient (MCC)
    # MCC = np.math.sqrt(TPR * TNR * PPV * NPV) - np.math.sqrt(FNR * FPR * FOR * FDR)
    mcc = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return mcc


def FOR(TN, FN):
    # False Omission Rate (FOR) = 1 - NPV
    # FOR = FN / PN
    For = FN / (FN + TN)
    return For * 100  # for Percentage


def PT(fpr, sensitivity):
    # Prevalence Threshold (PT)
    pt = (np.math.sqrt((sensitivity / 100) * (fpr / 100)) - (fpr / 100)) / ((sensitivity / 100) - (fpr / 100))
    return pt * 100  # for Percentage


def CSI(TP, FP, FN):
    # Threat Score (TS) or Critical Success Index (CSI)
    csi = TP / (TP + FN + FP)
    return csi * 100  # for Percentage


def BA(sensitivity, specificity):
    # Balanced Accuracy (BA)
    ba = (sensitivity + specificity) / 2
    return ba


def FM(sensitivity, precision):
    # Fowlkes–Mallows Index (FM)
    # FM = np.math.sqrt(precision * sensitivity)
    fm = np.math.sqrt((precision / 100) * (sensitivity / 100))
    return fm * 100


def BM(sensitivity, specificity):
    # Informedness or Bookmaker Informedness (BM)
    # BM = TPR + TNR - 1
    bm = ((sensitivity / 100) + (specificity / 100) - 1)
    return bm * 100


def MK(precision, npv):
    # Markedness (MK) or DeltaP (Δp)
    # MK = PPV + NPV - 1
    mk = ((precision / 100) + (npv / 100) - 1)
    return mk * 100


def PositiveLivelihoodRatio(tpr, fpr):
    # Positive Likelihood Ratio (LR+)
    lrplus = tpr / fpr
    return lrplus


def NegativeLivelihoodRatio(tnr, fnr):
    # Negative Likelihood Ratio (LR-)
    lrminus = fnr / tnr
    return lrminus


def DOR(lrplus, lrminus):
    # Diagnostic Odds Ratio (DOR)
    dor = lrplus / lrminus
    return dor


def Prevalence(TP, TN, FP, FN):
    # Prevalence = P / (P + N)
    prevalence = (TP + FN) / (TP + TN + FP + FN)
    return prevalence * 100


def ClassificationEvaluation(actual, predict):
    Max = preValidation(actual=actual, predict=predict)
    array, [TP, TN, FP, FN] = findConfusionMatrix(actual, predict, Max)
    accuracy = Accuracy(TP, TN, FP, FN)
    sensitivity = Sensitivity(TP, FN)
    specificity = Specificity(TN, FP)
    precision = Precision(TP, FP)
    fpr = FPR(TN, FP)
    fnr = FNR(TP, FN)
    npv = NPV(TN, FN)
    fdr = FDR(TP, FP)
    f1score = F1SCORE(TP, FP, FN)
    mcc = MCC(TP, TN, FP, FN)
    For = FOR(TN, FN)
    pt = PT(fpr, sensitivity)
    csi = CSI(TP, FP, FN)
    ba = BA(sensitivity, specificity)
    fm = FM(sensitivity, precision)
    bm = BM(sensitivity, specificity)
    mk = MK(precision, npv)
    lrplus = PositiveLivelihoodRatio(sensitivity, fpr)
    lrminus = NegativeLivelihoodRatio(specificity, fnr)
    dor = DOR(lrplus, lrminus)
    prevalence = Prevalence(TP, TN, FP, FN)
    Values = np.asarray([TP, TN, FP, FN, accuracy, sensitivity, specificity, precision, fpr, fnr, npv, fdr,
                         f1score, mcc, For, pt, csi, ba, fm, bm, mk, lrplus, lrminus, dor, prevalence])
    Verification(Values, array)
    return Values


def Verification(Values, array):
    Limit_0_100 = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    Limit_0_1 = np.array([13])
    if not (np.prod(array) == np.sum(Values[:4])):
        raise Exception("Something went wrong - Please check values")
    if (not (0 <= Values[Limit_0_100].all() <= 100)) or (not (0 <= Values[Limit_0_1].all() <= 1)):
        raise Exception('Something went wrong')


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def net_evaluation(actual, predict):
    Max = preValidation(actual=actual, predict=predict)
    array, [TP, TN, FP, FN] = findConfusionMatrix(actual, predict, Max)
    Dice = (2 * TP) / ((2 * TP) + FP + FN) * 100
    Jaccard = TP / (TP + FP + FN) * 100  # IOU and Jac card are same metrics
    psnr = PSNR(actual, predict)
    mse = mean_squared_error(actual, predict)
    IoU = TP / (TP + FP + FN) * 100
    accuracy = Accuracy(TP, TN, FP, FN)
    sensitivity = Sensitivity(TP, FN)
    specificity = Specificity(TN, FP)
    precision = Precision(TP, FP)
    fpr = FPR(TN, FP)
    fnr = FNR(TP, FN)
    npv = NPV(TN, FN)
    fdr = FDR(TP, FP)
    f1score = F1SCORE(TP, FP, FN)
    mcc = MCC(TP, TN, FP, FN)
    EVAL = [TP, TN, FP, FN, Dice, IoU, accuracy, psnr, mse, sensitivity, specificity, precision, fpr, fnr, npv,
            fdr, f1score, mcc]
    return EVAL




def UQI(img1, img2):
    N = img1.shape[0] * img1.shape[1]

    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    img1_var = np.var(img1)
    img2_var = np.var(img2)
    covar = np.mean((img1 - img1_mean) * (img2 - img2_mean))

    numerator = 4 * img1_mean * img2_mean * covar
    denominator = (img1_mean ** 2 + img2_mean ** 2) * (img1_var + img2_var)

    uqi = numerator / denominator
    return uqi


def phase_congruency(image):
    return sobel(image)


def FSIM(img1, img2):
    T1 = 0.85
    T2 = 160
    if len(img1.shape) == 2 or len(img2.shape) == 2:
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2RGB)
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)

    img1_y = rgb2yiq(img1)[:, :, 0]
    img2_y = rgb2yiq(img2)[:, :, 0]

    PC1 = phase_congruency(img1_y)
    PC2 = phase_congruency(img2_y)

    S_PC = (2 * PC1 * PC2 + T1) / (PC1 ** 2 + PC2 ** 2 + T1)
    S_G = (2 * sobel(img1_y) * sobel(img2_y) + T2) / (sobel(img1_y) ** 2 + sobel(img2_y) ** 2 + T2)

    FSIM = np.sum(S_PC * S_G) / np.sum(np.maximum(PC1, PC2))
    return FSIM



def vifp_mscale(ref, dist):
    sigma_nsq = 2

    EPS = 1e-10

    num, den = 0.0, 0.0
    for scale in range(4):
        N = 2.0 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if scale > 0:
            ref = gaussian_filter(ref, sd)[::2, ::2]
            dist = gaussian_filter(dist, sd)[::2, ::2]

        mu1 = gaussian_filter(ref, sd)
        mu2 = gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu1

        sigma1_sq = gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + EPS)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < EPS] = 0
        sv_sq[sigma1_sq < EPS] = sigma2_sq[sigma1_sq < EPS]
        sigma1_sq[sigma1_sq < EPS] = 0

        g[sigma2_sq < EPS] = 0
        sv_sq[sigma2_sq < EPS] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den
    return vifp


def VIF(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return vifp_mscale(img1, img2)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def SSIM(img1, img2):
    import cv2 as cv
    if len(img1.shape) == 3 or len(img2.shape) == 3:
        # Convert images to grayscale
        grayA = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        grayB = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    else:
        grayA = img1
        grayB = img2

    # Compute SSIM between two images
    score, _ = ssim(grayA, grayB, full=True)
    return score


def MSE(img1, img2):
    return mse(img1, img2)


def NRMSE(img1, img2):
    return np.sqrt(mse(img1, img2)) / (np.max(img1) - np.min(img1))


def average_difference(img1, img2):
    return np.mean(img1 - img2)


def NAE(img1, img2):
    return np.sum(np.abs(img1 - img2)) / np.sum(np.abs(img1))


def MAE(img1, img2):
    return np.mean(np.abs(img1 - img2))




def MS_SSIM(img1, img2, weights=None):
    """
    Compute the Multi-Scale Structural Similarity Index (MS-SSIM) between two images.

    Parameters:
        img1, img2: Input images (grayscale or color)
        weights: Weights for different scales, default recommended values from the original paper.

    Returns:
        ms_ssim: The Multi-Scale SSIM index.
    """
    if weights is None:
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    # Convert images to grayscale if they are color
    if len(img1.shape) == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mssim = []
    mcs = []
    levels = weights.size
    for level in range(levels):
        ssim_map, cs_map = ssim(img1, img2, full=True)
        mssim.append(ssim_map.mean())
        mcs.append(cs_map.mean())

        # Downsample images by a factor of 2 for next scale, except at the last scale
        if level < levels - 1:
            img1 = cv.pyrDown(img1)
            img2 = cv.pyrDown(img2)

    mssim = np.array(mssim)
    mcs = np.array(mcs)

    # MS-SSIM calculation: multiply contrast-structure components and luminance component at the last scale
    ms_ssim = np.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
    return ms_ssim


# Update evaluate_Image to include MS-SSIM:
def evaluate_Image(img1, img2):
    PSnr = PSNR(img1, img2)
    SSim = SSIM(img1, img2)
    mse_val = MSE(img1, img2)
    nrmse = NRMSE(img1, img2)
    AD = average_difference(img1, img2)
    nae = NAE(img1, img2)
    mae = MAE(img1, img2)
    vif = VIF(img1, img2)
    fsim = FSIM(img1, img2)
    uqi = UQI(img1, img2)
    ms_ssim = MS_SSIM(img1, img2)
    Eval = [PSnr, SSim, vif, fsim,ms_ssim]
    return Eval




