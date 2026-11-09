import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import pylab
from prettytable import PrettyTable
import cv2 as cv
import pylab
import scipy.stats as statsci
from scipy.stats import stats
import scipy.stats as statsci
warnings.filterwarnings("ignore")


def Plot_KFold_Reconstruction():
    eval = np.load('Eval_all_KFold.npy', allow_pickle=True)
    Terms = ['Peak Signal-to-Noise Ratio', 'Structural Similarity Index', 'Visual Information Fidelity',
             'Feature Similarity Index','Multi - Scale SSIM','MSE', 'NRMSE','Average Difference']
    Classifiers = ['Unet', 'ResUnet','DenseUnet', 'TransUnet', 'SVRUNet++-TNLF']
    Batch_size = ['1', '2', '3', '4', '5']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :5, :, Graph_Terms[j]]

            # Method Comparision
            Mtd_Graph = np.array([Graph[:, 0], Graph[:, 1], Graph[:, 2], Graph[:, 3], Graph[:, 4]])
            colors = ['#E366DA', '#5BE34D', '#00129A', '#8586FF', 'black']
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.15
            index = np.arange(len(Batch_size))
            for i, Classifier in enumerate(Classifiers):
                bars = ax.bar(index + i * bar_width, Mtd_Graph[i], bar_width, label=Classifiers[i], color=colors[i])
                ax.bar_label(
                    bars,
                    labels=[f'{val:.3f}' for val in Mtd_Graph[i]],
                    padding=3,
                    rotation=90,
                    fontsize=9,
                    color='black'
                )
            ax.set_xlabel('Images →', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold',
                          color='#35530a')
            ax.set_xticks(index + bar_width * (len(Classifiers) / 2 - 0.5))
            ax.set_xticklabels(Batch_size)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Classifiers, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold'})
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            path = "./Result/ImageConstruction%s_mtd.png" % (Terms[Graph_Terms[j]])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Images vs ' + Terms[Graph_Terms[j]])
            plt.savefig(path)
            # plt.show()
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def STATS(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def Plot_Conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['Terms', 'PSO-XADSPP-Yv9', 'WOA-XADSPP-Yv9', 'SCO-XADSPP-Yv9', 'GKSO-XADSPP-Yv9', 'FRGSO-XADSPP-Yv9']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((Fitness.shape[1], 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = STATS(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report',
          ' --------------------------------------------------')

    print(Table)

    # ANOVA test across all algorithms
    f_val, p_val = stats.f_oneway(*[Fitness[j, :] for j in range(len(Algorithm) - 1)])
    print(f"One-way ANOVA: F = {f_val:.4f}, p = {p_val:.4e}")
    if p_val < 0.05:
        print("→ Significant difference detected between algorithms (p < 0.05)")
    else:
        print("→ No significant difference detected (p ≥ 0.05)")

    # Pairwise t-tests between algorithms
    ttest_table = PrettyTable()
    ttest_table.field_names = ["Algorithm Pair", "t-value", "p-value"]

    for i in range(len(Algorithm) - 1):
        for j in range(i + 1, len(Algorithm) - 1):
            t_stat, p_val = stats.ttest_ind(Fitness[i, :], Fitness[j, :])
            ttest_table.add_row(
                [f"{Algorithm[i + 1]} vs {Algorithm[j + 1]}", round(t_stat, 4), f"{p_val:.4e}"])

    print(ttest_table)

    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Result/Convergence.png")
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.show()

def Plot_Seg_Results():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Image = ['Worst', 'Best', 'Mean', 'Median']  # , 'STD']

    Full = ['TERMS', 'PSO-XADSPP-Yv9', 'WOA-XADSPP-Yv9', 'SCO-XADSPP-Yv9', 'GKSO-XADSPP-Yv9', 'FRGSO-XADSPP-Yv9',
            'DETR', 'TransYOLO', 'ViT-Deformable DETR', 'XDSPP-Yv9', 'FRGSO-XADSPP-Yv9']
    Graph_terms = [0, 1, 2, 3, 5, 7]
    stats = np.zeros((len(Graph_terms), Eval.shape[-3] + 1, 5))  # (METRICS, ALGORITHM, STATS)
    Eval_all = Eval
    for k in range(len(Graph_terms)):
        for r in range(5):
            for j in range(Eval_all.shape[-3] + 1):
                if j < Eval_all.shape[-3]:
                    stats[k, j, 0] = np.max(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 1] = np.min(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 2] = np.mean(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 3] = np.median(Eval_all[r, j][:, Graph_terms[k] + 4])
                    stats[k, j, 4] = np.std(Eval_all[r, j][:, Graph_terms[k] + 4])

            alg_prop = stats[k, 4, :]
            stats[k, 9, :] = alg_prop

            Alg_Val = stats[k, :5, :-1]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(Alg_Val.shape[-1])
            X = x
            colour = ['#e50000', '#0504aa', '#76cd26',  '#b0054b',  'k']
            bar_width = 0.15
            offsets = [0.00, 0.15, 0.30, 0.45, 0.60]
            bars = []
            for i in range(5):
                bar_group = ax.bar(X + offsets[i], Alg_Val[i, :], color=colour[i], edgecolor='w', width=bar_width)
                bars.append(bar_group)
                highlight_map = {
                    0: [0, 4],  # BEST → 1st and 5th
                    1: [1],  # WORST → 2nd
                    2: [2],  # MEAN → 3rd
                    3: [3],  # MEDIAN → 4th
                }
                for j in range(len(X)):  # Loop over statistics (BEST, WORST, etc.)
                    bar_height = Alg_Val[i, j]
                    x_center = X[j] + offsets[i]
                    y_top = bar_height * 1.20
                    label_text = Full[i + 1]
                    if i in highlight_map.get(j, []):  # Check if this algorithm should be labeled in this stat

                        ax.text(x_center + 0.125, y_top + 1.25, label_text, ha='center', va='bottom',  # 1.5
                                fontsize=12, fontweight='bold')
                        ax.hlines(y=y_top, xmin=x_center, xmax=x_center + 0.21,
                                  color=colour[i], linewidth=2.5)
                        ax.vlines(x=x_center, ymin=bar_height, ymax=y_top,
                                  color='k', linestyle='dotted', linewidth=1.5)  # color=colour[i]

            ax.set_xticks(X + (bar_width * 2))
            ax.set_xticklabels(Image, fontsize=10, fontweight='bold')
            ax.set_xlabel('Statistical Analysis', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
            plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
            ax.set_axisbelow(True)
            plt.yticks(np.arange(0, np.max(Alg_Val) + 7, (np.max(Alg_Val) + 2) / 10))
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)
            plt.tight_layout()
            plt.savefig(f"./Result/Seg_{Terms[Graph_terms[k]]}_Alg.png")
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(
                'Algorithm comparision of Statistical Analysis vs ' + Terms[
                    Graph_terms[k]] + ' Algorithm Comparision')
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            Mtd_Val = stats[k, 5:, :-1]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(Mtd_Val.shape[-1])
            X = x
            colour = ['#00BFC4', '#FF9224', '#B5D334', '#9370DB', '#FF6347']
            bar_width = 0.15
            offsets = [0.00, 0.15, 0.30, 0.45, 0.60]
            bars = []
            for i in range(5):
                bar_group = ax.bar(X + offsets[i], Mtd_Val[i, :], color=colour[i], edgecolor='w', width=bar_width)
                bars.append(bar_group)
                highlight_map = {
                    0: [0, 4],  # BEST → 1st and 5th
                    1: [1],  # WORST → 2nd
                    2: [2],  # MEAN → 3rd
                    3: [3],  # MEDIAN → 4th
                }
                for j in range(len(X)):  # Loop over statistics (BEST, WORST, etc.)
                    bar_height = Mtd_Val[i, j]
                    x_center = X[j] + offsets[i]
                    # y_top = bar_height + 10  # 1.5
                    y_top = bar_height * 1.20
                    label_text = Full[i + 6]
                    if i in highlight_map.get(j, []):  # Check if this algorithm should be labeled in this stat

                        ax.text(x_center + 0.125, y_top + 1.25, label_text, ha='center', va='bottom',  # 1.5
                                fontsize=12, fontweight='bold')
                        ax.hlines(y=y_top, xmin=x_center, xmax=x_center + 0.21,
                                  color=colour[i], linewidth=2.5)
                        ax.vlines(x=x_center, ymin=bar_height, ymax=y_top,
                                  color='k', linestyle='dotted', linewidth=1.5)  # color=colour[i]

            ax.set_xticks(X + (bar_width * 2))
            ax.set_xticklabels(Image, fontsize=10, fontweight='bold')
            ax.set_xlabel('Statistical Analysis', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
            plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
            ax.set_axisbelow(True)
            plt.yticks(np.arange(0, np.max(Mtd_Val) + 7, (np.max(Mtd_Val) + 2) / 10))
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)
            plt.tight_layout()
            plt.savefig(f"./Result/Seg_{Terms[Graph_terms[k]]}_Mtd.png")
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(
                'Method comparision of Statistical Analysis vs ' + Terms[
                    Graph_terms[k]] + ' Method Comparision')
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def Plot_Seg_Table():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity',
             'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Full = ['TERMS', 'Unet', 'ResUnet', 'DenseUnet', 'TransUnet', 'SVRUNet++-TNLF']
    Graph_terms = [0, 1, 2]  # metrics to plot (Dice, IOU, Accuracy for example)

    for n in range(1):
        Eval_all = Eval[n]
        stats_data = np.zeros((len(Graph_terms), Eval_all.shape[-3], 5))  # max, min, mean, median, std
        ci_low_all = np.zeros((len(Graph_terms), Eval_all.shape[-3]))
        ci_high_all = np.zeros((len(Graph_terms), Eval_all.shape[-3]))
        p_values_all = np.ones((len(Graph_terms), Eval_all.shape[-3]))

        for k in range(len(Graph_terms)):
            for j in range(Eval_all.shape[-3]):
                data = Eval_all[j][:, Graph_terms[k] + 4]
                stats_data[k, j, 0] = np.max(data)
                stats_data[k, j, 1] = np.min(data)
                stats_data[k, j, 2] = np.mean(data)
                stats_data[k, j, 3] = np.median(data)
                stats_data[k, j, 4] = np.std(data)
                n_samples = len(data)
                se = statsci.sem(data)
                ci_low, ci_high = statsci.t.interval(0.95, n_samples - 1, loc=np.mean(data), scale=se)
                ci_low_all[k, j] = ci_low
                ci_high_all[k, j] = ci_high

            for j in range(1, Eval_all.shape[-3]):
                t_stat, p_val = statsci.ttest_ind(
                    Eval_all[0][:, Graph_terms[k] + 4],  # baseline (Unet)
                    Eval_all[j][:, Graph_terms[k] + 4],
                    equal_var=False
                )
                p_values_all[k, j] = p_val

            Table = PrettyTable()
            Table.add_column(Full[0], ['MAX', 'MIN', 'MEAN', 'MEDIAN', 'STD', '95% CI (Low)', '95% CI (High)', 'p-value'])

            for l in range(len(Full) - 1):
                model_stats = [
                    f"{stats_data[k, l, 0]:.4f}",  # max
                    f"{stats_data[k, l, 1]:.4f}",  # min
                    f"{stats_data[k, l, 2]:.4f}",  # mean
                    f"{stats_data[k, l, 3]:.4f}",  # median
                    f"{stats_data[k, l, 4]:.4f}",  # std
                    f"{ci_low_all[k, l]:.4f}",     # 95% CI low
                    f"{ci_high_all[k, l]:.4f}",    # 95% CI high
                    f"{p_values_all[k, l]:.4e}"    # p-value (scientific notation)
                ]
                Table.add_column(Full[l + 1], model_stats)

            print(f" ---------------------------------------Statistical Comparison for {Terms[Graph_terms[k]]}--------------------------------------- ")
            print(Table)



def plot_Seg_Loss():
    eval = np.load('Eval_all_loss.npy', allow_pickle=True)
    Terms = ['Peak Signal-to-Noise Ratio', 'Structural Similarity Index', 'Visual Information Fidelity',
             'Feature Similarity Index','Multi - Scale SSIM','MSE', 'NRMSE','Average Difference']
    Classifiers = ['Unet', 'ResUnet','DenseUnet', 'TransUnet', 'SVRUNet++-TNLF']
    Loss = ['SSIM', 'Perceptual VGG', 'Perceptual LPIPS', 'TNLF']
    Graph_Terms = [0, 1, 2, 4, 5, 6, 7]
    for j in range(len(Graph_Terms)):
        Graph = eval[:5, :, Graph_Terms[j]]

        # Method Comparision

        Value = Graph
        bar_width = 0.31
        x = np.arange(len(Loss))
        plt.figure(figsize=(8, 6))
        bars1 = plt.bar(x - bar_width, Value[:4, 2], width=bar_width, label=Classifiers[2], color='#00129A')
        bars2 = plt.bar(x, Value[:4, 3], width=bar_width, label=Classifiers[3], color='#8586FF')
        bars3 = plt.bar(x + bar_width, Value[:4, 4], width=bar_width, label=Classifiers[4], color='black')
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                yval = bar.get_height()
                if Terms[Graph_Terms[j]] == 'Feature Similarity Index':
                    plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.005, f"{yval:.3f}", ha='center', va='bottom',
                         fontsize=9)
                else:
                    plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.005, f"{yval:.3f}", ha='center', va='bottom',
                             fontsize=9)
        plt.xlabel('Loss Function →', fontsize=12, fontweight='bold', color='black')
        plt.ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold', color='#35530a')
        plt.xticks(x, Loss)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
        plt.box(True)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=12, frameon=False)
        plt.tight_layout()
        path = "./Result/%s_Loss.png" %(Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Loss Function vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def plot_SPE_tables():
    Eval = np.load('Eval_all_SPE.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_terms = [0, 1, 2, 3, 5, 12]

    Full = ['Steps Per Epoch',  'PSO-XADSPP-Yv9', 'WOA-XADSPP-Yv9', 'SCO-XADSPP-Yv9', 'GKSO-XADSPP-Yv9', 'FRGSO-XADSPP-Yv9','DETR', 'TransYOLO', 'ViT-Deformable DETR',
            'XDSPP-Yv9', 'FRGSO-XADSPP-Yv9']

    Losses = ['Mse loss', 'Categorical loss', 'Binary loss', 'Center loss', 'SparseCategoricalCrossentropy class']
    SPE = ['50', '100', '150', '200', '250']
    Table_stats = np.zeros((Eval.shape[-4], len(Graph_terms), Eval.shape[-3], 5))
    for p in range(Eval.shape[-4]):
        Eval_all = Eval[p]
        for i in range(len(Graph_terms)):
            for j in range(Eval_all.shape[-3]):
                if j < Eval_all.shape[-3]:
                    Table_stats[p, i, j, 0] = np.max(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 1] = np.min(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 2] = np.mean(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 3] = np.median(Eval_all[j][:, Graph_terms[i] + 4])
                    Table_stats[p, i, j, 4] = np.std(Eval_all[j][:, Graph_terms[i] + 4])

    alg_prop = Table_stats[:, :, 4, :]
    Table_stats[:, :, 9, :] = alg_prop
    for t in range(len(Graph_terms)):

        Table = PrettyTable()
        Table.add_column(Full[0], SPE)
        for k in range(len(Full) - 6):
            Table.add_column(Full[k + 1], Table_stats[:, t, k, 2])
        print('-------------------------------------------------- Mean Algorithm Comparison',
              Terms[Graph_terms[t]],
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Full[0], SPE)
        for k in range(5, 10):
            Table.add_column(Full[k + 1], Table_stats[:, t, k, 2])
        print('-------------------------------------------------- Mean Classifier Comparison',
              Terms[Graph_terms[t]],
              '--------------------------------------------------')
        print(Table)


def Segmentation_Images():
    Original = np.load('Images.npy', allow_pickle=True)
    Labels = np.load('GT_Img.npy', allow_pickle=True)
    segmented = np.load('Seg_img.npy', allow_pickle=True)
    Calssifier = ['YoloV3', 'YoloV5', 'YoloV8', 'EADSPP-YoloV9', 'Proposed']
    Images = [1120, 1344, 1725, 3260, 4977]
    for i in range(len(Images)):
        print(i, len(Images), Images[i])
        Origin = Original[Images[i]]
        Label = Labels[Images[i]]
        Seg_1 = segmented[Images[i]]
        for j in range(1):
            Orig_1 = cv.resize(Seg_1[j], (512, 512))
            Orig_2 = cv.resize(Seg_1[j + 1], (512, 512))
            Orig_3 = cv.resize(Seg_1[j + 2], (512, 512))
            Orig_4 = cv.resize(Seg_1[j + 3], (512, 512))
            Orig_5 = cv.resize(Seg_1[j + 4], (512, 512))
            plt.suptitle('Detected Images from the Dataset', fontsize=20)

            plt.subplot(3, 3, 1).axis('off')
            plt.imshow(Origin)
            plt.title('Original', fontsize=10)

            plt.subplot(3, 3, 3).axis('off')
            plt.imshow(Label)
            plt.title('Ground Truth', fontsize=10)

            plt.subplot(3, 3, 4).axis('off')
            plt.imshow(Orig_1)
            plt.title(Calssifier[0], fontsize=10)

            plt.subplot(3, 3, 5).axis('off')
            plt.imshow(Orig_2)
            plt.title(Calssifier[1], fontsize=10)

            plt.subplot(3, 3, 6).axis('off')
            plt.imshow(Orig_3)
            plt.title(Calssifier[2], fontsize=10)

            plt.subplot(3, 3, 7).axis('off')
            plt.imshow(Orig_4)
            plt.title(Calssifier[3], fontsize=10)

            plt.subplot(3, 3, 9).axis('off')
            plt.imshow(Orig_5)
            plt.title(Calssifier[4], fontsize=10)

            path = "./Result/Image_results/_Compared_Images_%s.png" % (i + 1)
            plt.savefig(path)
            plt.show()

            cv.imwrite('./Result/Image_results/Original_image_' + str(i + 1) + '.png',
                       Origin)
            cv.imwrite(
                './Result/Image_results/Segm_img_' + str(Calssifier[0]) + '_' + str(
                    i + 1) + '.png', Orig_1)
            cv.imwrite(
                './Result/Image_results/Segm_img_' + str(Calssifier[1]) + '_' + str(
                    i + 1) + '.png', Orig_2)
            cv.imwrite(
                './Result/Image_results/Segm_img_' + str(Calssifier[2]) + '_' + str(
                    i + 1) + '.png', Orig_3)
            cv.imwrite(
                './Result/Image_results/Segm_img_' + str(Calssifier[3]) + '_' + str(
                    i + 1) + '.png', Orig_4)
            cv.imwrite(
                './Result/Image_results/Segm_img_' + str(Calssifier[4]) + '_' + str(
                    i + 1) + '.png', Orig_5)
            cv.imwrite('./Result/Image_results/Ground_Truth_' + str(i + 1) + '.png',
                       Label)


def gradcam_images():
    Orignal_Image = np.load('Images.npy', allow_pickle=True)
    Gradcam_Image = np.load('XAI_images.npy', allow_pickle=True)
    overlay_Image = np.load('overlays.npy', allow_pickle=True)
    Zoomed_Nodules = np.load('Zoomed_Images.npy', allow_pickle=True)
    Images = [0, 43,100,105, 120,123, 149,  279, 1598, 2380]
    for i in range(len(Images)):
        print(i, len(Images))
        Orignal = cv.resize(Orignal_Image[Images[i]], (512, 512))
        Gradcam = cv.resize(Gradcam_Image[Images[i]], (512, 512))
        overlay = cv.resize(overlay_Image[Images[i]], (512, 512))
        Zoomed = cv.resize(Zoomed_Nodules[Images[i]], (512, 512))
        cv.imshow('Orignal Image', Orignal)
        cv.imshow('Gradcam Image', Gradcam)
        cv.imshow('Overlay', overlay)
        cv.imshow('zoomed', Zoomed)
        cv.waitKey(0)
        cv.imwrite('./Result/Gradcam/Orignal Images_' + str(i + 1) + '.png', Orignal)
        cv.imwrite('./Result/Gradcam/Gradcam Images_' + str(i + 1) + '.png', Gradcam)
        cv.imwrite('./Result/Gradcam/Overlay Images_' + str(i + 1) + '.png', overlay)
        cv.imwrite('./Result/Gradcam/ZoomedImage_' + str(i + 1) + '.png', Zoomed)

def to_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
        img = (img * 255) if img.max() <= 1.0 else img
        img = img.astype(np.uint8)
    return img

def Reconstructed_Images():
    Original = np.load('Images.npy', allow_pickle=True)
    reconstructed = np.load('Reconstructed_Image_CMP_.npy', allow_pickle=True)
    Calssifier = ['Unet', 'ResUnet','DenseUnet', 'TransUnet', 'Proposed']
    Images = [1120, 1344, 1725, 3260, 4977]
    for i in range(len(Images)):
        print(i, len(Images), Images[i])
        Origin = Original[Images[i]]
        Seg = reconstructed[Images[i]]
        for j in range(1):
            Orig_1 = cv.resize(Seg[j], (512, 512))
            Orig_2 = cv.resize(Seg[j + 1], (512, 512))
            Orig_3 = cv.resize(Seg[j + 2], (512, 512))
            Orig_4 = cv.resize(Seg[j + 3], (512, 512))
            Orig_5 = cv.resize(Seg[j + 4], (512, 512))
            plt.suptitle('Detected Images ', fontsize=20)

            plt.subplot(2, 3, 6).axis('off')
            plt.imshow(Origin)
            plt.title('Original', fontsize=10)

            plt.subplot(2, 3, 1).axis('off')
            plt.imshow(Orig_1)
            plt.title(Calssifier[0], fontsize=10)

            plt.subplot(2, 3, 2).axis('off')
            plt.imshow(Orig_2)
            plt.title(Calssifier[1], fontsize=10)

            plt.subplot(2, 3, 3).axis('off')
            plt.imshow(Orig_3)
            plt.title(Calssifier[2], fontsize=10)

            plt.subplot(2, 3, 4).axis('off')
            plt.imshow(Orig_4)
            plt.title(Calssifier[3], fontsize=10)

            plt.subplot(2, 3, 5).axis('off')
            plt.imshow(Orig_5)
            plt.title(Calssifier[4], fontsize=10)

            path = "./Result/Reconstructed/_Compared_Images_%s.png" % (i + 1)
            plt.savefig(path)
            plt.show()

            cv.imwrite('./Result/Reconstructed/Original_image_' + str(i + 1) + '.png',
                       to_uint8(Origin))
            cv.imwrite(
                './Result/Reconstructed/Segm_img_' + str(Calssifier[0]) + '_' + str(
                    i + 1) + '.png', to_uint8(Orig_1))
            cv.imwrite(
                './Result/Reconstructed/Segm_img_' + str(Calssifier[1]) + '_' + str(
                    i + 1) + '.png', to_uint8(Orig_2))
            cv.imwrite(
                './Result/Reconstructed/Segm_img_' + str(Calssifier[2]) + '_' + str(
                    i + 1) + '.png', to_uint8(Orig_3))
            cv.imwrite(
                './Result/Reconstructed/Segm_img_' + str(Calssifier[3]) + '_' + str(
                    i + 1) + '.png', to_uint8(Orig_4))
            cv.imwrite(
                './Result/Reconstructed/Segm_img_' + str(Calssifier[4]) + '_' + str(
                    i + 1) + '.png', to_uint8(Orig_5))



if __name__ == '__main__':
    Plot_Conv()
    plot_Seg_Loss()
    Plot_KFold_Reconstruction()
    Plot_Seg_Results()
    plot_SPE_tables()
    Plot_Seg_Table()
    Segmentation_Images()
    gradcam_images()
    Reconstructed_Images()