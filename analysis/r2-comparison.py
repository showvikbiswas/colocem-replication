import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

for folder in tqdm(os.listdir("results/xgb")):
    if folder != '10DPI_1':
        continue
    print(folder)
    if not os.path.exists(f"per_gene_results/{folder}"):
        print("per_gene_results/" + folder + " does not exist")
        continue
    for filename in tqdm(os.listdir("results/xgb/" + folder)):
        ncem_r2 = []
        xgb_r2 = []
        if filename == 'Unknown.csv':
            continue
        xgb_df = pd.read_csv(f"results/xgb/{folder}/{filename}")
        ncem_df = pd.read_csv(f"per_gene_results/{folder}/{filename}")
        # from ncem_df, remove [ and ] from first column
        ncem_df.iloc[:, 0] = ncem_df.iloc[:, 0].str.replace("[", "")
        ncem_df.iloc[:, 0] = ncem_df.iloc[:, 0].str.replace("]", "")
        # from xgb_df, remove rows where third column is 0
        xgb_df = xgb_df[xgb_df.iloc[:, 2] != 0]
        for i in range(len(xgb_df)):
            gene_name = xgb_df.iloc[i, 0]
            # find row in ncem_df with gene_name, which is in the first column
            ncem_row = ncem_df.loc[ncem_df.iloc[:, 0] == gene_name]
            if len(ncem_row) == 0:
                print("Gene not found in ncem_df: " + gene_name)
                continue
            assert len(ncem_row) == 1
            ncem_r2.append(ncem_row.iloc[0, 1])
            xgb_r2.append(xgb_df.iloc[i, 3])
        # print max of xgb_r2
        if len(xgb_r2) == 0:
            continue
        min_r2 = 0
        max_r2 = 1
        linspace = np.linspace(min_r2, max_r2, 1000)
        ncem_above_r2 = []
        xgb_above_r2 = []
        # for each value in linspace, find number of genes with r2 > value
        for i in linspace:
            ncem_above_r2.append(len([j for j in ncem_r2 if j > i]))
            xgb_above_r2.append(len([j for j in xgb_r2 if j > i]))
    
        plt.yscale('log')
        # increase font size everywhere
        plt.rcParams.update({'font.size': 20})
        plt.plot(linspace, ncem_above_r2, label="NCEM")
        plt.plot(linspace, xgb_above_r2, label="ColocEM")
        plt.legend()
        # plt.xlabel("R2")
        # plt.ylabel("Number of genes above R2")
        # increase x and y label size
        plt.xlabel(r"$R^2$", fontsize=20)
        plt.ylabel(r"Number of genes above $R^2$", fontsize=20)
        # increase x ticks size
        plt.xticks(fontsize=20)
        # increase y ticks size
        plt.yticks(fontsize=20)
        plt.tight_layout()
        os.makedirs(f"comp/cumulative_log_pdf/{folder}/", exist_ok=True)
        plt.savefig(f"comp/cumulative_log_pdf/{folder}/" + filename.split(".")[0] + ".png")
        plt.clf()