import ncem
import numpy as np
import seaborn as sns
import squidpy as sq
from scipy.stats import ttest_rel, ttest_ind
import scanpy as sc
import pandas as pd
from ncem.interpretation import InterpreterInteraction
from ncem.data import get_data_custom, customLoader
sns.set_palette("colorblind")
from sklearn.metrics import r2_score
import os
from tqdm import tqdm

for file in tqdm(os.listdir("data")):
    # example with 10DPI_1.h5ad
    if file != "10DPI_1.h5ad":
        continue
    try:
        print(file)
        if os.path.exists("tmp/values.txt"):
            os.remove("tmp/values.txt")
        time = file.split(".")[0]
        trainer = ncem.train.TrainModelLinear()
        trainer.init_estim(log_transform=False)
        np.set_printoptions(threshold=np.inf, linewidth=200)

        ad = sc.read_h5ad(os.path.join("filtered_data", file))
        ad.var_names_make_unique()
        ad.uns['spatial'] = ad.obsm['spatial']
        # sc.pp.highly_variable_genes(ad, n_top_genes=5000)
        # ad = ad[:, ad.var['highly_variable']]
        receptor_list_df = pd.read_csv(f"receptor_gene_list/{time}.csv", header=None)
        receptor_list = receptor_list_df[0].tolist()
        non_receptor_list_df = pd.read_csv(f"non_receptor_gene_list/{time}.csv", header=None)
        non_receptor_list = non_receptor_list_df[0].tolist()
        # in ad, keep only receptor_list and non_receptor_list genes
        ad = ad[:, receptor_list + non_receptor_list]
        genes = ad.var_names
        print(len(genes))
        trainer.estimator.data = ncem.data.customLoader(
            adata=ad, cluster='Annotation', patient=None, library_id=None, radius=30
        )
        ncem.data.get_data_custom(interpreter=trainer.estimator)

        ncv = 3
        epochs = 30
        epochs_warmup = 0
        max_steps_per_epoch = 20
        patience = 100
        lr_schedule_min_lr = 1e-10
        lr_schedule_factor = 0.5
        lr_schedule_patience = 50
        val_bs = 16
        max_val_steps_per_epoch = 10
        shuffle_buffer_size = None

        feature_space_id = "standard"
        cond_feature_space_id = "type"

        use_covar_node_label = False
        use_covar_node_position = False
        use_covar_graph_covar = False

        trainer.estimator.split_data_node(
            validation_split=0.1,
            test_split=0.1,
            seed=0
        )

        log_transform = False
        use_domain = True
        scale_node_size=False
        merge_node_types_predefined = True
        covar_selection = []
        output_layer='linear'

        model_class = 'interactions'
        optimizer = 'adam'
        domain_type = 'patient'

        learning_rate = 0.05
        l1 = 0.
        l2 = 0.

        batch_size = 64
        n_eval_nodes = 10

        trainer.estimator.init_model(
            optimizer=optimizer,
            learning_rate=learning_rate,
            n_eval_nodes_per_graph=n_eval_nodes,

            l2_coef=l2,
            l1_coef=l1,
            use_interactions=True,
            use_domain=use_domain,
            scale_node_size=scale_node_size,
            output_layer=output_layer,
        )

        trainer.estimator.train(
            epochs=epochs,
            epochs_warmup=epochs_warmup,
            batch_size=batch_size,
            max_steps_per_epoch=max_steps_per_epoch,
            validation_batch_size=val_bs,
            max_validation_steps=max_val_steps_per_epoch,
            patience=patience,
            lr_schedule_min_lr=lr_schedule_min_lr,
            lr_schedule_factor=lr_schedule_factor,
            lr_schedule_patience=lr_schedule_patience,
            monitor_partition="val",
            monitor_metric="loss",
            shuffle_buffer_size=shuffle_buffer_size,
            early_stopping=True,
            reduce_lr_plateau=True,
        )

        # delete tmp/values.txt if it exists
        if os.path.exists("tmp/values.txt"):
            os.remove("tmp/values.txt")

        trainer.estimator.simulation = False

        split_per_node_type, evaluation_per_node_type, targets = trainer.estimator.evaluate_per_node_type()

        # for cell_type, (true, pred) in targets.items():
        #     r2_values = dict()
        #     gene_names = ad.var_names
        #     # print(len(true))
        #     if len(true) == 0:
        #         continue
        #     for i in range(500):
        #         combined_true = []
        #         combined_pred = []
        #         for j in range(len(pred)):
        #             combined_true.append(true[j].numpy().squeeze()[:, i])
        #             combined_pred.append(pred[j].squeeze()[:, i])
        #         combined_true = np.concatenate(combined_true)
        #         combined_pred = np.concatenate(combined_pred)
        #         r2_values[gene_names[i]] = r2_score(combined_true, combined_pred)
        #     os.makedirs(f"per_gene_results/{time}/", exist_ok=True)
        #     r2_values = pd.DataFrame.from_dict(r2_values, orient='index', columns=['R2'])
        #     # print(r2_values.max())
        #     r2_values.to_csv(f"per_gene_results/{time}/{cell_type}.csv")

        file = open("tmp/values.txt", "r")
        values = file.read()

        values = values.split("\n---------------------------------")
        # print(len(values))
        values = values[:-1]

        # strip whitespace and newlines at beginning and end
        values = [value.strip() for value in values]
        # values = values.strip()

        # print(values[1][:10])

        # exit()

        values_ = values

        for group in values_:
            # split values into first line and rest
            values = group.split("\n", 1)
            cell_type = values[0]
            values = values[1]
            print(cell_type)
            # from values, remove "true: " and "pred: "
            values = values.replace("true: ", "")
            values = values.replace("pred: ", "")

            # values now contains numpy arrays of true and pred
            # split by \n
            values = values.split("\n\n")

            # remove last element, which is empty
            # values = values[:-1]

            # remove ending whitespace
            values = [value.strip() for value in values]

            # first, third, fifth... elements are true
            # second, fourth, sixth... elements are pred
            true = values[0::2]
            pred = values[1::2]
            true_array = []
            pred_array = []
            for i in range(len(true)):
                # convert true to numpy arrays
                x = true[i].strip()[2:-2]
                x_lines = x.split("\n ")
                x_lines = [line.replace("[", "").replace("]", "") for line in x_lines]
                x_lines = [line.split(" ") for line in x_lines]
                x_lines = [[float(s) for s in line if s != ""] for line in x_lines]
                x_lines = np.array(x_lines)
                true_array.append(x_lines)
                # convert pred to numpy arrays
                y = pred[i].strip()[2:-2]
                y_lines = y.split("\n ")
                y_lines = [line.replace("[", "").replace("]", "") for line in y_lines]
                y_lines = [line.split(" ") for line in y_lines]
                y_lines = [[float(s) for s in line if s != ""] for line in y_lines]
                y_lines = np.array(y_lines)
                pred_array.append(y_lines)
            true_array = np.array(true_array)
            pred_array = np.array(pred_array)

            # concatenate true_array along axis 0
            # concatenate pred_array along axis 0
            true_array = np.concatenate(true_array, axis=0)
            pred_array = np.concatenate(pred_array, axis=0)

            # find columns where all values are 0 in true_array

            # cols = []
            # for i in range(true_array.shape[1]):
            #     if np.all(true_array[:, i] == 0):
            #         cols.append(i)

            # # remove these columns from true_array and pred_array
            # true_array = np.delete(true_array, cols, axis=1)
            # pred_array = np.delete(pred_array, cols, axis=1)

            # true_array and pred_array are now numpy arrays
            # calculate r2 values along columns
            r2_values = dict()
            # r2_values = r2_score(true_array, pred_array)
            assert true_array.shape[1] == len(genes)
            for i in range(true_array.shape[1]):
                r2_values[genes[i]] = r2_score(true_array[:, i], pred_array[:, i])
            os.makedirs(f"per_gene_results/{time}/", exist_ok=True)
            r2_values = pd.DataFrame.from_dict(r2_values, orient='index', columns=['R2'])
            r2_values.to_csv(f"per_gene_results/{time}/{cell_type}.csv")
    except Exception as e:
        print(e)
        continue