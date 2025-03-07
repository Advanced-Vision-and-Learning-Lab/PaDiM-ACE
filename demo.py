
import numpy as np
import torch
import pdb
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from anomalib.engine import Engine
from anomalib.models import Padim, Dfm, WinClip
from anomalib.data import Folder

torch.cuda.empty_cache()

mstar = Folder(
    name="mstar",
    root="./datasets/soc",
    mask_dir="./ground_truth",
    normal_dir="./train/norm",
    abnormal_dir="./test/anom",
    train_batch_size=8,
    eval_batch_size=8
)
mstar.setup()

model = Padim(loss="lace", n_features=256)
engine = Engine()
engine.fit(model=model, datamodule=mstar)

# predictions = engine.predict(datamodule=datamodule)
# gt_labels = torch.cat([pred.gt_label for pred in predictions])
# pred_labels = torch.cat([pred.pred_label for pred in predictions])

# ap_score = average_precision_score(gt_labels,pred_labels)
# print(ap_score)

# precision, recall, _ = precision_recall_curve(gt_labels, pred_labels)
# pdb.set_trace()
# plt.plot(recall, precision)
# plt.xlabel("Recall")
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.savefig("precision-recall-curve.png")

test_results = engine.test(model=model,
                           datamodule=mstar, 
                           ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
print(test_results)


#     predictions = engine.predict(datamodule=datamodule)
#     gt_labels = torch.cat([pred.gt_label for pred in predictions])
#     pred_labels = torch.cat([pred.pred_label for pred in predictions])

#     test_results = engine.test(model=model,
#                            datamodule=datamodule, 
#                            ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
    
#     ap_score = average_precision_score(gt_labels,pred_labels)
#     print("Average precision score:" + str(ap_score))
#     print(test_results)

# def main():
#     # Initialize datamodules
#     mstar = Folder(
#         name="MSTAR",
#         root="./datasets/soc",
#         mask_dir="./ground_truth",
#         normal_dir="./train/norm",
#         abnormal_dir="./test/anom",
#         train_batch_size=16,
#         eval_batch_size=16
#     )
#     mstar.setup()

#     hrsid = Folder(
#         name="HRSID",
#         root="./datasets/HRSID",
#         mask_dir="./ground_truth",
#         normal_dir="./train/norm",
#         abnormal_dir="./test/anom",
#         train_batch_size=16,
#         eval_batch_size=16
#     )
#     hrsid.setup()

#     ssdd = Folder(
#         name="SSDD",
#         root="./datasets/SSDD",
#         mask_dir="./ground_truth",
#         normal_dir="./train/norm",
#         abnormal_dir="./test/anom",
#         train_batch_size=16,
#         eval_batch_size=16
#     )
#     ssdd.setup()

#     ########Experiment 1: Compare against different models#########

#     # Initialize and setup models
#     padim_lace = Padim(loss="lace")
#     padim = Padim()
#     dfm = Dfm()
#     winclip = WinClip()
#     model.setup(stage="transistor")

#     # List of models we want to test (one shot anomalib segmentation models)
#     models = {"padim_lace":padim_lace, "padim":padim, "dfm":dfm, "winclip":winclip}

#     # List of datasets to test on
#     datamodules = [mstar,hrsid,ssdd]

#     for i in range(3):
#         for model_name, model in models.items():
#             for datamodule in datamodules:
#                 engine = Engine()
#                 engine.fit(model=model, datamodule=datamodule)
#                 test_results = engine.test(model=model,
#                                         datamodule=datamodule, 
#                                         ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
#                 f = open(f"{model_name}_{datamodule.name}_results.txt","a")
#                 f.write(f"Run {i+1}: {str(test_results)}\n")
#                 f.close()
    
#     #######Experiment 2: Investigate different backbones#########

#     #List of backbones to test (two CNNs and two transformers)
#     backbones = ["resnet18", "convnextv2_tiny", "swinv2_tiny_window8_256", "mobilevit_xs"]
#     backbones = ["mobilevit_xs"]

#     for i in range(3):
#         for backbone in backbones:
#             if backbone == "resnet18":
#                 model = Padim(loss="lace")
#             elif backbone == "convnextv2_tiny":
#                 model = Padim(backbone=backbone, layers=["stages.0", "stages.1", "stages.2"], n_features=100, loss="lace")
#             elif backbone == "swinv2_tiny_window8_256":
#                 model = Padim(backbone=backbone, layers=["layers.0", "layers.1", "layers.2"], n_features=100, loss="lace")
#             elif backbone == "mobilevit_xs":
#                 model = Padim(backbone="mobilevit_xs", layers=["stages.0", "stages.1", "stages.2"], n_features=100, loss="lace")
    
#             engine = Engine()
#             engine.fit(model=model, datamodule=mstar)
#             test_results = engine.test(model=model,
#                                         datamodule=mstar, 
#                                         ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
#             f = open(f"padim_lace_{backbone}_results.txt","a")
#             f.write(f"Run {i+1}: {str(test_results)}\n")
#             f.close()
    
#     ########Experiment 3: Investigate different covariance types#########
#     cov_types = ["full", "diagonal", "isotropic"]

#     for i in range(1):
#         for cov_type in cov_types:
#             model = Padim(loss="lace", cov_type=cov_type)

#             engine = Engine()
#             engine.fit(model=model, datamodule=mstar)
#             test_results = engine.test(model=model,
#                                        datamodule=mstar,
#                                        ckpt_path = engine.trainer.checkpoint_callback.best_model_path)
#             f = open(f"padim_lace_{cov_type}_results.txt","a")
#             f.write(f"Run {i+1}: {str(test_results)}\n")
#             f.close()

# if __name__ == "__main__":
#     main()





















