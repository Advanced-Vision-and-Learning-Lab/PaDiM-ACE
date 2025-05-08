from anomalib.engine import Engine
from anomalib.models import Padim, Dfm, WinClip
from anomalib.data import Folder
from anomalib.post_processing import OneClassPostProcessor, PostProcessor

def main(experiment=None):
    # Initialize datamodules
    mstar = Folder(
        name="MSTAR",
        root="./datasets/soc",
        mask_dir="./ground_truth",
        normal_dir="./train/norm",
        abnormal_dir="./test/anom",
        train_batch_size=16,
        eval_batch_size=16
    )
    mstar.setup()

    hrsid = Folder(
        name="HRSID",
        root="./datasets/HRSID",
        mask_dir="./ground_truth",
        normal_dir="./train/norm",
        abnormal_dir="./test/anom",
        train_batch_size=16,
        eval_batch_size=16
    )
    hrsid.setup()

    ssdd = Folder(
        name="SSDD",
        root="./datasets/SSDD",
        mask_dir="./ground_truth",
        normal_dir="./train/norm",
        abnormal_dir="./test/anom",
        train_batch_size=16,
        eval_batch_size=16
    )
    ssdd.setup()
    
    if experiment is None or experiment == 1:
        """Experiment 1: Compare against different models"""
        print("Comparing different models...")
        # Initialize and setup models
        # post_processor = OneClassPostProcessor(image_sensitivity=0.6, pixel_sensitivity=0.6)
        padim_lace = Padim(loss="lace")
        padim = Padim()
        dfm = Dfm()
        winclip = WinClip()
        winclip.setup(stage="transistor")

        # List of models we want to test (one shot anomalib segmentation models)
        models = {"padim_lace":padim_lace,"padim":padim,"dfm":dfm,"winclip":winclip}

        # List of datasets to test on
        datamodules = [ssdd]

        for i in range(1):
            for datamodule in datamodules:
                for model_name, model in models.items():
                    engine = Engine()
                    engine.fit(model=model, datamodule=datamodule)
                    test_results = engine.test(model=model,
                                            datamodule=datamodule, 
                                            ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
                    f = open(f"{model_name}_{datamodule.name}_results.txt","a")
                    f.write(f"Run {i+1}: {str(test_results)}\n")
                    f.write(f"model_path: {str(engine.trainer.checkpoint_callback.best_model_path)}\n")
                    f.close()

    if experiment is None or experiment == 2: 
        """Experiment 2: Investigate different backbones"""
        print("Testing different backbones...")
        
        #List of backbones to test (two CNNs and two transformers)
        # backbones = ["resnet18", "convnextv2_tiny", "swinv2_tiny_window8_256", "mobilevit_xs"]
        backbones = ["mobilevit_xs"]
        post_processor = OneClassPostProcessor(image_sensitivity=0.6, pixel_sensitivity=0.6)
        for i in range(2):
            for backbone in backbones:
                if backbone == "resnet18":
                    model = Padim(loss="lace", 
                                  post_processor=post_processor)
                elif backbone == "convnextv2_tiny":
                    model = Padim(backbone=backbone, 
                                  layers=["stages.0", "stages.1", "stages.2"], 
                                  n_features=100, 
                                  loss="lace", 
                                  post_processor=post_processor)
                elif backbone == "swinv2_tiny_window8_256":
                    model = Padim(backbone=backbone, 
                                  layers=["layers.0", "layers.1", "layers.2"], 
                                  n_features=100, 
                                  loss="lace", 
                                  post_processor=post_processor)
                elif backbone == "mobilevit_xs":
                    model = Padim(backbone="mobilevit_xs", 
                                  layers=["stages.0", "stages.1", "stages.2"], 
                                  n_features=100, 
                                  loss="lace",
                                  post_processor=post_processor)

                engine = Engine()
                engine.fit(model=model, datamodule=hrsid)
                test_results = engine.test(model=model,
                                            datamodule=hrsid, 
                                            ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
                f = open(f"padim_lace_{backbone}_results.txt","a")
                f.write(f"Run {i+2}: {str(test_results)}\n")
                f.write(f"model_path: {str(engine.trainer.checkpoint_callback.best_model_path)}\n")
                f.close()

    if experiment is None or experiment == 3:
        """Experiment 3: Investigate different covariance types"""
        print("Testing different cov mat types...")
        cov_types = ["full", "diagonal", "isotropic"]
        post_processor = OneClassPostProcessor(image_sensitivity=0.6, pixel_sensitivity=0.6)
        for i in range(1):
            for cov_type in cov_types:
                model = Padim(loss="lace", cov_type=cov_type, post_processor=post_processor)

                engine = Engine()
                engine.fit(model=model, datamodule=hrsid)
                test_results = engine.test(model=model,
                                        datamodule=hrsid,
                                        ckpt_path = engine.trainer.checkpoint_callback.best_model_path)
                f = open(f"padim_lace_{cov_type}_results.txt","a")
                f.write(f"Run {i+1}: {str(test_results)}\n")
                f.write(f"model_path: {str(engine.trainer.checkpoint_callback.best_model_path)}\n")
                f.close()
    
    if experiment == 4:
        """Experiment 4: Investigate isotropic aggregation"""
        agg_types = ["mean_diagonal", "mean_full", "determinant", "trace"]

        for i in range(3):
            post_processor = OneClassPostProcessor(image_sensitivity=0.6, pixel_sensitivity=0.6)
            model = Padim(loss="lace", post_processor=post_processor, cov_type="isotropic")
            engine = Engine()
            engine.fit(model=model, datamodule=hrsid)
            test_results = engine.test(model=model,
                                    datamodule=hrsid,
                                    ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
            f = open(f"padim_lace_{agg_types[2]}_results.txt","a")
            f.write(f"Run {i+1}: {str(test_results)}\n")
            f.write(f"model_path: {str(engine.trainer.checkpoint_callback.best_model_path)}\n")
            f.close()
    
    if experiment == 5:
        """Experiment 5: Investigate different aggregation methods"""
        # post_processor = OneClassPostProcessor(image_sensitivity=0.4, pixel_sensitivity=0.4)
        for i in range(3):
            model = Padim(loss="lace")
            engine = Engine()
            engine.fit(model=model, datamodule=hrsid)
            test_results = engine.test(model=model,
                                    datamodule=hrsid,
                                    ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
            f = open(f"padim_lace_sig_mat(mean difference^2)_results.txt","a")
            f.write(f"Run {i+1}: {str(test_results)}\n")
            f.write(f"model_path: {str(engine.trainer.checkpoint_callback.best_model_path)}\n")
            f.close()

if __name__ == "__main__":
    main(experiment=5)
    
















