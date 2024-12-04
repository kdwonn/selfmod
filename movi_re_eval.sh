poetry run ocl_eval ++trainer.devices=[2] train_config_path=reprod_movi/movi_c/usual-breeze-334/config +evaluation=projects/bridging/metrics_movi_c ++trainer.precision=16-mixed
poetry run ocl_eval ++trainer.devices=[2] train_config_path=reprod_movi/movi_c/zany-mountain-333/config +evaluation=projects/bridging/metrics_movi_c ++trainer.precision=16-mixed
poetry run ocl_eval ++trainer.devices=[2] train_config_path=reprod_movi/movi_e/eager-cloud-335/config +evaluation=projects/bridging/metrics_movi_e ++trainer.precision=16-mixed
poetry run ocl_eval ++trainer.devices=[2] train_config_path=reprod_movi/movi_e/happy-sound-332/config +evaluation=projects/bridging/metrics_movi_e ++trainer.precision=16-mixed

# poetry run ocl_eval ++trainer.devices=[2] train_config_path=ours_movi/movi_c/dulcet-violet-327/config +evaluation=slot_dict/metrics_movi_c_ia3 ++trainer.precision=16-mixed
# poetry run ocl_eval ++trainer.devices=[2] train_config_path=ours_movi/movi_c/eternal-night-351/config +evaluation=slot_dict/metrics_movi_c_ia3 ++trainer.precision=16-mixed
# poetry run ocl_eval ++trainer.devices=[2] train_config_path=ours_movi/movi_e/good-wood-350/config +evaluation=slot_dict/metrics_movi_e_ia3 ++trainer.precision=16-mixed
# poetry run ocl_eval ++trainer.devices=[2] train_config_path=ours_movi/movi_e/icy-frost-319/config +evaluation=slot_dict/metrics_movi_e_ia3 ++trainer.precision=16-mixed


