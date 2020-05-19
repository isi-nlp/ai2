import os

commands = [
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-cn_all_cs_20k.yaml --random_seed 0 --save_path outputs/roberta-large-physicaliqa-cn_all_cs_20k_rs0\"',
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-cn_all_cs_20k.yaml --random_seed 10061880 --save_path outputs/roberta-large-physicaliqa-cn_all_cs_20k_rs10061880\"',
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc.yaml --random_seed 0 --save_path outputs/roberta-large-physicaliqa-10pc_rs0\"',
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc.yaml --random_seed 10061880 --save_path outputs/roberta-large-physicaliqa-10pc_rs10061880\"',
    'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc.yaml --random_seed 0 --save_path outputs/roberta-large-physicaliqa-10pc-arc1_rs0 --goal_inc_answers true\"',
    'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc.yaml --random_seed 10061880 --save_path outputs/roberta-large-physicaliqa-10pc-arc1_rs10061880 --goal_inc_answers true\"',
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc.yaml --random_seed 0 --save_path outputs/roberta-large-physicaliqa-10pc-arc2_rs0 --embed_all_sep_mean true\"',
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc.yaml --random_seed 10061880 --save_path outputs/roberta-large-physicaliqa-10pc-arc2_rs10061880 --embed_all_sep_mean true\"',
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc-cn_all_cs_20k.yaml --random_seed 0 --save_path outputs/roberta-large-physicaliqa-10pc-cn_all_cs_20k_rs0\"',
    # 'sbatch bin/run_saga.sh \"--config_file configs/physicaliqa-10pc-cn_all_cs_20k.yaml --random_seed 10061880 --save_path outputs/roberta-large-physicaliqa-10pc-cn_all_cs_20k_rs10061880\"',
]

for command in commands:
    os.system(command)