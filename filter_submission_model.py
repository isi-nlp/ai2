import os

models = f"""
'alphanli_100_include_answers_in_context_0', 
'alphanli_100_cn_10k_standard_0', 
'alphanli_100_cn_10k_include_answers_in_context_0', 
'alphanli_100_standard_42'
physicaliqa_100_cn_10k_standard_42
physicaliqa_100_include_answers_in_context_10061880
physicaliqa_100_cn_10k_include_answers_in_context_0
physicaliqa_100_standard_42
'hellaswag_100_standard_0',
'hellaswag_100_include_answers_in_context_42', 
'hellaswag_100_cn_10k_standard_0', 
'hellaswag_100_cn_10k_include_answers_in_context_0',
'hellaswag_100_cn_10k_embed_all_sep_mean_0', 
'hellaswag_100_embed_all_sep_mean_42'
'socialiqa_100_cn_10k_standard_42', 
'socialiqa_100_cn_10k_include_answers_in_context_0', 
'socialiqa_100_cn_10k_embed_all_sep_mean_10061880',
'socialiqa_100_include_answers_in_context_0', 
'socialiqa_100_embed_all_sep_mean_42',
'socialiqa_100_standard_42
"""

for model in models.split():
    m = model.replace('\'','').replace(',','').replace(' ','')
    task = m.split('_')[0]
    print(m)
    for i in range(4):
        try:
            os.system(f'cp outputs/{m}/checkpoints/_ckpt_epoch_{i}.ckpt {task}_submission_models/{m}.ckpt')
        except:
            pass