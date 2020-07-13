import os

models = f"""
alphanli_100_include_answers_in_context_0,82.25
alphanli_100_standard_42,83.81
alphanli_100_include_answers_in_context_42,81.4
alphanli_100_standard_0,82.64
alphanli_100_standard_10061880,83.49
alphanli_100_include_answers_in_context_10061880,81.14
alphanli_100_cn_10k_standard_0,84.4
alphanli_100_cn_10k_standard_42,82.77
alphanli_100_cn_10k_standard_10061880,83.49
alphanli_100_cn_10k_include_answers_in_context_0,82.44
alphanli_100_cn_10k_include_answers_in_context_42,81.85
alphanli_100_cn_10k_include_answers_in_context_10061880,81.14
physicaliqa_100_include_answers_in_context_42,74.7
physicaliqa_100_include_answers_in_context_0,76.33
physicaliqa_100_standard_10061880,77.48
physicaliqa_100_standard_0,57.24
physicaliqa_100_standard_42,51.41
physicaliqa_100_include_answers_in_context_10061880,77.64
physicaliqa_100_cn_10k_standard_0,76.5
physicaliqa_100_cn_10k_standard_42,78.29
physicaliqa_100_cn_10k_standard_10061880,78.24
physicaliqa_100_cn_10k_include_answers_in_context_0,77.53
physicaliqa_100_cn_10k_include_answers_in_context_42,76.01
physicaliqa_100_cn_10k_include_answers_in_context_10061880,75.84
socialiqa_100_standard_0,77.02
socialiqa_100_standard_42,77.99
socialiqa_100_standard_10061880,77.89
socialiqa_100_include_answers_in_context_0,76.2
socialiqa_100_include_answers_in_context_42,76.66
socialiqa_100_include_answers_in_context_10061880,77.48
socialiqa_100_cn_10k_standard_0,76.92
socialiqa_100_cn_10k_standard_42,78.1
socialiqa_100_cn_10k_standard_10061880,77.48
socialiqa_100_cn_10k_include_answers_in_context_0,77.23
socialiqa_100_cn_10k_include_answers_in_context_42,76.1
socialiqa_100_cn_10k_include_answers_in_context_10061880,77.53
hellaswag_100_standard_0,82.96
hellaswag_100_standard_42,83.79
hellaswag_100_cn_10k_standard_42,83.85
hellaswag_100_include_answers_in_context_42,83.28
hellaswag_100_cn_10k_standard_0,84.47
hellaswag_100_standard_10061880,83.76
hellaswag_100_cn_10k_include_answers_in_context_10061880,82.13
hellaswag_100_include_answers_in_context_0,82.25
hellaswag_100_include_answers_in_context_10061880,81.8
hellaswag_100_cn_10k_include_answers_in_context_0,82.19
hellaswag_100_cn_10k_standard_10061880,84.08
hellaswag_100_cn_10k_include_answers_in_context_42,81.92
"""

for model in models.split():
    m = model.replace('\'','').split(',')[0].replace(',','').replace(' ','')
    task = m.split('_')[0]
    print(m)
    os.system(f'ls outputs/{m}/checkpoints/')
    try:
        os.system(f'cp outputs/{m}/checkpoints/_ckpt_epoch_3.ckpt {task}_submission_models/{m}.ckpt')
    except:
        pass