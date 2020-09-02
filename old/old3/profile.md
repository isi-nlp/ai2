## Old

Profiler Report

Action              	|  Mean duration (s)	|  Total time (s)
-----------------------------------------------------------------
validation_step_end 	|  0.031212       	|  57.587
on_train_start      	|  0.0005904      	|  0.0005904
on_epoch_start      	|  0.0019959      	|  0.0079837
get_train_batch     	|  0.0060954      	|  98.259
on_batch_start      	|  5.0381e-05     	|  0.81194
training_step_end   	|  0.015408       	|  248.32
model_forward       	|  0.072715       	|  1171.9
model_backward      	|  0.087835       	|  1415.5
on_after_backward   	|  8.34e-06       	|  0.13441
on_batch_end        	|  0.0011959      	|  19.273
optimizer_step      	|  0.067827       	|  68.098
on_epoch_end        	|  5.9487e-05     	|  0.00023795
on_train_end        	|  0.0026074      	|  0.0026074

2020-08-25 16:31:43.538 | INFO     | eval:evaluate:90 - Accuracy score: 0.758
2020-08-25 16:31:44.120 | INFO     | eval:evaluate:103 - 95.0 confidence interval 73.3 and 77.5, average: 75.6

real	62m41.867s
user	52m36.016s
sys	8m11.462s

## New

Profiler Report

Action              	|  Mean duration (s)	|  Total time (s)
-----------------------------------------------------------------
on_validation_epoch_start	|  3.337e-05      	|  0.00016685
validation_step_end 	|  0.07114        	|  130.97
on_validation_epoch_end	|  4.0684e-05     	|  0.00020342
on_train_start      	|  0.00058777     	|  0.00058777
on_epoch_start      	|  0.0010356      	|  0.0041424
on_train_epoch_start	|  2.3121e-05     	|  9.2484e-05
get_train_batch     	|  0.0098724      	|  159.14
on_batch_start      	|  6.3154e-05     	|  1.0178
on_train_batch_start	|  4.4467e-05     	|  0.71662
training_step_end   	|  0.048296       	|  778.33
model_forward       	|  0.10989        	|  1771.0
model_backward      	|  0.22516        	|  3628.7
on_after_backward   	|  6.9533e-06     	|  0.11206
on_batch_end        	|  6.7766e-05     	|  1.0921
on_train_batch_end  	|  0.0013271      	|  21.387
optimizer_step      	|  0.081379       	|  82.03
on_epoch_end        	|  3.6638e-05     	|  0.00014655
on_train_epoch_end  	|  2.0796e-05     	|  8.3183e-05
on_train_end        	|  0.00053751     	|  0.00053751

2020-08-25 15:22:23.178 | INFO     | eval:evaluate:90 - Accuracy score: 0.767
2020-08-25 15:22:23.635 | INFO     | eval:evaluate:103 - 95.0 confidence interval 74.4 and 78.3, average: 76.5

real	103m5.831s
user	81m35.120s
sys	20m43.412s

## New (with `torch==1.2.0`)

Profiler Report

Action              	|  Mean duration (s)	|  Total time (s)
-----------------------------------------------------------------
on_validation_epoch_start	|  2.7711e-05     	|  0.00013855
validation_step_end 	|  0.086168       	|  158.64
on_validation_epoch_end	|  4.1437e-05     	|  0.00020719
on_train_start      	|  0.0008452      	|  0.0008452
on_epoch_start      	|  0.00074046     	|  0.0029618
on_train_epoch_start	|  2.3448e-05     	|  9.3793e-05
get_train_batch     	|  0.0099697      	|  160.71
on_batch_start      	|  6.3074e-05     	|  1.0165
on_train_batch_start	|  4.4487e-05     	|  0.71696
training_step_end   	|  0.061716       	|  994.62
model_forward       	|  0.12642        	|  2037.4
model_backward      	|  0.24748        	|  3988.4
on_after_backward   	|  7.8244e-06     	|  0.1261
on_batch_end        	|  6.6401e-05     	|  1.0701
on_train_batch_end  	|  0.0017511      	|  28.221
optimizer_step      	|  0.08286        	|  83.523
on_epoch_end        	|  3.3962e-05     	|  0.00013585
on_train_epoch_end  	|  2.0124e-05     	|  8.0496e-05
on_train_end        	|  0.00062757     	|  0.00062757

2020-08-25 22:57:34.762 | INFO     | eval:evaluate:90 - Accuracy score: 0.761
2020-08-25 22:57:35.187 | INFO     | eval:evaluate:103 - 95.0 confidence interval 73.6 and 77.8, average: 75.9

real	114m41.030s
user	88m6.267s
sys	25m24.813s
