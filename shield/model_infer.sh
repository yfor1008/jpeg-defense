# python model_infer.py -p=../data/adv_images/orig/ -m=mobilenet_v2 > orig_mobilenet_v2.txt

# python model_infer.py -p=../data/adv_images/orig/ -m=resnet_v2_50 > orig_resnet_v2_50.txt

# python model_infer.py -p=../data/adv_images/orig/ -m=resnet_v2_152 > orig_resnet_v2_152.txt



python model_infer.py -p=../data/adv_images/resnet_v2_50/fgsm/ -m=resnet_v2_50 > resnet_v2_50_fgsm_resnet_v2_50.txt

python model_infer.py -p=../data/adv_images/resnet_v2_50/fgsm/ -m=resnet_v2_152 > resnet_v2_50_fgsm_resnet_v2_152.txt

python model_infer.py -p=../data/adv_images/resnet_v2_50/fgsm/ -m=mobilenet_v2 > resnet_v2_50_fgsm_mobilenet_v2.txt

python model_infer.py -p=../data/adv_images/resnet_v2_50/df/ -m=resnet_v2_50 > resnet_v2_50_df_resnet_v2_50.txt

python model_infer.py -p=../data/adv_images/resnet_v2_50/df/ -m=resnet_v2_152 > resnet_v2_50_df_resnet_v2_152.txt

python model_infer.py -p=../data/adv_images/resnet_v2_50/df/ -m=mobilenet_v2 > resnet_v2_50_df_mobilenet_v2.txt



python model_infer.py -p=../data/adv_images/resnet_v2_152/fgsm/ -m=resnet_v2_50 > resnet_v2_152_fgsm_resnet_v2_50.txt

python model_infer.py -p=../data/adv_images/resnet_v2_152/fgsm/ -m=resnet_v2_152 > resnet_v2_152_fgsm_resnet_v2_152.txt

python model_infer.py -p=../data/adv_images/resnet_v2_152/fgsm/ -m=mobilenet_v2 > resnet_v2_152_fgsm_mobilenet_v2.txt

python model_infer.py -p=../data/adv_images/resnet_v2_152/df/ -m=resnet_v2_50 > resnet_v2_152_df_resnet_v2_50.txt

python model_infer.py -p=../data/adv_images/resnet_v2_152/df/ -m=resnet_v2_152 > resnet_v2_152_df_resnet_v2_152.txt

python model_infer.py -p=../data/adv_images/resnet_v2_152/df/ -m=mobilenet_v2 > resnet_v2_152_df_mobilenet_v2.txt



python model_infer.py -p=../data/adv_images/mobilenet_v2/fgsm/ -m=resnet_v2_50 > mobilenet_v2_fgsm_resnet_v2_50.txt

python model_infer.py -p=../data/adv_images/mobilenet_v2/fgsm/ -m=resnet_v2_152 > mobilenet_v2_fgsm_resnet_v2_152.txt

python model_infer.py -p=../data/adv_images/mobilenet_v2/fgsm/ -m=mobilenet_v2 > mobilenet_v2_fgsm_mobilenet_v2.txt

# python model_infer.py -p=../data/adv_images/mobilenet_v2/df/ -m=resnet_v2_50 > mobilenet_v2_df_resnet_v2_50.txt

# python model_infer.py -p=../data/adv_images/mobilenet_v2/df/ -m=resnet_v2_152 > mobilenet_v2_df_resnet_v2_152.txt

# python model_infer.py -p=../data/adv_images/mobilenet_v2/df/ -m=mobilenet_v2 > mobilenet_v2_df_mobilenet_v2.txt
