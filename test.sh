# python main.py --data_train="DIV2K"  --save='DIV2KSigmoid' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3
# python main.py --test_only --load='DIV2KSigmoid' --n_colors=3
python main.py --test_only --load='DIV2KSigmoid' --n_colors=3 --data_test="DIV2K_VID"