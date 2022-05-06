# python main.py --data_train="DIV2K"  --save='DIV2KResSigmoidPixelShuffle' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3
# python main.py --test_only --load='DIV2KSigmoidTail' --n_colors=3
# python main.py --test_only --load='DIV2KResSigmoidTail' --n_colors=3 --data_test="DIV2K_VID"
# python main.py --model='edsr' --data_train="DIV2K"  --save='DIV2KEDSR' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3
# python main.py --test_only --model='edsr' --load='DIV2KEDSR' --n_colors=3 --data_test="DIV2K_VID"
# python main.py --model='fsrcnn' --data_train="DIV2K"  --save='DIV2KFSRCNN' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3
# python main.py --model='espcn' --data_train="DIV2K"  --save='DIV2KESPCN' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3
# python main.py --model='espcn' --test_only --load='DIV2KESPCN' --n_colors=3 --data_test="DIV2K_VID"

# python main.py --model='fsrcnn' --test_only --load='DIV2KFSRCNN' --n_colors=3 --data_test="DIV2K_VID"

# python main.py --test_only --load='DIV2KResSigmoidTail' --n_colors=3 --data_test="DIV2K_VID"


# python main.py --model='kerneledsr' --data_train="DIV2K"  --save='DIV2KKernelEDSRResx4' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3 --n_resblocks=4
python main.py --model='kerneledsr' --test_only --load='DIV2KKernelEDSRResx4' --n_colors=3 --data_test="DIV2K_VID" --n_resblocks=4