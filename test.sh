# python main.py --data_train="DIV2K"  --save='DIV2KResSigmoidPixelShuffle' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3
# python main.py --test_only --load='DIV2KSigmoidTail' --n_colors=3
# python main.py --test_only --load='DIV2KResSigmoidTail' --n_colors=3 --data_test="DIV2K_VID"
# python main.py --model='edsr' --data_train="DIV2K"  --save='DIV2KEDSR' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3
# python main.py --test_only --model='edsr' --load='DIV2KEDSR' --n_colors=3 --data_test="DIV2K_VID"

# python main.py --model='espcn' --test_only --load='DIV2KESPCN' --n_colors=3 --data_test="DIV2K_VID"

# python main.py --model='fsrcnn' --test_only --load='DIV2KFSRCNN' --n_colors=3 --data_test="DIV2K_VID"

# python main.py --test_only --load='DIV2KResSigmoidTail' --n_colors=3 --data_test="DIV2K_VID"


# python main.py --model='kerneledsr' --data_train="DIV2K"  --save='DIV2KKernelEDSRResx4' --dir_data='/home/yejiannan/SIGGRAPH2022/data' --epochs=500 --patch_size=64 --n_colors=3 --n_resblocks=4
# python main.py --model='kerneledsr' --test_only --load='DIV2KKernelEDSRResx4' --n_colors=3 --data_test="DIV2K_VID" --n_resblocks=4
# FSRCNN训练
# python main.py --model='fsrcnn' --data_train="DIV2K"  --save='DIV2KFSRCNN' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8
# ESPCN训练
# python main.py --model='espcn' --data_train="DIV2K"  --save='DIV2KESPCN' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8

# ESPCN测试
# python main.py --model='espcn' --test_only --load='DIV2KESPCN' --n_colors=3 --data_test="DIV2K_VID" --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset'
# FSRCNN测试
# python main.py --model='fsrcnn' --test_only --load='DIV2KFSRCNN' --n_colors=3 --data_test="DIV2K_VID" --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset'

# EDSR x8 2倍训练
# python main.py --model='edsr' --data_train="DIV2K"  --save='DIV2KEDSRx8' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=8
# python main.py --model='edsr' --test_only --load='DIV2KEDSRx8' --n_colors=3 --data_test="DIV2K_VID" --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --n_resblocks=8
# EDSR x8 2倍训练 x4->x2 实际用
# python main.py --model='edsr' --data_train="DIV2KLOW"  --save='DIV2KEDSRx8Scale4to2' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=8

# EDSR x16 2倍训练
# python main.py --model='edsr' --data_train="DIV2K"  --save='DIV2KEDSRx16' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=16
# python main.py --model='edsr' --test_only --load='DIV2KEDSRx16' --n_colors=3 --data_test="DIV2K_VID" --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --n_resblocks=16
# EDSR x16 4倍训练
# python main.py --model='edsr' --data_train="DIV2K"  --save='DIV2KEDSRx16Scale4' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=16 --scale='4'

# kernelEDSR X8 2倍训练
# python main.py --model='kerneledsr' --data_train="DIV2K"  --save='DIV2KKernelEDSRx8' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=8
# python main.py --model='kerneledsr' --test_only --load='DIV2KKernelEDSRx8' --n_colors=3 --data_test="DIV2K_VID" --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --n_resblocks=8
# kernelEDSR x8 2倍训练 x4->x2 实际用
# python main.py --model='kerneledsr' --data_train="DIV2KLOW"  --save='DIV2KKernelEDSRx8Scale4to2' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=8
# kernelEDSR X16 2倍训练
# python main.py --model='kerneledsr' --data_train="DIV2K"  --save='DIV2KKernelEDSRx16' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=16
python main.py --model='kerneledsr' --test_only --load='DIV2KKernelEDSRx16' --n_colors=3 --data_test="DIV2K_VID" --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --n_resblocks=16
# kernelEDSR X16 4倍训练
# python main.py --model='kerneledsr' --data_train="DIV2K"  --save='DIV2KKernelEDSRx16Scale4' --dir_data='/home/yejiannan/Project/Kernel-Supersampling/dataset' --epochs=500 --patch_size=64 --n_colors=3 --batch_size=8 --n_resblocks=16 --scale='4'
