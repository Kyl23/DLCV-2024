# bash hw2_1.sh out_dir1
# python3 digit_classifier.py --folder out_dir1 --checkpoint Classifier.pth

# bash hw2_2.sh hw2_data/face/noise out_dir2 hw2_data/face/UNet.pt
# python3 mse_cal.py hw2_data/face/GT out_dir2

bash hw2_3.sh hw2_data/textual_inversion/input.json out_dir3 stable-diffusion/sd-v1-4.ckpt
python3 evaluation/grade_hw2_3.py --json_path hw2_data/textual_inversion/input.json --input_dir hw2_data/textual_inversion/ --output_dir out_dir3
