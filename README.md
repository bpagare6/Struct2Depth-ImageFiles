# struct2depth

## Steps to run the code

1. Download the pre-trained model from [here](https://drive.google.com/file/d/1mjb4ioDRH8ViGbui52stSUDwhkGrDXy8/view)
2. Add your **PNG** images in the input folder
3. Set the variables from terminal:
    - input_dir="./input/"
    - output_dir="./output/"
4. Set the path to model:
    - model_checkpoint="your/model/checkpoint"
    while giving the model checkpoint don't give any extension just the model-number (eg. model_checkpoint="trained-models/model-199160")
5. Runt the code:
  python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion true \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint
