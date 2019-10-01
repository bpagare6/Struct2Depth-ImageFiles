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

### The original code can be found [here](https://github.com/tensorflow/models/tree/master/research/struct2depth)
### The respective research paper can be found V. Casser, S. Pirk, R. Mahjourian, A. Angelova, Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos, AAAI Conference on Artificial Intelligence, 2019[ https://arxiv.org/pdf/1811.06152.pdf]( https://arxiv.org/pdf/1811.06152.pdf)
