# 第一步创建一个自己的虚拟环境：
conda create -n new_name python=3.10
# 第二步进入到自己的虚拟环境：
conda activate new_name 
# 第三步： (直接复制以下所有命令到控制台回车运行就好，后面缺什么再补充就好了,非常快)
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchsummary -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxruntime==1.15.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pycocotools==2.0.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install faster-coco-eval==1.6.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install PyYAML -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install calflops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install onnxsim==0.4.36 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxruntime-gpu==1.18.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations pytorch_wavelets==1.3.0 tidecv PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
