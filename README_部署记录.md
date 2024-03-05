# Llama2-Chinese 部署记录
[Llama-Chinese链接](https://github.com/LlamaFamily/Llama-Chinese)  
建议在docker中运行,在wsl中Linux无法访问外网，会导致hf的权重和模型无法下载

### 创建一个空文件夹
```
git clone https://github.com/facebookresearch/llama-recipes.git
cd llama-recipes
```

### 安装环境
```
pip install -r requirements.txt
cd src/llama_recipes
mv finetuning.py ../  # 把llama_recipes目录下的finetuning.py移动到src路径下,一会执行finetuning.py文件
```

### 模型下载
[hf原始模型模型7b ](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)  
[hf-cn模型7b](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat)  
更多模型可以参考[原文](https://github.com/LlamaFamily/Llama-Chinese)  
下载完的模型要放在src/llama_recipes/ 路径下，文件夹命名为Llama-2-7b-chat-hf

### 修改参数：
```
目录：src/llama_recipes/config/training.py
model_name: str="./llama_recipes/Llama-2-7b-chat-hf(模型文件夹名字)"
batch_size_training: int=2 （看情况调，硬件不行的情况下设置为1）
context_length: int=2048
num_epochs: int=3
output_dir: str = ".PEFT_model" （自己找个位置放输出结果）
修改数据集的地方在 src/llama_recipes/datasets/samsum_dataset.py，这里可以换成自己的数据集。
```

### 运行文件
```
cd llama-recipes/src  # 切换到llama-recipes/src/目录下
python finetuning.py --use_peft --peft_method lora --quantization --model_name
```

### 推理测试
可以在加载模型前问一些问题，然后合并lora参数后再问相同的问题，观察两个答案的区别。  
在**修改参数**中的output_dir下就是微调后的lora模型，需要和原模型一起加载使用  
**----以下作为参考----**  
通过[PEFT](https://github.com/huggingface/peft)加载预训练模型参数和微调模型参数，以下示例代码中，base_model_name_or_path为预训练模型参数保存路径，finetune_model_path为微调模型参数保存路径。 
```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
# 例如: finetune_model_path='FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
finetune_model_path=''  
config = PeftConfig.from_pretrained(finetune_model_path)
# 例如: base_model_name_or_path='meta-llama/Llama-2-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()
input_ids = tokenizer(['<s>Human: 介绍一下北京\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

