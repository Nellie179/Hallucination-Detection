import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
from typing import Optional, Tuple

class TSVLayer(nn.Module):
    def __init__(self, tsv, lam):
        super(TSVLayer, self).__init__()
        self.tsv = tsv
        self.lam = lam

    def forward(self, x):
        if self.tsv is not None:
            # 🚀 修复原作者强写 .half() 的致命 BUG，继承原汁原味的 bfloat16/float32
            orig_dtype = x.dtype
            y = self.lam[0] * self.tsv.repeat(1, x.shape[1], 1)
            y = y.to(x.device).to(orig_dtype)
            x = x + y
            return x
        else:
            return x

class LlamaDecoderLayerWrapper(nn.Module):
    def __init__(self, llama_decoder_layer, tsv_layer, model_name='llama3.1-8B'):
        super().__init__()
        self.llama_decoder_layer = llama_decoder_layer
        self.tsv_layer = tsv_layer
        self.model_name = model_name

    def forward(self, *args, **kwargs):
        # 🚀 终极解法：不抄底层，直接无缝代理原底座的 forward！
        # 完美继承新版 Transformers 的所有特性、SDPA 传参和 Accelerate Hooks
        outputs = self.llama_decoder_layer(*args, **kwargs)

        # HuggingFace 的 DecoderLayer 默认返回 Tuple，第一个元素必为 hidden_states
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            # 拦截：在残差流的最末端挂载 TSV
            hidden_states = self.tsv_layer(hidden_states)
            # 重新打包回 Tuple 原路返回
            return (hidden_states,) + outputs[1:]
        else:
            # 极端情况兜底
            hidden_states = outputs
            hidden_states = self.tsv_layer(hidden_states)
            return hidden_states

    # =========================================================
    # 🚀 顶会级基建：防死锁动态属性透传 (Robust Dynamic Proxy)
    # =========================================================
    def __getattr__(self, name):
        # 1. 优先让 PyTorch 原生机制查找 (parameters, buffers, registered modules)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
            
        # 2. 防御无限递归死锁：必须确保底座已经安全注册在内部 _modules 字典中，再进行透传
        if 'llama_decoder_layer' in self._modules:
            return getattr(self.llama_decoder_layer, name)
            
        # 3. 兜底防御：如果到底都没有，抛出标准错误，绝不乱吞异常
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)

def find_longest_modulelist(model, path=""):
    longest_path = path
    longest_len = 0
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len

def find_module(block, keywords):
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")

def get_embedding_layer(model: PreTrainedModel):
    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_lm_head(model: PreTrainedModel):
    keywords = ["lm_head", "embed_out"]
    return find_module(model, keywords)

def get_lm_pipeline(model: PreTrainedModel):
    model_class = model.__class__.__name__

    if model_class == "LlamaForCausalLM":
        return nn.Sequential(model.model.norm, model.lm_head)
    elif model_class == "RWForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoXForCausalLM":
        return nn.Sequential(model.gpt_neox.final_layer_norm, model.embed_out)
    return get_lm_head(model)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path

def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_tsv_layers(model: PreTrainedModel, tsv: Tensor, alpha: list, args):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    attn_keywords = ["self_attn"]
    
    assert len(tsv) == len(layers)
    if args.component == 'mlp':
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                original_mlp = find_module(layer, mlp_keywords)
                layer.mlp = nn.Sequential(original_mlp, TSVLayer(tsv[i], alpha)) 

    elif args.component == 'attn':
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                original_attn = find_module(layer, attn_keywords)
                layer.self_attn = nn.Sequential(original_attn, TSVLayer(tsv[i], alpha)) 
                
    elif args.component == 'res':
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                decoder_layer = layers[i]
                # 挂载我们新写的无敌代理 Wrapper
                layers[i] = LlamaDecoderLayerWrapper(decoder_layer, TSVLayer(tsv[i], alpha), args.model_name)