import h5py

# 打开刚刚生成的文件（只读模式 "r"）
with h5py.File("TEST/output_tensors_test.h5", "r") as f:
    # 看看存了哪些 sample_id
    sample_ids = list(f.keys())
    print(f"存入的样本有: {sample_ids[:3]} ...")

    # 挑第一个 sample 进去看看
    first_sample = f[sample_ids[0]]
    print(f"\n原始 Prompt: {first_sample.attrs['original_prompt']}")
    print(f"模型输出: {first_sample.attrs['model_output']}")

    # 看看存了哪些 token 组
    tokens_grp = first_sample["generated_tokens"]
    token_keys = list(tokens_grp.keys())
    print(f"\n保存的 Tokens 目录: {token_keys}")

    # 随便挑一个 token，看看它的属性和张量形状
    first_token = tokens_grp[token_keys[0]]
    print(f"Token '{first_token.attrs['text']}' 的正向索引(forward_idx)是 {first_token.attrs['forward_idx']}")
    print(f"Token '{first_token.attrs['text']}' 的反向索引(backward_idx)是 {first_token.attrs['backward_idx']}")

    # 查看具体层的张量信息
    layer_keys = list(first_token.keys())
    # 加 [:] 真正把张量数据从硬盘读进内存
    first_layer_tensor = first_token[layer_keys[0]][:]

    print(f"\n提取出的张量 [{layer_keys[0]}] 形状是: {first_layer_tensor.shape}")
    print(f"数据类型: {first_layer_tensor.dtype}")