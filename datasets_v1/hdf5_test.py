import h5py

with h5py.File("TEST/output_tensors_test.h5", "r") as f:
    sample_ids = list(f.keys())
    print(f"Stored samples identified: {sample_ids[:3]} ...")

    first_sample = f[sample_ids[0]]
    print(f"\nOriginal Prompt Trace: {first_sample.attrs['original_prompt']}")
    print(f"Model Generative Output: {first_sample.attrs['model_output']}")

    tokens_grp = first_sample["generated_tokens"]
    token_keys = list(tokens_grp.keys())
    print(f"\nSaved Target Tokens Directory: {token_keys}")

    first_token = tokens_grp[token_keys[0]]
    print(f"Token '{first_token.attrs['text']}' forward index (forward_idx): {first_token.attrs['forward_idx']}")
    print(f"Token '{first_token.attrs['text']}' backward index (backward_idx): {first_token.attrs['backward_idx']}")

    layer_keys = list(first_token.keys())
    first_layer_tensor = first_token[layer_keys[0]][:]

    print(f"\nExtracted Activation Tensor [{layer_keys[0]}] Array Shape: {first_layer_tensor.shape}")
    print(f"Tensor Precision Format: {first_layer_tensor.dtype}")