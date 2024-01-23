def return_all_tokens_vit_hook(module, input, output):
    module.captured_output = output