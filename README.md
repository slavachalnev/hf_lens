# hf_lens

Experimental


```python
bert_model = AutoModel.from_pretrained("bert-base-uncased")
hooked_model = HookedModule(bert_model)
hooked_model.print_model_structure()

def example_hook(module, input, output):
    print(f"Hook called with tensor of shape: {output.shape}")

def example_hook2(module, input, output):
    print(f"Hook2 called with tensor output shape {output[0].shape}")

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
input_ids = torch.tensor([[101, 2054, 2003, 2026, 2171, 102]])

with hooked_model.hooks(fwd=[('encoder.layer.0.attention.self.query', example_hook)]):
    output = hooked_model(input_ids)
    with hooked_model.hooks(fwd=[('encoder.layer.0', example_hook2)]):
        output = hooked_model(input_ids)
    output = hooked_model(input_ids)
```