from peft import LoraConfig, get_peft_model
from model import load_model, load_processor, print_trainable_parameters

def apply_lora(model):
    # Logging the number of parameters before applying PEFT method
    print_trainable_parameters(model)

    config = LoraConfig(
        r=256,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)

    return lora_model