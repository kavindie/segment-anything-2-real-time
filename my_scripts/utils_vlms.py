import torch
from transformers import AriaProcessor, AriaForConditionalGeneration

model_id_or_path = "rhymes-ai/Aria"
model = AriaForConditionalGeneration.from_pretrained(
    model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16
)
processor = AriaProcessor.from_pretrained(model_id_or_path, use_fast=True)

def answerwAria(
    question,
    image = None,
):
    """
    Generate captions for the segmented image using the Aria model.
    
    Args:
        question: The question to be answered.
        image: The input image.

    Returns:
        answer to the question based on the image
    """
    if image is None:
        inputs = processor(
            text=question,
            return_tensors="pt"
        )

    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"text": f"{question}", "type": "text"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    
    inputs.to(model.device)


    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            do_sample=True,
            temperature=0.1,#0.9
        )
    output_ids = output[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(output_ids, skip_special_tokens=True)
    response = response.replace("<|im_end|>", "").strip()
    return response

