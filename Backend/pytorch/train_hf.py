import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import evaluate
import numpy as np

# Configuration
MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DATA_DIR = "../../DATA" # Relative to Backend/pytorch/
OUTPUT_DIR = "../models/fine_tuned_deepfake"

def main():
    print(f"Loading data from {DATA_DIR}...")
    # Load dataset
    try:
        ds = load_dataset("imagefolder", data_dir=DATA_DIR)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Split dataset if no validation split exists
    if "validation" not in ds:
        print("Splitting dataset into train/validation...")
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds["validation"] = ds.pop("test")

    print(f"Train size: {len(ds['train'])}")
    print(f"Validation size: {len(ds['validation'])}")
    
    # Check labels
    labels = ds["train"].features["label"].names
    print(f"Labels: {labels}")
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    
    # Define transforms
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    
    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples
        
    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples
        
    # Apply transforms
    print("Applying transforms...")
    ds["train"].set_transform(train_transforms)
    ds["validation"].set_transform(val_transforms)
    
    # Load model
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True # In case we need to resize head (though ideally sticking to 2 classes)
    )
    
    # Training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    # Metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)
        
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    print("Training finished. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    trainer.create_model_card()

if __name__ == "__main__":
    main()
