import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from utils import get_metrics


class EncoderDecoderArchitecture:
    def __init__(self, model_name, learning_rate, num_epochs, batch_size, device, train_dataset, test_dataset, eval_dataset, fp16=False, seed=42, max_new_tokens=6):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device)

        if not hasattr(self.model.config, "num_hidden_layers"):
            if hasattr(self.model.config, "encoder") and hasattr(self.model.config.encoder, "num_hidden_layers"):
                self.model.config.num_hidden_layers = self.model.config.encoder.num_hidden_layers
            elif hasattr(self.model.config, "num_layers"):
                self.model.config.num_hidden_layers = self.model.config.num_layers
            elif hasattr(self.model.config, "encoder") and hasattr(self.model.config.encoder, "num_layers"):
                self.model.config.num_hidden_layers = self.model.config.encoder.num_layers
            else:
                self.model.config.num_hidden_layers = 12

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.fp16 = fp16
        self.seed = seed
        self.max_new_tokens = max_new_tokens

    def train(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./finetuned_models/{self.model_name}_finetuned",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size//2 if self.batch_size >= 2 else 1,
            gradient_accumulation_steps=1,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=5,
            eval_strategy="epoch",
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            report_to="none",
            lr_scheduler_type='constant',
            seed=self.seed,
            dataloader_drop_last=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self._prepare_trainer_dataset(self.train_dataset),
            eval_dataset=self._prepare_trainer_dataset(self.eval_dataset),
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )

        print(f"Fine-tuning of {self.model_name} started.")
        print("-------------------")
        start = time.time()
        trainer.train()
        time_taken = time.time() - start
        print(f"Fine-tuning time for {self.model_name} with {self.num_epochs} epochs: {time_taken:.2f}s")
        print("-------------------")
        print(f"Fine-tuning of {self.model_name} finished.")

    def _preprocess_batch(self, batch):
        inputs = self.tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        labels = self.tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs["labels"] = labels["input_ids"]
        return inputs

    def _prepare_trainer_dataset(self, dataset) -> Dataset:
        dataset = dataset.map(self._preprocess_batch, batched=True)
        return dataset

    def _prepare_inference_loader(self, dataset) -> DataLoader:
        dataset = dataset.map(self._preprocess_batch, batched=True, remove_columns=["input_text", "target_text", "ids"])
        
        if 't5gemma' in self.model_name.lower():
            from transformers import default_data_collator
            loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=default_data_collator)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
        return loader

    def predict(self, split="test", calculate_metrics=True):
        dataset = self.test_dataset if split == "test" else self.train_dataset if split == "train" else self.eval_dataset
        true = list(dataset['target_text'])
        ids = list(dataset['ids'])
        loader = self._prepare_inference_loader(dataset)

        predictions = []

        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens
                )

            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_preds)

        metrics_scores = get_metrics(true, predictions, self.model_name)
        if calculate_metrics and metrics_scores is not None:
            print(f"Metrics are successfully computed for {self.model_name}.")

            accuracy = metrics_scores["accuracy"]
            precision = metrics_scores["precision"]
            recall = metrics_scores["recall"]
            f1 = metrics_scores["f1"]
        else:
            accuracy, precision, recall, f1 = None, None, None, None

        return ids, predictions, true, accuracy, precision, recall, f1
