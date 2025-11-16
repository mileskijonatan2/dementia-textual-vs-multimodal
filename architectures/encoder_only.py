import torch
import time
import numpy as np
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import get_metrics


class EncoderOnlyArchitecture:
    def __init__(self, model_name, learning_rate, num_epochs, batch_size, device, train_dataset, test_dataset, eval_dataset, fp16=False, seed=42, weight_decay=0.0005):
        """
        Class for training and inference of all encoder-decoder based LLMs

        :param model_name: name of the model
        :type model_name: str
        :param learning_rate: learning rate
        :type learning_rate: float
        :param num_epochs: number of epochs
        :type num_epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param device: device to run the model on
        :type device: str
        :param train_dataset: training dataset
        :type train_dataset: Dataset
        :param test_dataset: test dataset
        :type test_dataset: Dataset
        :param eval_dataset: evaluation dataset
        :type eval_dataset: Dataset
        :param fp16: whether to use fp16
        :type fp16: bool
        :param seed: random seed
        :type seed: int
        :param weight_decay: Adam optimizer weight decay
        :type weight_decay: float
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map=device, num_labels=2)
        self.fp16 = fp16
        self.seed = seed
        self.weight_decay = weight_decay
        self.metrics = {'accuracy': evaluate.load("accuracy"), 'precision': evaluate.load("precision"),
                        'recall': evaluate.load("recall"), "f1": evaluate.load("f1")}
        self.trainer = None  # it is added here so that the trainer can be accessed from predict() function for easier inference

    def train(self):
        """
        Fine-tune the model on the training dataset

        :return: None
        """
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = self.metrics['accuracy'].compute(predictions=preds, references=labels)
            prec = self.metrics['precision'].compute(predictions=preds, references=labels)
            rec = self.metrics['recall'].compute(predictions=preds, references=labels)
            f1_score = self.metrics['f1'].compute(predictions=preds, references=labels)
            return {"accuracy": acc['accuracy'], "precision": prec['precision'], "recall": rec['recall'],
                    "f1": f1_score['f1']}

        training_args = TrainingArguments(
            output_dir=f"./finetuned_models/{self.model_name}_finetuned",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size//2 if self.batch_size >= 2 else 1,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate, #1e-4
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay, #0.0005
            logging_strategy="epoch",
            load_best_model_at_end=True,
            seed=42,
            fp16=self.fp16,
            report_to=[],
            lr_scheduler_type="constant",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self._prepare_trainer_dataset(self.train_dataset),
            eval_dataset=self._prepare_trainer_dataset(self.eval_dataset),
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        print(f"Fine-tuning of {self.model_name} started.")
        print("-------------------")
        start = time.time()
        self.trainer.train()
        time_taken = time.time() - start
        print(f"Fine-tuning time for {self.model_name} with {self.num_epochs} epochs: {time_taken:.2f}s")
        print("-------------------")
        print(f"Fine-tuning of {self.model_name} finished.")

    def _preprocess_batch(self, batch):
        """
        Transform each batch into suitable format for encoder-only model
        :param batch: DataLoader batch
        :type batch: torch.tensor
        """
        return self.tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=512)

    def _prepare_trainer_dataset(self, dataset):
        """
        Transforms Dataset object for HuggingFace Trainer
        :param dataset: train or eval dataset for HuggingFace trainer
        :type dataset: Dataset
        """
        dataset = dataset.map(self._preprocess_batch, batched=True)
        dataset = dataset.rename_columns({"target_text": "label"})
        dataset = dataset.map(lambda x: {"label": 0 if x["label"] == "control" else 1})
        return dataset

    def predict(self, split="test", calculate_metrics=True):
        """
        Inference on specified dataset split. Can return classification metrics results for the prediction results.
        :param split: train, test or eval split
        :type split: str
        :param calculate_metrics: whether to calculate metrics or not
        :type calculate_metrics: bool
        """
        dataset = self.test_dataset if split == "test" else self.train_dataset if split == "train" else self.eval_dataset
        true = list(dataset['target_text'])  # ground truths
        ids = list(dataset['ids'])

        dataset = self._prepare_trainer_dataset(dataset)

        pred_output = self.trainer.predict(dataset)
        preds = np.argmax(pred_output.predictions, axis=-1)

        predictions = ['control' if p == 0 else 'dementia' for p in preds]

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
