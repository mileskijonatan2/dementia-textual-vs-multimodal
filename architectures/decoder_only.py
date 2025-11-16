import torch
import time
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModelForCausalLM
from utils import get_metrics


class DecoderOnlyArchitecture:
    def __init__(self, model_name, learning_rate, num_epochs, batch_size, device, train_dataset, test_dataset, eval_dataset, bf16=True, max_length=2048, seed=42):
        """
        Class for training and inference of all decoder-only based LLMs

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
        :param bf16: whether to use bf16
        :type bf16: bool
        :param max_length: maximum length of sequence
        :type max_length: int
        :param seed: random seed
        :type seed: int
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset

        self.nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            quantization_config=self.nf4_config,
            use_cache=False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.bf16 = bf16
        self.max_length = max_length

        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        self.seed = seed

    def train(self):
        """
        Fine-tune the model on the training dataset

        :return: None
        """
        args = SFTConfig(
            output_dir=f"./finetuned_models/{self.model_name}_finetuned",
            do_eval=True,
            do_predict=True,
            #dataset_text_field="text",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size//2 if self.batch_size >= 2 else 1,
            gradient_accumulation_steps=1,
            #warmup_steps=5,
            max_length=self.max_length,
            packing=False,
            logging_strategy="epoch",
            save_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=self.learning_rate,
            bf16=self.bf16,
            seed=self.seed,
            dataloader_drop_last=False,
            report_to="none",
            lr_scheduler_type='constant',
            completion_only_loss=True,
        )

        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,
            # formatting_func=create_prompt,
            args=args,
            train_dataset=self._map_to_prompt_completion_dataset(self.train_dataset),
            eval_dataset=self._map_to_prompt_completion_dataset(self.eval_dataset),
        )

        print(f"Fine-tuning of {self.model_name} started.")
        print("-------------------")
        start = time.time()
        trainer.train()
        time_taken = time.time() - start
        print(f"Fine-tuning time for {self.model_name} with {self.num_epochs} epochs: {time_taken:.2f}s")
        print("-------------------")
        print(f"Fine-tuning of {self.model_name} finished.")

    def _map_to_prompt_completion_dataset(self, dataset):
        """
        Transforms the dataset into a prompt completion dataset, suitable for decoder-only models
        :param dataset: dataset
        :type dataset: Dataset
        """
        new_dataset = dataset.rename_columns({"input_text": "prompt", "target_text": "completion"})
        # The following code yields warning for tokenization with LLaMA models
        """
        return new_dataset.map(
            lambda example: {"prompt": f"Question: {example['prompt']}\nAnswer:"}
        )
        """
        return new_dataset

    def predict(self, split="test", calculate_metrics=True):
        """
        Inference on specified dataset split. Can return classification metrics results for the prediction results.
        :param split: train, test or eval split
        :type split: str
        :param calculate_metrics: whether to calculate metrics or not
        :type calculate_metrics: bool
        """
        self.tokenizer.padding_side = "left"
        dataset = self.test_dataset if split == "test" else self.train_dataset if split == "train" else self.eval_dataset
        dataset = self._map_to_prompt_completion_dataset(dataset=dataset)
        true = list(dataset['completion'])  # ground truths
        ids = list(dataset['ids'])
        prompts = list(dataset['prompt'])
        predictions = []

        for i in tqdm(range(0, len(prompts), self.batch_size)):
            batch = prompts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            input_lengths = [len(ids) for ids in inputs["input_ids"]]

            with torch.no_grad():
                if self.bf16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=3,
                            #use_cache=True
                            #temperature=0.7,
                            #top_p=0.9,
                            #do_sample=True
                        )
                else:
                    outputs = self.model.generate(**inputs, max_new_tokens=3)

            response_only_outputs = []
            for inp, output in enumerate(outputs):
                gen_tokens = output[input_lengths[inp]:]
                response_only_outputs.append(gen_tokens)

            decoded_preds = self.tokenizer.batch_decode(response_only_outputs, skip_special_tokens=True)
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
