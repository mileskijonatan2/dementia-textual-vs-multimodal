import torch
import time
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModelForCausalLM
from utils import get_metrics


class DecoderOnlyArchitecture:
    def __init__(self, model_name, learning_rate, num_epochs, batch_size, device, train_dataset, test_dataset, eval_dataset, bf16=True, max_length=2048, seed=42, max_new_tokens=3, padding_side="right", temperature=None, do_sample=None):
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
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = padding_side
        self.bf16 = bf16
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.get_target_modules(model_name)
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        self.seed = seed

    def train(self):
        args = SFTConfig(
            output_dir=f"./finetuned_models/{self.model_name}_finetuned",
            do_eval=True,
            do_predict=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size//2 if self.batch_size >= 2 else 1,
            gradient_accumulation_steps=1,
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

    def get_target_modules(self, model_name):
        if "gemma" in model_name:
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        if "mistral" in model_name.lower():
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "w1", "w2", "w3"
            ]
        if "llama" in model_name.lower():
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        if "gpt-neo" in model_name.lower() or "gpt-j" in model_name.lower():
            return [
                "q_proj", "k_proj", "v_proj", "out_proj"
            ]
        if "gpt2" in model_name.lower():
            return [
                "c_attn", "c_proj", "c_fc"
            ]
        if "qwen" in model_name.lower() or "deepseek" in model_name.lower():
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        raise ValueError(f"Unknown model: {model_name}")

    def _map_to_prompt_completion_dataset(self, dataset):
        new_dataset = dataset.rename_columns({"input_text": "prompt", "target_text": "completion"})
        return new_dataset

    def predict(self, split="test", calculate_metrics=True):
        self.tokenizer.padding_side = "left"
        dataset = self.test_dataset if split == "test" else self.train_dataset if split == "train" else self.eval_dataset
        dataset = self._map_to_prompt_completion_dataset(dataset=dataset)
        true = list(dataset['completion'])
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

            gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
            if self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature
            if self.do_sample is not None:
                gen_kwargs["do_sample"] = self.do_sample

            with torch.no_grad():
                if self.bf16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model.generate(
                            **inputs,
                            **gen_kwargs,
                        )
                else:
                    outputs = self.model.generate(**inputs, **gen_kwargs)

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
