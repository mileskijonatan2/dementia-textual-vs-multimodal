import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import librosa
import time
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple
from peft import LoraConfig
from peft import get_peft_model
from transformers import TrainingArguments, Trainer
from utils import get_metrics

#os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class AudioDataCollator:
    def __init__(self, processor, task_prompt: str, max_length: int = None, audio_path='./dataset/data/DementiaBank/audio/{}/cookie/cleaned/{}.mp3', debug=False):
        self.processor = processor
        self.task_prompt = task_prompt
        self.max_length = max_length or processor.feature_extractor.n_samples
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.audio_path = audio_path
        self.debug = debug

    def process_audio(self, id: str, label: str) -> Tuple:
        """Process a single audio file."""
        try:
            label = 'Control' if label == 'control' else 'Dementia'
            audio, sr = librosa.load(self.audio_path.format(label, id), sr=None)
            target_sr = 16000
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

            # Truncate or pad audio
            if len(audio) > self.max_length:
                if self.debug:
                    print(f"Audio length ({len(audio)}) exceeds max_length ({self.max_length}). Truncating")
                audio = audio[:self.max_length]
            # else: # better to allow for padding in the processor, perhaps
            #     audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')

            return audio.astype(np.float32), self.audio_path.format(label, id)  # Ensure float32 for compatibility
        except Exception as e:
            raise RuntimeError(f"Error processing audio from {id}")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        valid_examples = []
        audios = []
        combined_texts = []  # Combined prompt and caption for decoder-only model

        # Process each example
        for example in examples:
            try:
                # Process audio
                # 'id', 'transcript', 'start-end', 'class'
                audio, url = self.process_audio(example['ids'], example['target_text'])
                audios.append(audio)

                # Combine prompt and caption into a single input sequence
                conversation = [
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": url},
                        {"type": "text", "text": self.task_prompt.format(example['input_text'])}
                    ]},
                    {"role": "assistant", "content": example["target_text"]}  # Caption
                ]
                combined_text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=False, tokenize=False
                )
                combined_texts.append(combined_text)

                valid_examples.append(example)
            except Exception as e:
                print(f"Failed to process {example['id']}: {str(e)}")
                continue

        if not valid_examples:
            raise ValueError("NO valid errors in batch")

        if self.debug:
            # Debugging: Validate inputs
            print(f"\n=== Debugging Inputs ===")
            print(f"Number of combined texts: {len(combined_texts)}")
            print(f"Number of audios: {len(audios)}")

        # Process combined inputs with the processor
        try:
            inputs = self.processor(
                text=list(combined_texts),
                audio=list(audios),
                return_tensors="pt",
                padding=True
            )
        except Exception as e:
            print(f"Processor error: {str(e)}")
            raise

        labels = inputs["input_ids"].clone()

        # Debugging: Print the input IDs and tokenized data
        if self.debug:
            print(f"\n=== Debugging Inputs ===")
            print(f"Input IDs:\n{inputs['input_ids']}")
            print(f"Input IDs Shape: {inputs['input_ids'].shape}")
            print(f"Input IDs Type: {type(inputs['input_ids'])}")
            print(f"Input features: {inputs}")

        # Mask the prompt portion dynamically for each example
        for i in range(len(combined_texts)):
            try:
                # Ensure we handle indexing correctly
                input_ids_row = inputs["input_ids"][i]
                if self.debug:
                    print(f"\nProcessing example {i}")
                    print(f"Input IDs for this example: {input_ids_row}")
                # Find the tokenized sequence for "<|im_start|>assistant"
                assistant_start_tokens = self.processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
                if self.debug:
                    print(f"Assistant Start Token IDs: {assistant_start_tokens}")

                # Search for the sequence of tokens in the input
                assistant_start_idx = -1
                for j in range(len(input_ids_row) - len(assistant_start_tokens) + 1):
                    if torch.equal(input_ids_row[j : j + len(assistant_start_tokens)], torch.tensor(assistant_start_tokens)):
                        assistant_start_idx = j
                        break

                if self.debug:
                    print(f"Assistant Start Index: {assistant_start_idx}")

                if assistant_start_idx != -1:  # Check if the sequence exists in the input
                    # Mask everything before the start of the assistant's response
                    labels[i, :assistant_start_idx + len(assistant_start_tokens)] = -100
                else:
                    # Fallback if the sequence is not found
                    print(f"Warning: '<|im_start|>assistant' not found in input for example {i}.")
                    labels[i, :] = -100  # Mask entire sequence if the sequence is missing
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            #"input_features": inputs.input_features,
            #"feature_attention_mask": inputs.feature_attention_mask,
            "labels": labels
        }


class Qwen2AudioModel:
    def __init__(self, model_name, learning_rate, num_epochs, batch_size, device, train_dataset, test_dataset, eval_dataset, audio_path, task_prompt="Is this pair of audio and transcript control or dementia patient: {}\nAnswer: ",  bf16=True, seed=42, debug=False):
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
        :param audio_path: relative path to the folder containing the audio records
        :type audio_path: str
        :param task_prompt: instruction prompt for the task
        :type task_prompt: str
        :param bf16: whether to use bf16
        :type bf16: bool
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
        self.torch_dtype = torch.bfloat16
        self.bf16 = bf16

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # bfloat16 if on ampere, lovelace, ada or hopper
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            quantization_config=self.quant_config,
            device_map='auto',
        )

        self.audio_path = audio_path
        self.task_prompt = task_prompt

        self.data_collator = AudioDataCollator(
            processor=self.processor,
            task_prompt=task_prompt,
            audio_path=audio_path
        )

        self.data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            shuffle=False
        )

        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            use_rslora=True,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.seed = seed
        self.debug = debug

    def train(self):
        """
        Fine-tune the model on the training dataset

        :return: None
        """
        training_args = TrainingArguments(
            output_dir=f"./finetuned_models/{self.model_name}_finetuned",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size//2 if self.batch_size >= 2 else 1,
            gradient_accumulation_steps=1,
            # warmup_steps=50,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            lr_scheduler_type="constant",
            logging_strategy="epoch",
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=self.bf16,
            seed=self.seed,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            report_to=[],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False}  # causes issues to set True for Qwen, although
        )

        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args
        )

        print(f"Fine-tuning of {self.model_name} started.")
        print("-------------------")
        start = time.time()
        trainer.train()
        time_taken = time.time() - start
        print(f"Fine-tuning time for {self.model_name} with {self.num_epochs} epochs: {time_taken:.2f}s")
        print("-------------------")
        print(f"Fine-tuning of {self.model_name} finished.")

    def run_example(self, transcript: str, id: str, label: str):
        """

        Runs inference on an audio file and prints the result.

        :param transcript: transcript of audio file
        :type transcript: str
        :param id: participant id
        :type id: str
        :param label: class label (control/dementia)
        :type label: str
        """
        try:
            label = 'Control' if label == 'control' else 'Dementia'

            # audio_data, sr = librosa.load(self.audio_path.format(label, id), sr=None)
            # target_sr = 16000
            # audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            # sr = target_sr
            # sampling_rate = sr

            # Prepare conversation
            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": self.audio_path.format(label, id)},
                    {"type": "text", "text": self.task_prompt.format(transcript)},
                ]}
            ]

            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            if self.debug:
                print(f"Templated text:\n{text}")

            audios = []

            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio" and "audio_url" in ele:
                            try:
                                # Confirm the sampling rate
                                sr = int(self.processor.feature_extractor.sampling_rate)

                                # Load and append audio
                                audio_data, sr_loaded = librosa.load(
                                    ele["audio_url"], sr=None
                                )

                                from librosa import resample

                                # Resample audio data to match the processor's sampling rate
                                audio_data = resample(audio_data, orig_sr=sr_loaded, target_sr=sr)

                                if self.debug:
                                    print(f"Loaded with sample rate {sr_loaded}")

                                audios.append(audio_data)

                            except Exception as e:
                                print(f"Failed to process audio from {ele['audio_url']}: {e}")

                inputs = self.processor(
                    text=text,
                    audio=audios,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    # max_length=512
                ).to("cuda")

                generate_ids = self.model.generate(**inputs, max_new_tokens=3)

                # print(f"Input shape: {inputs.input_ids.size(1)}, Generated shape: {generate_ids.shape}")

                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                response = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                return response

        except Exception as e:
            print(f"An error occured in function run_example: {e}")

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
        prompts = list(dataset['input_text'])

        self.model.eval()

        predictions = []
        for prompt, id, gt in zip(prompts, ids, true):
            predictions.append(self.run_example(prompt, id, gt))

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







