import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Union, Any

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
import evaluate

# Import for PEFT if added
# from peft import LoraConfig, get_peft_model, TaskType # Example for LoRA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Model registry - รายการโมเดลที่รองรับ
SUPPORTED_MODELS = {
    # ภาษาไทย
    "wangchanberta": {
        "base": "airesearch/wangchanberta-base-att-spm-uncased",
        "large": "airesearch/wangchanberta-base-wiki-newmm",
        "type": "masked_lm",
        "tokenizer_kwargs": {"use_fast": False}
    },
    "phayathaibert": {
        "base": "phayathai/phayathaibert",
        "type": "masked_lm",
        "tokenizer_kwargs": {}
    },
    "mt5": {
        "base": "google/mt5-small",
        "large": "google/mt5-base",
        "xl": "google/mt5-large",
        "type": "seq2seq_lm",
        "tokenizer_kwargs": {}
    },
    "xxl-thai-embedding": {
        "base": "xxl-thai/xxl-thai-embedding",
        "type": "masked_lm",
        "tokenizer_kwargs": {}
    },
    "thai2transformers": {
        "base": "flax-community/thai2transformers-bert-base-wisesight",
        "type": "masked_lm",
        "tokenizer_kwargs": {}
    },
    "thai-xlm-roberta": {
        "base": "flax-community/thai-xlm-roberta-base",
        "type": "masked_lm",
        "tokenizer_kwargs": {}
    },
    
    # โมเดลหลายภาษา
    "xlm-roberta": {
        "base": "xlm-roberta-base",
        "large": "xlm-roberta-large",
        "type": "masked_lm",
        "tokenizer_kwargs": {}
    },
    "mbert": {
        "base": "bert-base-multilingual-cased",
        "type": "masked_lm",
        "tokenizer_kwargs": {}
    },
}

# Task types และ metrics ที่เกี่ยวข้อง
TASK_CONFIGS = {
    "classification": {
        "model_type": "sequence_classification",
        "metrics": ["accuracy", "f1", "precision", "recall"],
        "data_collator": DataCollatorWithPadding,
        "output_mode": "classification",
        "model_args": {"problem_type": "single_label_classification"}
    },
    "multi_label": {
        "model_type": "sequence_classification",
        "metrics": ["accuracy", "f1", "precision", "recall"],
        "data_collator": DataCollatorWithPadding,
        "output_mode": "classification",
        "model_args": {"problem_type": "multi_label_classification"}
    },
    "regression": {
        "model_type": "sequence_classification",
        "metrics": ["mse", "rmse", "mae"],
        "data_collator": DataCollatorWithPadding,
        "output_mode": "regression",
        "model_args": {"problem_type": "regression"}
    },
    "summarization": {
        "model_type": "seq2seq_lm",
        "metrics": ["rouge"],
        "data_collator": DataCollatorForSeq2Seq,
        "output_mode": "text",
        "model_args": {}
    },
    "translation": {
        "model_type": "seq2seq_lm",
        "metrics": ["bleu", "sacrebleu"],
        "data_collator": DataCollatorForSeq2Seq,
        "output_mode": "text",
        "model_args": {}
    },
    "question_answering": {
        "model_type": "question_answering",
        "metrics": ["squad"],
        "data_collator": DataCollatorWithPadding,
        "output_mode": "qa",
        "model_args": {}
    },
    "language_modeling": {
        "model_type": "causal_lm",
        "metrics": ["perplexity"],
        "data_collator": DataCollatorForLanguageModeling,
        "output_mode": "text",
        "model_args": {}
    }
}

class ModelFineTuner:
    def __init__(
        self,
        model_name: str,
        model_size: str = "base",
        task_type: str = "classification",
        num_labels: Optional[int] = None,
        max_length: int = 512,
        custom_model_path: Optional[str] = None,
        use_peft: bool = False, # Add PEFT option
        peft_config: Optional[Dict[str, Any]] = None, # PEFT configuration
        **kwargs
    ):
        """
        ระบบสำหรับ fine-tune โมเดล NLP ภาษาไทย
        
        Args:
            model_name (str): ชื่อโมเดล (wangchanberta, mt5, etc.)
            model_size (str): ขนาดของโมเดล (base, large, xl)
            task_type (str): ประเภทงาน (classification, summarization, etc.)
            num_labels (int, optional): จำนวน labels สำหรับงาน classification
            max_length (int): ความยาวสูงสุดของข้อความ
            custom_model_path (str, optional): เส้นทางของโมเดลที่ไม่ได้อยู่ในรายการที่รองรับ
            use_peft (bool): ใช้ Parameter-Efficient Fine-Tuning (PEFT) หรือไม่
            peft_config (dict, optional): การตั้งค่า PEFT
        """
        self.model_name = model_name
        self.model_size = model_size
        self.task_type = task_type
        self.num_labels = num_labels
        self.max_length = max_length
        self.custom_model_path = custom_model_path
        self.use_peft = use_peft
        self.peft_config = peft_config
        self.kwargs = kwargs
        
        # ตั้งค่า device สำหรับการเทรน
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # โหลด task configuration
        if task_type not in TASK_CONFIGS:
            raise ValueError(f"Task type '{task_type}' is not supported. Available tasks: {list(TASK_CONFIGS.keys())}")
        
        self.task_config = TASK_CONFIGS[task_type]
        
        # โหลดโมเดลและ tokenizer
        if custom_model_path:
            self.model_path = custom_model_path
            self.model_config = {"type": self.task_config["model_type"], "tokenizer_kwargs": {}}
            logger.info(f"Using custom model from: {custom_model_path}")
        else:
            if model_name not in SUPPORTED_MODELS:
                raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(SUPPORTED_MODELS.keys())}")
            
            model_config = SUPPORTED_MODELS[model_name]
            if model_size not in model_config:
                raise ValueError(f"Size '{model_size}' is not available for model '{model_name}'. Available sizes: {list(model_config.keys() - ['type', 'tokenizer_kwargs'])}")
            
            self.model_path = model_config[model_size]
            self.model_config = model_config
            logger.info(f"Using model: {self.model_path}")
        
        # โหลด tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # สร้างโมเดล
        self.model = self._create_model()
        
        # ตั้งค่า metrics
        self.metrics = self._setup_metrics()
    
    def _load_tokenizer(self):
        """โหลด tokenizer ของโมเดล"""
        tokenizer_kwargs = self.model_config.get("tokenizer_kwargs", {})
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
            # เพิ่ม special tokens ถ้าจำเป็น
            if "pad_token" not in tokenizer.special_tokens_map and tokenizer.pad_token_id is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def _create_model(self):
        """สร้างโมเดลตาม task type ที่กำหนด"""
        model_type = self.task_config["model_type"]
        model_args = self.task_config["model_args"].copy()
        
        try:
            base_model = None
            if model_type == "sequence_classification":
                if self.num_labels is None:
                    raise ValueError("For classification tasks, num_labels must be specified")
                model_args["num_labels"] = self.num_labels
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path, **model_args
                )
            elif model_type == "seq2seq_lm":
                base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            elif model_type == "masked_lm":
                base_model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            elif model_type == "causal_lm":
                base_model = AutoModelForCausalLM.from_pretrained(self.model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Set model parameters
            if "pad_token" not in self.tokenizer.special_tokens_map and base_model.config.pad_token_id is None:
                if hasattr(base_model.config, "eos_token_id") and base_model.config.eos_token_id is not None:
                    base_model.config.pad_token_id = base_model.config.eos_token_id
            
            # Apply PEFT if enabled
            if self.use_peft and self.peft_config:
                # Example for LoRA, adapt as needed for other PEFT methods
                # if self.peft_config.get("peft_type", "").lower() == "lora":
                #     peft_task_type = None
                #     if self.task_type in ["classification", "multi_label", "regression"]:
                #         peft_task_type = TaskType.SEQ_CLS
                #     elif self.task_type in ["summarization", "translation"]:
                #         peft_task_type = TaskType.SEQ_2_SEQ_LM
                #     elif self.task_type == "language_modeling" and model_type == "causal_lm":
                #         peft_task_type = TaskType.CAUSAL_LM
                #     # Add more task type mappings as needed

                #     if peft_task_type:
                #         lora_config_params = self.peft_config.get("lora_config", {})
                #         config = LoraConfig(task_type=peft_task_type, **lora_config_params)
                #         base_model = get_peft_model(base_model, config)
                #         logger.info(f"Applied LoRA PEFT to the model.")
                #         base_model.print_trainable_parameters()
                #     else:
                #         logger.warning(f"PEFT task type not determined for task {self.task_type}. PEFT not applied.")
                # else:
                #    logger.warning(f"Unsupported PEFT type: {self.peft_config.get('peft_type')}. PEFT not applied.")
                logger.info("PEFT application logic would be here.") # Placeholder for PEFT logic

            return base_model
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def _setup_metrics(self):
        """ตั้งค่า metrics สำหรับการประเมินโมเดล"""
        metric_dict = {}
        for metric_name in self.task_config["metrics"]:
            try:
                metric_dict[metric_name] = evaluate.load(metric_name)
            except Exception as e:
                logger.warning(f"Failed to load metric {metric_name}: {str(e)}")
        return metric_dict
    
    def _tokenize_function(self, examples):
        """ฟังก์ชันสำหรับการแปลงข้อความเป็น tokens"""
        # สำหรับงาน classification และ regression
        if self.task_type in ["classification", "multi_label", "regression"]:
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
        # สำหรับงาน summarization
        elif self.task_type == "summarization":
            model_inputs = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            labels = self.tokenizer(
                examples["summary"],
                padding="max_length",
                truncation=True,
                max_length=min(self.max_length, 128)  # ความยาว summary มักจะสั้นกว่า
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        # สำหรับงาน translation
        elif self.task_type == "translation":
            model_inputs = self.tokenizer(
                examples["source_text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            labels = self.tokenizer(
                examples["target_text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        # สำหรับงาน question_answering
        elif self.task_type == "question_answering":
            questions = [q.strip() for q in examples["question"]]
            contexts = [c.strip() for c in examples["context"]]
            
            return self.tokenizer(
                questions,
                contexts,
                padding="max_length",
                truncation="only_second",
                max_length=self.max_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
        # สำหรับงาน language_modeling
        elif self.task_type == "language_modeling":
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
        else:
            raise ValueError(f"Unsupported task type for tokenization: {self.task_type}")
    
    def _compute_metrics(self, eval_pred):
        """คำนวณ metrics จากผลลัพธ์การประเมิน"""
        predictions, labels = eval_pred
        
        # สำหรับงาน classification
        if self.task_type in ["classification", "multi_label"]:
            if self.task_type == "multi_label":
                # แปลงเป็น binary predictions สำหรับ multi-label
                predictions = (predictions > 0).astype(int)
            else:
                # แปลงเป็น class indices สำหรับ single-label
                predictions = np.argmax(predictions, axis=1)
            
            results = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == "accuracy":
                    results[metric_name] = metric.compute(predictions=predictions, references=labels)["accuracy"]
                elif metric_name in ["f1", "precision", "recall"]:
                    results[metric_name] = metric.compute(predictions=predictions, references=labels, average="weighted")[metric_name]
            return results
        
        # สำหรับงาน regression
        elif self.task_type == "regression":
            predictions = predictions.squeeze()
            results = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == "mse":
                    results[metric_name] = metric.compute(predictions=predictions, references=labels)["mse"]
                elif metric_name == "rmse":
                    results[metric_name] = np.sqrt(metric.compute(predictions=predictions, references=labels)["mse"])
                elif metric_name == "mae":
                    results[metric_name] = metric.compute(predictions=predictions, references=labels)["mae"]
            return results
        
        # สำหรับงาน text generation (summarization, translation)
        elif self.task_type in ["summarization", "translation"]:
            pred_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            results = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == "rouge":
                    results.update(metric.compute(predictions=pred_str, references=label_str))
                elif metric_name in ["bleu", "sacrebleu"]:
                    results[metric_name] = metric.compute(predictions=pred_str, references=[[l] for l in label_str])[metric_name]
            return results
        
        # สำหรับงาน language_modeling
        elif self.task_type == "language_modeling":
            results = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == "perplexity":
                    results["perplexity"] = float(np.exp(np.mean(predictions.loss)))
            return results
        
        else:
            logger.warning(f"Metrics computation not defined for task type: {self.task_type}")
            return {}
    
    def prepare_dataset(
        self,
        data: Union[str, pd.DataFrame, Dict[str, Dataset]],
        text_column: str = "text",
        label_column: str = "label",
        train_test_split_ratio: Optional[float] = 0.2,
        validation_split: bool = True,
        seed: int = 42,
        **kwargs
    ) -> DatasetDict:
        """
        เตรียมข้อมูลสำหรับการ fine-tune
        
        Args:
            data: ข้อมูลอินพุต (ไฟล์, DataFrame, หรือ Dataset)
            text_column: ชื่อคอลัมน์ที่มีข้อความ
            label_column: ชื่อคอลัมน์ที่มี label
            train_test_split_ratio: สัดส่วนการแบ่งข้อมูล test (0.0-1.0)
            validation_split: ต้องการแบ่ง validation set หรือไม่
            seed: seed สำหรับการสุ่มแบ่งข้อมูล
            **kwargs: พารามิเตอร์เพิ่มเติมสำหรับงานเฉพาะ
        
        Returns:
            DatasetDict: ชุดข้อมูลที่พร้อมสำหรับการเทรน
        """
        # โหลดข้อมูล
        if isinstance(data, str):
            # โหลดจากไฟล์
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.tsv'):
                df = pd.read_csv(data, sep='\t')
            elif data.endswith('.json')หรือ data.endswith('.jsonl'):
                df = pd.read_json(data, lines=data.endswith('.jsonl'))
            else:
                try:
                    # พยายามโหลด dataset จาก Hugging Face Hub
                    dataset = load_dataset(data)
                    return dataset
                except:
                    raise ValueError(f"Unsupported file format: {data}")
            
            dataset = Dataset.from_pandas(df)
        
        elif isinstance(data, pd.DataFrame):
            # แปลงจาก pandas DataFrame
            dataset = Dataset.from_pandas(data)
        
        elif isinstance(data, dict) and all(isinstance(v, Dataset) for v in data.values()):
            # เป็น DatasetDict อยู่แล้ว
            return DatasetDict(data)
        
        else:
            raise ValueError("Unsupported data type. Please provide a file path, DataFrame, or Dataset.")
        
        # แบ่งข้อมูลเป็น train/test/validation
        if "train" not in dataset.column_names and "test" not in dataset.column_names:
            # แบ่งตาม ratio ที่กำหนด
            if validation_split:
                # แบ่งเป็น train/test/validation
                train_val_test = dataset.train_test_split(test_size=train_test_split_ratio, seed=seed)
                train_val = train_val_test["train"].train_test_split(test_size=train_test_split_ratio, seed=seed)
                
                dataset_dict = DatasetDict({
                    "train": train_val["train"],
                    "validation": train_val["test"],
                    "test": train_val_test["test"]
                })
            else:
                # แบ่งเป็นแค่ train/test
                splits = dataset.train_test_split(test_size=train_test_split_ratio, seed=seed)
                dataset_dict = DatasetDict({
                    "train": splits["train"],
                    "test": splits["test"]
                })
        else:
            # เป็น DatasetDict อยู่แล้ว
            dataset_dict = dataset
        
        # แปลงรูปแบบข้อมูลตาม task type
        column_mapping = {}
        
        if self.task_type in ["classification", "multi_label", "regression"]:
            column_mapping = {text_column: "text", label_column: "label"}
        
        elif self.task_type == "summarization":
            column_mapping = {
                kwargs.get("document_column", text_column): "text",
                kwargs.get("summary_column", "summary"): "summary"
            }
        
        elif self.task_type == "translation":
            column_mapping = {
                kwargs.get("source_column", "source_text"): "source_text",
                kwargs.get("target_column", "target_text"): "target_text" 
            }
        
        elif self.task_type == "question_answering":
            column_mapping = {
                kwargs.get("question_column", "question"): "question",
                kwargs.get("context_column", "context"): "context",
                kwargs.get("answer_column", "answer"): "answer"
            }
        
        # Rename columns
        for split in dataset_dict:
            for old_col, new_col in column_mapping.items():
                if old_col in dataset_dict[split].column_names:
                    dataset_dict[split] = dataset_dict[split].rename_column(old_col, new_col)
        
        # แปลงข้อมูลเป็น tokens
        tokenized_dataset = dataset_dict.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset_dict["train"].column_names
                           if col not in ["input_ids", "attention_mask", "token_type_ids", "labels"]]
        )
        
        return tokenized_dataset
    
    def train(
        self,
        dataset: DatasetDict,
        output_dir: str,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        use_early_stopping: bool = True,
        early_stopping_patience: int = 3,
        fp16: bool = False,
        report_to: Optional[List[str]] = None, # For experiment tracking (e.g., ["wandb", "tensorboard"])
        **kwargs
    ):
        """
        Fine-tune โมเดลด้วยข้อมูลที่เตรียมไว้
        
        Args:
            dataset: ชุดข้อมูลที่เตรียมไว้สำหรับการเทรน
            output_dir: ไดเรกทอรีสำหรับบันทึกโมเดล
            batch_size: ขนาด batch
            learning_rate: อัตราการเรียนรู้
            num_epochs: จำนวน epochs
            warmup_ratio: สัดส่วนการ warmup
            weight_decay: ค่า weight decay
            evaluation_strategy: กลยุทธ์การประเมิน ("no", "steps", "epoch")
            save_strategy: กลยุทธ์การบันทึกโมเดล ("no", "steps", "epoch")
            load_best_model_at_end: โหลดโมเดลที่ดีที่สุดเมื่อเทรนเสร็จ
            use_early_stopping: ใช้ early stopping หรือไม่
            early_stopping_patience: จำนวน epochs ที่ต้องรอก่อนหยุดเทรน
            fp16: ใช้ mixed precision training หรือไม่
            report_to: บริการติดตามการทดลอง (เช่น ["wandb", "tensorboard"])
            **kwargs: พารามิเตอร์เพิ่มเติม
            
        Returns:
            TrainingOutput: ผลลัพธ์การเทรน
        """
        # สร้างไดเรกทอรีเป้าหมาย
        os.makedirs(output_dir, exist_ok=True)
        
        # ตรวจสอบความพร้อมของข้อมูล
        if not isinstance(dataset, DatasetDict):
            raise ValueError("Dataset must be a DatasetDict")
        
        if "train" not in dataset:
            raise ValueError("Dataset must contain a 'train' split")
        
        # กำหนด training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=kwargs.get("save_total_limit", 2),
            metric_for_best_model=kwargs.get("metric_for_best_model", "loss"),
            greater_is_better=kwargs.get("greater_is_better", False),
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=kwargs.get("logging_steps", 100),
            fp16=fp16 and torch.cuda.is_available(),
            report_to=report_to if report_to else ["tensorboard"], # Use provided report_to or default
            dataloader_num_workers=kwargs.get("num_workers", 2),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            disable_tqdm=False,
            push_to_hub=kwargs.get("push_to_hub", False),
            hub_model_id=kwargs.get("hub_model_id", None),
            hub_token=kwargs.get("hub_token", None),
            # Add deepspeed config if needed
            # deepspeed=kwargs.get("deepspeed_config_path", None)
        )
        
        # ตั้งค่า data collator
        data_collator_cls = self.task_config["data_collator"]
        
        if self.task_type in ["summarization", "translation"]:
            data_collator = data_collator_cls(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
        elif self.task_type == "language_modeling":
            data_collator = data_collator_cls(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
        else:
            data_collator = data_collator_cls(tokenizer=self.tokenizer)
        
        # สร้าง callbacks
        callbacks = []
        if use_early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            ))
        
        # สร้าง trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", dataset.get("test", None)),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics if len(self.metrics) > 0 else None,
            callbacks=callbacks
        )
        
        # บันทึกรายละเอียดการเทรน
        train_config = {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "model_path": self.model_path,
            "task_type": self.task_type,
            "max_length": self.max_length,
            "num_labels": self.num_labels,
            "training_args": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                # เพิ่มคอนฟิกอื่นๆ
            }
        }
        
        with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(train_config, f, ensure_ascii=False, indent=2)
        
        # เริ่มการเทรน
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # บันทึกโมเดลสุดท้าย
        # If using PEFT, need to handle saving adapter weights correctly
        # For example, if model is a PeftModel:
        # if self.use_peft:
        #     trainer.model.save_pretrained(output_dir)
        # else:
        #     trainer.save_model(output_dir)
        trainer.save_model(output_dir) # Original saving, adjust if PEFT is used.
        self.tokenizer.save_pretrained(output_dir)
        
        # บันทึกผลลัพธ์การเทรน
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # ประเมินโมเดลกับชุดข้อมูล test
        if "test" in dataset:
            logger.info("Evaluating model on test set...")
            test_results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
            trainer.log_metrics("test", test_results)
            trainer.save_metrics("test", test_results)
        
        return {
            "model_path": output_dir,
            "train_results": metrics,
            "test_results": test_results if "test" in dataset else None
        }
    
    def save(self, output_dir: str):
        """บันทึกโมเดลและ tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # บันทึกข้อมูลเพิ่มเติม
        model_info = {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "task_type": self.task_type,
            "num_labels": self.num_labels,
            "max_length": self.max_length
        }
        
        with open(os.path.join(output_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
        
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLP models for Thai language")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., wangchanberta, mt5)")
    parser.add_argument("--model_size", type=str, default="base", help="Model size (base, large, xl)")
    parser.add_argument("--task", type=str, default="classification", help="Task type (classification, summarization, etc.)")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labels for classification tasks")
    parser.add_argument("--input_file", type=str, required=True, help="Input data file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing text data")
    parser.add_argument("--label_column", type=str, default="label", help="Column name containing labels")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--custom_model", type=str, default=None, help="Path to custom model")
    parser.add_argument("--use_peft", action="store_true", help="Use Parameter-Efficient Fine-Tuning (PEFT)")
    parser.add_argument("--peft_config_file", type=str, default=None, help="Path to PEFT configuration JSON file")
    parser.add_argument("--report_to", nargs="+", default=["tensorboard"], help="Experiment tracking services (e.g., wandb tensorboard)")
    
    args = parser.parse_args()

    peft_config_dict = None
    if args.use_peft and args.peft_config_file:
        try:
            with open(args.peft_config_file, 'r') as f:
                peft_config_dict = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load PEFT config from {args.peft_config_file}: {e}")
            return
    elif args.use_peft and not args.peft_config_file:
        logger.warning("`--use_peft` is set but `--peft_config_file` is not provided. PEFT will not be applied unless a default config is handled internally.")
        # Or set a default PEFT config here if desired
    
    # สร้าง fine-tuner
    fine_tuner = ModelFineTuner(
        model_name=args.model,
        model_size=args.model_size,
        task_type=args.task,
        num_labels=args.num_labels,
        max_length=args.max_length,
        custom_model_path=args.custom_model,
        use_peft=args.use_peft,
        peft_config=peft_config_dict
    )
    
    # เตรียมข้อมูล
    dataset = fine_tuner.prepare_dataset(
        data=args.input_file,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    # เทรนโมเดล
    results = fine_tuner.train(
        dataset=dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        report_to=args.report_to
    )
    
    logger.info(f"Training completed. Model saved to {args.output_dir}")
    logger.info(f"Training results: {results['train_results']}")
    if results['test_results']:
        logger.info(f"Test results: {results['test_results']}")

if __name__ == "__main__":
    main()
