import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"
from functools import partial
import nltk
from src.contextual_bart import BartForContextualRecovery
from src.dataset_processor import load_all_data
from src.utils import SmartCollator, get_args, setuptokenizer
from src.dataset_processor import (
    ContextGenerationDataset,
)
from transformers import BartConfig
from src.model_utils import CustomTrainer, get_training_arguments
import torch
from src.config import DATASET_PATH
from transformers.trainer_callback import EarlyStoppingCallback
import pickle as pk
import pandas as pd
from src.dataset_processor import ContextualGenerationData,ContextGenerationDatasetBoundary


nltk.download("punkt")


def generate_tokenizer_and_data(
    data_dir,
    model_base,
    sep_token,
    max_seq_len,
    is_not_auto_encoder_data=False,
    use_random_restrictive=False,
    **unused_args,
):

    # load the dataset

    train_data_packet = load_all_data(data_dir, mode="train",new_format_data=use_random_restrictive)
    test_data_packet = load_all_data(data_dir, mode="dev",new_format_data=use_random_restrictive)

    print(f"Training Data size: {len(train_data_packet)}")
    print(f"Testing Data size: {len(test_data_packet)}")

    model_base = model_base
    tokenizer = setuptokenizer(
        model_base=model_base, special_tokens=[], additional_tokens=[sep_token]
    )
    # tokenizer.add_tokens([])
    
    context_data_class = ContextGenerationDataset
    if use_random_restrictive:
        context_data_class = ContextGenerationDatasetBoundary
    train_dataset = context_data_class(
        tokenizer=tokenizer,
        nb_records=len(train_data_packet),
        use_random_restrictive=use_random_restrictive,
        max_len=max_seq_len,
        context_seperator=sep_token,
        is_auto_encoder_data=not is_not_auto_encoder_data,
        use_special_token=True,
        section_boundary=(0.3, 0.60)
    )
    train_dataset.change_data_mode(1)
    train_dataset.set_record(train_data_packet)

    test_dataset = context_data_class(
        tokenizer=tokenizer,
        nb_records=len(test_data_packet),
        use_random_restrictive=use_random_restrictive,
        max_len=max_seq_len,
        context_seperator=sep_token,
        is_auto_encoder_data=not is_not_auto_encoder_data,
        use_special_token=True,
        
        section_boundary=(0.3, 0.60)
    )
    test_dataset.change_data_mode(1)
    test_dataset.set_record(test_data_packet)

    return train_dataset, test_dataset


def model_init(
    vocab_size,
    context_delimiter_id,
    model_base="facebook/bart-base",
    use_random_restriction=False,
    section_prob=(0.3, 0.60),#(0.45, 0.65),
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    def build_model():
        bart_config = BartConfig.from_pretrained(model_base)
        bart_config.context_delimiter_id = context_delimiter_id
        bart_config.use_random_restriction = use_random_restriction
        bart_config.section_prob = section_prob

        generator = BartForContextualRecovery.from_pretrained(
            model_base, config=bart_config, ignore_mismatched_sizes=True
        )

        # update the tokens
        generator.resize_token_embeddings(vocab_size)  # type: ignore
        return generator.to(device)  # type: ignore

    return build_model


if __name__ == "__main__":
    arguments = get_args()
    configs = vars(arguments)
    train_dataset, test_dataset = generate_tokenizer_and_data(**configs)

    training_arguments = get_training_arguments(**configs)
    context_delimiter_id = train_dataset.tokenizer.get_vocab()[arguments.sep_token]

    model_builder = model_init(
        vocab_size=len(train_dataset.tokenizer),
        context_delimiter_id=context_delimiter_id,
        model_base=arguments.model_base,
        use_random_restriction=arguments.use_random_restrictive,
    )

    custom_trainer = CustomTrainer(
        model_init=model_builder,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=SmartCollator(
            pad_token_id=train_dataset.tokenizer.pad_token_id,
            max_len=arguments.max_seq_len,
        ),  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    )

    custom_trainer.train()
    pk.dump(
        arguments,
        open(arguments.output_dir + "/" + arguments.run_id + "/train_args.ap", "wb"),
    )
