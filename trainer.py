import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


nltk.download("punkt")


def generate_tokenizer_and_data(args):

    # load the dataset

    train_data_packet = load_all_data(DATASET_PATH, mode="train")
    test_data_packet = load_all_data(DATASET_PATH, mode="dev")

    print(f"Training Data size: {len(train_data_packet)}")
    print(f"Training Data size: {len(test_data_packet)}")

    model_base = args.model_base
    tokenizer = setuptokenizer(
        model_base=model_base,
        special_tokens=[],
    )
    tokenizer.add_tokens(["[SEP]"])

    train_dataset = ContextGenerationDataset(
        tokenizer=tokenizer,
        nb_records=len(train_data_packet),
        use_random_restrictive=args.use_random_restrictive
    )
    train_dataset.change_data_mode(1)
    train_dataset.set_record(train_data_packet)

    test_dataset = ContextGenerationDataset(
        tokenizer=tokenizer,
        nb_records=len(test_data_packet),
        use_random_restrictive=args.use_random_restrictive
    )
    test_dataset.change_data_mode(1)
    test_dataset.set_record(test_data_packet)

    return train_dataset, test_dataset


def model_init(
    vocab_size,
    context_delimiter_id,
    model_base="facebook/bart-base",
    use_random_restriction=False,
    section_prob=(0.45, 0.65),
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
    args = get_args()
    train_dataset, test_dataset = generate_tokenizer_and_data(args)
    training_arguments = get_training_arguments(args)
    context_delimiter_id = train_dataset.tokenizer.get_added_vocab()["[SEP]"]

    model_builder = model_init(
        vocab_size=len(train_dataset.tokenizer),
        context_delimiter_id=context_delimiter_id,
        model_base=args.model_base,
        use_random_restriction=args.use_random_restrictive,
    )

    custom_trainer = CustomTrainer(
        model_init=model_builder,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=SmartCollator(
            pad_token_id=train_dataset.tokenizer.pad_token_id, max_len=args.max_seq_len
        ),  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    custom_trainer.train()
    pk.dump(args, open(args.output_dir + "/" + args.run_id + "/train_args.ap", "wb"))
