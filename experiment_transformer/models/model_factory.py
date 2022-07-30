
from models.bert.modeling_bert import BertForSequenceClassification



def from_pretrained(model_args, config):
    if(model_args.model_name_or_path.startswith("bert")):
        return BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            )

    raise Exception(f"Unknown model {model_args.model_name_or_path}.")