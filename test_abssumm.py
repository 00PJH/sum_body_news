from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration



# Load Model and Tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")


def newSum(text):
    # Encoding
    input_text = text

    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1026)

    # Generate Summary Text Ids
    summary_text_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        # length_penalty=1.0,
        # max_length=300,
        # min_length=12,
        # num_beams=6,
        # repetition_penalty=1.5,
        # no_repeat_ngram_size=15,

        length_penalty=0.6,       # 길이 페널티 조정
        max_length=250,           # 최대 길이 조정
        min_length=75,            # 최소 길이 조정
        num_beams=9,              # 빔 수 증가
        repetition_penalty=2.0,   # 반복 페널티 증가
        no_repeat_ngram_size=5,   # 반복되지 않도록 할 n-그램 크기 조정
        )

    # Decoding Text Ids
    # print(tokenizer.decode(summary_text_ids[0], skip_special_tokens=True))
    return tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

