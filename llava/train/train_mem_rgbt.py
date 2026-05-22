from llava.train.train_rgbt import train as train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
