"""
Lab 0 mini: the *consumer* side of the nanochat tokenizer.

Distilled from nanochat/tokenizer.py (tag speedrun-4xh200-baseline). Kept, semantically identical:
  - loading the trained tokenizer from disk (a pickled tiktoken.Encoding)
  - SPECIAL_TOKENS and how they get ids
  - encode / decode / encode_special / bos
  - render_conversation  (Chat SFT: ids + loss mask)
  - render_for_completion (RL / eval: prompt priming the assistant)
  - token_bytes           (the per-token byte count used by bits-per-byte eval)
Cut, per SPEC.md: train_from_iterator (rustbpe training), from_pretrained (gpt2/gpt4 tokenizers),
save, and the tok_train / tok_eval scripts.

Every function notes the shape of what flows through it. Everything here is Python lists of ints
except token_bytes, which is the only tensor.
"""

import copy
import os
import pickle
from functools import lru_cache

import tiktoken
import torch

# --------------------------------------------------------------------------------------------------
# Where the trained tokenizer lives. The reference resolves this via nanochat.common.get_base_dir()
# ($NANOCHAT_BASE_DIR, default ~/.cache/nanochat) + "tokenizer"; the lab pins the SPEC.md path.
TOKENIZER_DIR = "/svl/u/koven/nanochat_data/tokenizer"

# nanochat/tokenizer.py:9-21
# Special tokens live *after* all the BPE merges in the id space (see from_directory below).
# Only <|bos|> is used in pretraining; the other 8 exist for rendering conversations in SFT/RL.
SPECIAL_TOKENS = [
    "<|bos|>",              # every document / row starts with this; delimits documents
    "<|user_start|>",       # user turn
    "<|user_end|>",
    "<|assistant_start|>",  # assistant turn
    "<|assistant_end|>",
    "<|python_start|>",     # assistant calls the python tool
    "<|python_end|>",
    "<|output_start|>",     # python tool's output fed back to the assistant
    "<|output_end|>",
]


class Tokenizer:
    """nanochat/tokenizer.py:34 RustBPETokenizer, consumer side only.

    self.enc is a tiktoken.Encoding: it holds
      - mergeable_ranks: dict[bytes, int], the BPE merge table (token bytes -> rank == token id)
      - special_tokens:  dict[str, int],   name -> id, ids continue right after mergeable_ranks
      - pat_str:         the regex that pre-splits text into chunks before BPE runs on each chunk
    """

    def __init__(self, enc, bos_token):                     # nanochat/tokenizer.py:37
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)  # int

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):   # nanochat/tokenizer.py:64
        # tokenizer.pkl is the whole tiktoken.Encoding, pickled by tok_train.py after training.
        # Ids: 0..(n_merges-1) are BPE tokens, then the 9 SPECIAL_TOKENS in the order listed above,
        # so vocab_size == n_merges + 9 (== 32768 for the speedrun tokenizer).
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    # ---- vocab / special tokens --------------------------------------------------------------
    def get_vocab_size(self):                               # nanochat/tokenizer.py:80
        return self.enc.n_vocab                             # int

    def get_special_tokens(self):                           # nanochat/tokenizer.py:83
        return self.enc.special_tokens_set                  # set[str]

    @lru_cache(maxsize=32)
    def encode_special(self, text):                         # nanochat/tokenizer.py:89
        # exact-match lookup of one special token string -> its id. Raises if `text` isn't one.
        return self.enc.encode_single_token(text)           # int

    def get_bos_token_id(self):                             # nanochat/tokenizer.py:93
        return self.bos_token_id                            # int

    # ---- text <-> ids -----------------------------------------------------------------------
    def encode(self, text, prepend=None, append=None, num_threads=8):   # nanochat/tokenizer.py:96
        """str -> list[int]  or  list[str] -> list[list[int]] (ragged; rows differ in length).

        encode_ordinary means: special-token strings inside `text` are NOT recognised, they get
        BPE'd like any other text. Special tokens only enter a sequence via prepend/append or via
        render_conversation. That is what makes "<|bos|>" typed by a user harmless.
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)                        # list[int], len = n_tokens
            if prepend is not None:
                ids.insert(0, prepend_id)                               # len n_tokens + 1
            if append is not None:
                ids.append(append_id)                                   # len n_tokens + (1|2)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)  # list[list[int]]
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs):                    # nanochat/tokenizer.py:123
        return self.encode(*args, **kwargs)

    def decode(self, ids):                                  # nanochat/tokenizer.py:126
        # list[int] -> str. Special tokens decode to their literal "<|...|>" text.
        # Bytes that don't form valid UTF-8 (a token can be a partial multibyte char) get U+FFFD.
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):          # nanochat/tokenizer.py:129
        return self.enc.decode_single_token_bytes(token_id)  # bytes, len >= 1 (every merge concatenates two
                                                            # non-empty tokens); no principled upper bound,
                                                            # only "longest chunk that got fully merged"

    # ---- conversations -> ids (+ mask) -------------------------------------------------------
    def render_conversation(self, conversation, max_tokens=2048):       # nanochat/tokenizer.py:140
        """One chat conversation -> (ids, mask), both list[int] of the same length L <= max_tokens.

        conversation = {"messages": [{"role": "user"|"assistant"|"system", "content": ...}, ...]}
        Roles must strictly alternate user, assistant, user, ... (a leading system message is
        folded into the first user message). Assistant content is either a str or a list of parts
        {"type": "text"|"python"|"python_output", "text": str}.

        mask[i] == 1  <=> position i is a token the assistant is trained to *produce*.
        Note the mask marks the token at position i itself; the SFT dataloader shifts it by one
        (mask[1:]) so that it lines up with the *targets* (scripts/chat_sft.py:285-290).
        """
        ids, mask = [], []                                              # both grow to length L

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # A leading system message is merged into the following user message with "\n\n" between.
        # There is no <|system_start|> token: the model never sees "system" as a distinct role.
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)  # avoid mutating the caller's dict
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)                                              # ids = [bos]
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)                        # list[int]
                add_tokens(user_start, 0)                               # <|user_start|>  mask 0
                add_tokens(value_ids, 0)                                # user text       mask 0
                add_tokens(user_end, 0)                                 # <|user_end|>    mask 0
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)                          # <|assistant_start|> mask 0:
                                                                        # it's the *prompt* for the
                                                                        # assistant, not its output
                if isinstance(content, str):
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)                            # assistant text  mask 1
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)                    # mask 1
                        elif part["type"] == "python":
                            # the assistant *decides* to call the tool, so the delimiters are
                            # supervised too
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # comes from the interpreter at test time, so nothing here is supervised
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)                            # <|assistant_end|> mask 1:
                                                                        # the model must learn to
                                                                        # *stop*

        # hard truncation; the tail (usually the last assistant turn) is simply dropped
        ids = ids[:max_tokens]                                          # len L = min(len, max_tokens)
        mask = mask[:max_tokens]                                        # same L
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):   # nanochat/tokenizer.py:226
        """Debug helper: green = mask 1 (trained on), red = mask 0. Tokens joined with '|'."""
        RED, GREEN, GRAY, RESET = '\033[91m', '\033[92m', '\033[90m', '\033[0m'
        tokens = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):          # nanochat/tokenizer.py:241
        """Prompt for generation: drop the final assistant message, render the rest, and append
        <|assistant_start|> so the model is primed to produce the assistant turn.
        Returns ids only (list[int]); there is no mask because nothing is trained here.
        Used by chat_eval (scripts/chat_eval.py:41,106) and the RL setting."""
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()
        ids, mask = self.render_conversation(conversation)              # ids: list[int] len L
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)                                     # len L + 1
        return ids


# --------------------------------------------------------------------------------------------------
# token_bytes: the bridge between "loss in nats per token" and "bits per byte".
# Written once by scripts/tok_train.py:78-91, read by base_train / base_eval / chat_sft, consumed
# in nanochat/loss_eval.py:evaluate_bpb as   bpb = sum(nll[y]) / (ln 2 * sum(token_bytes[y])).
def get_token_bytes(device="cpu", tokenizer_dir=TOKENIZER_DIR):        # nanochat/tokenizer.py:270
    """-> int32 tensor of shape (vocab_size,). token_bytes[t] = number of UTF-8 bytes token t
    decodes to; the 9 special tokens are stored as 0 so they contribute no bytes to bpb."""
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)               # (vocab_size,) int32
    return token_bytes


def get_tokenizer():                                        # nanochat/tokenizer.py:264
    return Tokenizer.from_directory(TOKENIZER_DIR)


if __name__ == "__main__":
    # smoke test only: loads and checks the invariants the comments above claim. Prints no values.
    tok = get_tokenizer()
    tb = get_token_bytes()
    assert tb.shape == (tok.get_vocab_size(),) and tb.dtype == torch.int32
    assert set(SPECIAL_TOKENS) == tok.get_special_tokens()
    n_merges = tok.get_vocab_size() - len(SPECIAL_TOKENS)
    assert [tok.encode_special(s) for s in SPECIAL_TOKENS] == list(range(n_merges, tok.get_vocab_size()))
    assert all(tb[tok.encode_special(s)] == 0 for s in SPECIAL_TOKENS)
    text = "hello world, 你好 123456"
    assert tok.decode(tok.encode(text)) == text
    conv = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    ids, mask = tok.render_conversation(conv)
    assert len(ids) == len(mask) and ids[0] == tok.get_bos_token_id()
    assert tok.render_for_completion(conv) == ids[:len(tok.render_conversation({"messages": conv["messages"][:1]})[0])] + [tok.encode_special("<|assistant_start|>")]
    print("lab0 mini: ok")
