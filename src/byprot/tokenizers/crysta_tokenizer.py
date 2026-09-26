# dmlm/tokenizers/crysta_tokenizer.py

import pickle
import re
from byprot.crystallm._tokenizer import CIFTokenizer
from byprot import utils

log = utils.get_logger(__name__)

class CrystaTokenizerWrapper:
    """Wrapper around CrystaLLM's CIFTokenizer or meta.pkl vocab."""

    def __init__(self, meta=None):
        """
        Args:
            meta: dict from meta.pkl, containing vocab + special tokens.
        """
        log.info(f'Function CrystaTokenizerWrapper.__init__ Start!')
        if meta is not None:
            log.info(f'Function CrystaTokenizerWrapper.__init__: use pre-built vocab from meta.pkl!')

            # use pre-built vocab from meta.pkl
            self._id_to_token = meta["itos"]  # list: index -> token
            self._token_to_id = meta["stoi"]  # dict: token -> index

            # special tokens
            self.pad_token_id = meta.get("pad_token_id")
            self.eos_token_id = meta.get("eos_token_id")
            self.bos_token_id = meta.get("bos_token_id")
            self.mask_token_id = meta.get("mask_token_id")

            self.pad_token = self._id_to_token[self.pad_token_id] if self.pad_token_id is not None else "<pad>"
            self.eos_token = self._id_to_token[self.eos_token_id] if self.eos_token_id is not None else "<eos>"
            self.bos_token = self._id_to_token[self.bos_token_id] if self.bos_token_id is not None else "<bos>"
            self.mask_token = self._id_to_token[self.mask_token_id] if self.mask_token_id is not None else "<mask>"
            self.unk_token = "<unk>"

        else:
            log.info(f'Function CrystaTokenizerWrapper.__init__: build vocab from CIFTokenizer!')
            # fallback: build vocab from CIFTokenizer
            self.base_tokenizer = CIFTokenizer()

            # inherit vocab
            self._token_to_id = self.base_tokenizer.token_to_id
            self._id_to_token = self.base_tokenizer.id_to_token

            # define special tokens
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"
            self.bos_token = "<bos>"
            self.mask_token = "<mask>"
            self.unk_token = "<unk>"
            #  self.x_token = "X"

            # assign ids (append to vocab)
            max_id = max(self._id_to_token.keys())
            self.unk_token_id = max_id
            self.pad_token_id = max_id + 1
            self.eos_token_id = max_id + 2
            self.bos_token_id = max_id + 3
            self.mask_token_id = max_id + 4
            #  self.x_token_id = max_id + 5

            # extend mappings
            self._token_to_id[self.unk_token] = self.unk_token_id
            self._token_to_id[self.pad_token] = self.pad_token_id
            self._token_to_id[self.eos_token] = self.eos_token_id
            self._token_to_id[self.bos_token] = self.bos_token_id
            self._token_to_id[self.mask_token] = self.mask_token_id
            #  self._token_to_id[self.x_token] = self.x_token_id

            self._id_to_token[self.unk_token_id] = self.unk_token
            self._id_to_token[self.pad_token_id] = self.pad_token
            self._id_to_token[self.eos_token_id] = self.eos_token
            self._id_to_token[self.bos_token_id] = self.bos_token
            self._id_to_token[self.mask_token_id] = self.mask_token
            #  self._id_to_token[self.x_token_id] = self.x_token

        log.info(f'Function CrystaTokenizerWrapper.__init__ Done!')

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def __len__(self):
        return self.vocab_size  # lets the network pick up the vocab size automatically
    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def encode(self, tokens):
        return [self._token_to_id.get(t, self._token_to_id.get(self.unk_token)) for t in tokens]

    # ------------------- CSP: composition -> data prefix tokens ------------------- #
    @staticmethod
    def _split_composition(composition):
        """Split a reduced formula (e.g. 'NaCl' / 'LiFePO4') into token strings using the vocab rules.

        Consistent with the tokenization of the ``data_<formula>`` first line of the training data:
        - ``[A-Z][a-z]*`` element symbols (Na, Cl, Fe...)
        - each digit is its own token (the vocab only contains single digits 0-9)

        Only element symbols and digits are accepted; any other character ('.', '-', '(', space...)
        raises immediately, so that an unknown string is never silently encoded as <unk> and used
        as a condition.
        """
        comp = composition.strip()
        if comp.startswith("data_"):
            comp = comp[len("data_"):]
        if not comp:
            raise ValueError(f"Empty composition: {composition!r}")
        tokens = re.findall(r"([A-Z][a-z]*|\d)", comp)
        if "".join(tokens) != comp:
            raise ValueError(
                f"Composition {composition!r} contains characters that are not "
                f"element symbols or digits (got tokens {tokens!r})."
            )
        return tokens

    def tokenize_composition(self, composition):
        """Return list[str]: 'data_' + the split element/digit tokens (without the trailing '\\n').

        Raises if any token is missing from the vocab (or if its id does not decode back);
        <unk> is never used silently.
        """
        tokens = ["data_"] + self._split_composition(composition)
        for t in tokens:
            tid = self._token_to_id.get(t)
            if tid is None:
                raise ValueError(
                    f"Token {t!r} (from composition {composition!r}) not in vocab! "
                    f"Cannot build a CSP condition with unknown tokens."
                )
            # guards against remappings such as the space-group '_sg' token,
            # which would break id -> token round-tripping
            if self._id_to_token.get(tid) != t:
                raise ValueError(
                    f"Token {t!r} (from composition {composition!r}) does not "
                    f"decode back to itself (id {tid} -> {self._id_to_token.get(tid)!r})."
                )
        return tokens

    def encode_composition(self, composition, add_newline=True):
        """CSP helper: encode a composition into prefix token ids identical to the training data.

        Args:
            composition: composition as a reduced formula, either 'NaCl' or 'data_NaCl'.
            add_newline: whether to append the '\\n' token (in the training data the data_
                line ends with '\\n', and the condition span contains that '\\n'; the model
                then continues writing from 'loop_').

        Returns:
            token_ids: list[int]
            token_strings: list[str]
        """
        token_strings = self.tokenize_composition(composition)
        if add_newline:
            nl = self._token_to_id.get("\n")
            if nl is None:
                raise ValueError("'\\n' token not in vocab; cannot append it to the condition.")
            token_strings.append("\n")
        token_ids = [self._token_to_id[t] for t in token_strings]
        return token_ids, token_strings

    def decode(self, ids):
        return ''.join([self._id_to_token.get(i, self.unk_token) for i in ids])

    def batch_decode(self, batches):
        return [self.decode(batch) for batch in batches]
