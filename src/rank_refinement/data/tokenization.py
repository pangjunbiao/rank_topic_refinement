from __future__ import annotations
import logging
import re
from pathlib import Path
from ..runtime.artifacts import digest, file_hash, write_text

HAS_LETTER = re.compile(r"[A-Za-z\u3400-\u9fff]")


class ChineseTokenizer:
    def __init__(self, root, policy):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
            import jieba
            import stopwordsiso
        jieba.setLogLevel(logging.WARNING)
        dictionary = Path(jieba.__file__).parent / "dict.txt"
        protected = {t.lower() for t in policy["preserve_tokens"]}
        self.stopwords = set(stopwordsiso.stopwords("zh")) - protected
        self.policy = policy
        self.dictionary = dictionary
        self.tokenizer = jieba.Tokenizer(str(dictionary))
        cache = Path(root) / "data" / "cache" / "tokenization"
        cache.mkdir(parents=True, exist_ok=True)
        self.tokenizer.tmp_dir = str(cache)
        self.tokenizer.cache_file = "jieba_" + file_hash(dictionary)[:16] + ".cache"
        self.tokenizer.initialize()
        self.asset_manifest = {"jieba_version": "0.42.1", "dictionary_sha256": file_hash(dictionary),
                               "stopwords_source": "https://github.com/stopwords-iso/stopwords-iso",
                               "stopwordsiso_version": "0.6.1", "effective_stopwords_sha256": digest(sorted(self.stopwords)),
                               "effective_stopwords_count": len(self.stopwords), "protected_tokens": sorted(protected),
                               "hmm": policy["hmm"], "dictionary_additions": "none"}

    def tokenize(self, text):
        tokens = []
        for word in self.tokenizer.cut(text, HMM=bool(self.policy["hmm"]), cut_all=False):
            word = word.strip().lower()
            if not word or word in self.stopwords:
                continue
            if self.policy["drop_pure_numbers"] and word.isdecimal():
                continue
            if HAS_LETTER.search(word):
                tokens.append(word)
        return tokens

    def save_assets(self, folder):
        write_text(Path(folder) / "effective_stopwords_zh.txt", "\n".join(sorted(self.stopwords)) + "\n")
        write_text(Path(folder) / "protected_zh_tokens.txt", "\n".join(self.asset_manifest["protected_tokens"]) + "\n")
