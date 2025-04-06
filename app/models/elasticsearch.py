es_index_body={
    "settings": {
        "index": {
            "max_ngram_diff": 10
        },
        "analysis": {
            "tokenizer": {
                "ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 2,
                    "max_gram": 5,
                    "token_chars": ["letter", "digit"]
                },
                "nori_tokenizer": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed"
                }
            },
            "analyzer": {
                "ngram_analyzer": {
                    "type": "custom",
                    "tokenizer": "ngram_tokenizer"
                },
                "nori_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer"
                },
            }
        }
    },
    "mappings": {
        "properties": {
            "name": {
                "type": "text",
                "fields": {
                    "nori": {
                        "type": "text",
                        "analyzer": "nori_analyzer"
                    },
                    "ngram": {
                        "type": "text",
                        "analyzer": "ngram_analyzer",
                        "search_analyzer": "ngram_analyzer"
                    }
                }
            }
        }
    }
}