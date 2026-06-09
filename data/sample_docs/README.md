# Seed Knowledge Base Corpus

Curated ArXiv papers for ResearchIQ Phase 2 RAG demo.

## Topics

| Category | Documents |
|----------|-----------|
| Transformers | Attention Is All You Need, BERT |
| LLMs | GPT-3, LLaMA, InstructGPT, GPT-4 |
| Vision | ViT, CLIP, SAM |
| Architecture | Mamba |

## Setup

```bash
python scripts/download_seed_docs.py
python scripts/seed_knowledge_base.py
```

## Demo queries

1. What are the key components of the original transformer architecture?
2. How do vision transformers differ from CNN-based approaches?
3. What are recent alternatives to standard transformer attention?
4. How are modern LLMs aligned to follow instructions?

## Smoke-test retrieval queries

- `self-attention mechanism in transformers` → Attention Is All You Need
- `bidirectional pre-training for NLP` → BERT
- `vision transformer patch embeddings` → ViT
- `image text contrastive learning` → CLIP
- `linear time sequence modeling alternative to transformers` → Mamba
