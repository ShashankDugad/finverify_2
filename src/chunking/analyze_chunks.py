import json
import tiktoken
from collections import defaultdict

CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
enc = tiktoken.get_encoding("cl100k_base")

# Stats
token_counts = []
chunks_per_transcript = defaultdict(int)
companies = set()
empty_chunks = 0
short_chunks = 0  # < 100 tokens
overlap_test_pairs = []

with open(CHUNKS_FILE, "r") as f:
    prev_chunk = None
    for i, line in enumerate(f):
        chunk = json.loads(line)
        text = chunk["text"]
        tokens = enc.encode(text)
        token_count = len(tokens)
        token_counts.append(token_count)
        
        key = f"{chunk['company']}_{chunk['year']}_Q{chunk['quarter']}"
        chunks_per_transcript[key] = chunk["total_chunks"]
        companies.add(chunk["company"])
        
        if token_count == 0:
            empty_chunks += 1
        if token_count < 100:
            short_chunks += 1
        
        # Test overlap between consecutive chunks of same transcript
        if prev_chunk and prev_chunk["company"] == chunk["company"] and \
           prev_chunk["year"] == chunk["year"] and prev_chunk["quarter"] == chunk["quarter"] and \
           chunk["chunk_index"] == prev_chunk["chunk_index"] + 1:
            if len(overlap_test_pairs) < 5:
                prev_tokens = enc.encode(prev_chunk["text"])
                curr_tokens = enc.encode(chunk["text"])
                # Check if end of prev overlaps with start of curr
                overlap = 0
                for j in range(min(100, len(prev_tokens), len(curr_tokens))):
                    if prev_tokens[-(j+1):] == curr_tokens[:j+1]:
                        overlap = j + 1
                overlap_test_pairs.append({
                    "prev_end": prev_chunk["text"][-200:],
                    "curr_start": chunk["text"][:200],
                    "expected_overlap": 50
                })
        
        prev_chunk = chunk
        
        if i >= 28630:
            break

print("=" * 60)
print("CHUNK QUALITY REPORT")
print("=" * 60)

print(f"\n1. BASIC STATS")
print(f"   Total chunks: {len(token_counts)}")
print(f"   Total companies: {len(companies)}")
print(f"   Total transcripts: {len(chunks_per_transcript)}")

print(f"\n2. TOKEN DISTRIBUTION")
print(f"   Min tokens: {min(token_counts)}")
print(f"   Max tokens: {max(token_counts)}")
print(f"   Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
print(f"   Empty chunks (0 tokens): {empty_chunks}")
print(f"   Short chunks (<100 tokens): {short_chunks}")

# Token count buckets
buckets = {"<100": 0, "100-300": 0, "300-500": 0, "500-520": 0, ">520": 0}
for tc in token_counts:
    if tc < 100:
        buckets["<100"] += 1
    elif tc < 300:
        buckets["100-300"] += 1
    elif tc < 500:
        buckets["300-500"] += 1
    elif tc <= 520:
        buckets["500-520"] += 1
    else:
        buckets[">520"] += 1

print(f"\n3. TOKEN COUNT BUCKETS")
for bucket, count in buckets.items():
    pct = count / len(token_counts) * 100
    print(f"   {bucket}: {count} ({pct:.1f}%)")

print(f"\n4. CHUNKS PER TRANSCRIPT")
chunk_counts = list(chunks_per_transcript.values())
print(f"   Min: {min(chunk_counts)}")
print(f"   Max: {max(chunk_counts)}")
print(f"   Avg: {sum(chunk_counts)/len(chunk_counts):.1f}")

print(f"\n5. OVERLAP VERIFICATION (first 2 pairs)")
for i, pair in enumerate(overlap_test_pairs[:2]):
    print(f"\n   Pair {i+1}:")
    print(f"   Prev chunk ends with: ...{pair['prev_end'][-100:]}")
    print(f"   Curr chunk starts with: {pair['curr_start'][:100]}...")

print("\n" + "=" * 60)
