#!/usr/bin/env python3
"""FinVERIFY CLI Demo"""
import sys
sys.path.insert(0, '/scratch/sd5957/finverify_2/src')
from generator.answer_generator import FinVerifyRAG

def main():
    print("=" * 60)
    print("FinVERIFY: Financial Fact-Checking Demo")
    print("=" * 60)
    
    rag = FinVerifyRAG()
    print("\nReady! Type 'quit' to exit.\n")
    
    while True:
        question = input("Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        
        response = rag.answer(question, top_k=3)
        print(f"\nAnswer: {response['answer']}")
        print("Citations:")
        for i, c in enumerate(response['citations']):
            print(f"  [{i+1}] {c['source']} (score: {c['score']:.2f})")
        print()

if __name__ == "__main__":
    main()
