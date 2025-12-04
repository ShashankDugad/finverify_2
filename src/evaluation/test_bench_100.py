"""
FinVERIFY Test Bench - 100 Questions with F1 and EM Evaluation
"""
import json
import re
import string
from collections import Counter
import sys
sys.path.insert(0, '/scratch/sd5957/finverify_2/src')
from generator.answer_generator import FinVerifyRAG


def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """Exact match (EM) score."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def partial_match_score(prediction, ground_truth):
    """Check if key numbers/terms from ground truth appear in prediction."""
    # Extract numbers from both
    pred_nums = set(re.findall(r'[\d.]+', prediction))
    gold_nums = set(re.findall(r'[\d.]+', ground_truth))
    
    if gold_nums and pred_nums:
        overlap = len(pred_nums & gold_nums) / len(gold_nums)
        return overlap
    return 0.0


# 100 Test Questions with Ground Truth
TEST_SET = [
    # ==================== APPLE (20 questions) ====================
    {"query": "What was Apple's iPhone revenue in Q4 2023?", "ground_truth": "$43.8 billion", "company": "Apple"},
    {"query": "What was Apple's services revenue in Q4 2023?", "ground_truth": "$22.3 billion", "company": "Apple"},
    {"query": "What was Apple's total revenue in Q4 2023?", "ground_truth": "$89.5 billion", "company": "Apple"},
    {"query": "What was Apple's gross margin in Q4 2023?", "ground_truth": "45.2%", "company": "Apple"},
    {"query": "How did Apple's Mac revenue perform in Q4 2023?", "ground_truth": "$7.6 billion", "company": "Apple"},
    {"query": "What was Apple's iPad revenue in Q4 2023?", "ground_truth": "$6.4 billion", "company": "Apple"},
    {"query": "What was Apple's earnings per share in Q4 2023?", "ground_truth": "$1.46", "company": "Apple"},
    {"query": "How much cash does Apple have on hand?", "ground_truth": "$162 billion", "company": "Apple"},
    {"query": "What was Apple's revenue in Greater China?", "ground_truth": "$15 billion", "company": "Apple"},
    {"query": "What is Apple's installed base of devices?", "ground_truth": "2 billion", "company": "Apple"},
    {"query": "What was Apple's Q1 2024 revenue?", "ground_truth": "$119.6 billion", "company": "Apple"},
    {"query": "How did Apple's wearables segment perform?", "ground_truth": "$12 billion", "company": "Apple"},
    {"query": "What was Apple's operating income in Q4 2023?", "ground_truth": "$27 billion", "company": "Apple"},
    {"query": "What is Apple's dividend per share?", "ground_truth": "$0.24", "company": "Apple"},
    {"query": "What was Apple's R&D spending?", "ground_truth": "$7.3 billion", "company": "Apple"},
    {"query": "How many iPhones did Apple sell in Q4 2023?", "ground_truth": "50 million units", "company": "Apple"},
    {"query": "What was Apple's Americas revenue?", "ground_truth": "$40 billion", "company": "Apple"},
    {"query": "What is Apple's services gross margin?", "ground_truth": "70%", "company": "Apple"},
    {"query": "What was Apple's Europe revenue?", "ground_truth": "$25 billion", "company": "Apple"},
    {"query": "How much did Apple spend on share buybacks?", "ground_truth": "$25 billion", "company": "Apple"},

    # ==================== NVIDIA (15 questions) ====================
    {"query": "How much did NVIDIA's data center revenue grow?", "ground_truth": "427% year over year", "company": "Nvidia"},
    {"query": "What was NVIDIA's total revenue in Q1 2025?", "ground_truth": "$26 billion", "company": "Nvidia"},
    {"query": "What was NVIDIA's data center revenue in Q1 2025?", "ground_truth": "$22.6 billion", "company": "Nvidia"},
    {"query": "What was NVIDIA's gross margin?", "ground_truth": "78%", "company": "Nvidia"},
    {"query": "What was NVIDIA's gaming revenue?", "ground_truth": "$2.6 billion", "company": "Nvidia"},
    {"query": "What is NVIDIA's earnings per share?", "ground_truth": "$6.12", "company": "Nvidia"},
    {"query": "What was NVIDIA's operating income?", "ground_truth": "$18 billion", "company": "Nvidia"},
    {"query": "How much did NVIDIA spend on R&D?", "ground_truth": "$3 billion", "company": "Nvidia"},
    {"query": "What is NVIDIA's automotive revenue?", "ground_truth": "$329 million", "company": "Nvidia"},
    {"query": "What was NVIDIA's professional visualization revenue?", "ground_truth": "$427 million", "company": "Nvidia"},
    {"query": "What is NVIDIA's net income?", "ground_truth": "$14.9 billion", "company": "Nvidia"},
    {"query": "How much cash does NVIDIA have?", "ground_truth": "$31 billion", "company": "Nvidia"},
    {"query": "What was NVIDIA's revenue guidance for next quarter?", "ground_truth": "$28 billion", "company": "Nvidia"},
    {"query": "What is NVIDIA's AI chip demand outlook?", "ground_truth": "exceeds supply", "company": "Nvidia"},
    {"query": "How did NVIDIA's China revenue perform?", "ground_truth": "declined due to restrictions", "company": "Nvidia"},

    # ==================== MICROSOFT (15 questions) ====================
    {"query": "What is Microsoft Azure's revenue growth rate?", "ground_truth": "30% to 31%", "company": "Microsoft"},
    {"query": "What was Microsoft's total revenue?", "ground_truth": "$62 billion", "company": "Microsoft"},
    {"query": "What was Microsoft's cloud revenue?", "ground_truth": "$33 billion", "company": "Microsoft"},
    {"query": "What is Microsoft's intelligent cloud revenue?", "ground_truth": "$28 billion", "company": "Microsoft"},
    {"query": "What was Microsoft's operating income?", "ground_truth": "$27 billion", "company": "Microsoft"},
    {"query": "What is Microsoft's earnings per share?", "ground_truth": "$3.00", "company": "Microsoft"},
    {"query": "How did Microsoft 365 perform?", "ground_truth": "12% growth", "company": "Microsoft"},
    {"query": "What was Microsoft's LinkedIn revenue growth?", "ground_truth": "10%", "company": "Microsoft"},
    {"query": "What is Microsoft's gaming revenue?", "ground_truth": "$15 billion", "company": "Microsoft"},
    {"query": "How much did Microsoft invest in OpenAI?", "ground_truth": "$10 billion", "company": "Microsoft"},
    {"query": "What was Microsoft's gross margin?", "ground_truth": "69%", "company": "Microsoft"},
    {"query": "What is Microsoft's Windows revenue?", "ground_truth": "$6 billion", "company": "Microsoft"},
    {"query": "How much free cash flow did Microsoft generate?", "ground_truth": "$20 billion", "company": "Microsoft"},
    {"query": "What was Microsoft's advertising revenue?", "ground_truth": "$12 billion", "company": "Microsoft"},
    {"query": "What is Microsoft Copilot adoption rate?", "ground_truth": "rapid growth", "company": "Microsoft"},

    # ==================== AMAZON (15 questions) ====================
    {"query": "What was Amazon's AWS operating margin?", "ground_truth": "29.6%", "company": "Amazon"},
    {"query": "What was Amazon's total revenue?", "ground_truth": "$170 billion", "company": "Amazon"},
    {"query": "What was Amazon's AWS revenue?", "ground_truth": "$25 billion", "company": "Amazon"},
    {"query": "What is Amazon's North America segment revenue?", "ground_truth": "$90 billion", "company": "Amazon"},
    {"query": "What was Amazon's international revenue?", "ground_truth": "$35 billion", "company": "Amazon"},
    {"query": "What is Amazon's advertising revenue?", "ground_truth": "$14 billion", "company": "Amazon"},
    {"query": "What was Amazon's operating income?", "ground_truth": "$15 billion", "company": "Amazon"},
    {"query": "What is Amazon's net income?", "ground_truth": "$10 billion", "company": "Amazon"},
    {"query": "How did Amazon Prime membership perform?", "ground_truth": "200 million members", "company": "Amazon"},
    {"query": "What is Amazon's free cash flow?", "ground_truth": "$50 billion", "company": "Amazon"},
    {"query": "What was Amazon's AWS growth rate?", "ground_truth": "17%", "company": "Amazon"},
    {"query": "How much did Amazon spend on capex?", "ground_truth": "$15 billion", "company": "Amazon"},
    {"query": "What is Amazon's third-party seller services revenue?", "ground_truth": "$40 billion", "company": "Amazon"},
    {"query": "What was Amazon's subscription services revenue?", "ground_truth": "$10 billion", "company": "Amazon"},
    {"query": "How is Amazon's AI business performing?", "ground_truth": "multi-billion dollar run rate", "company": "Amazon"},

    # ==================== META (10 questions) ====================
    {"query": "What was Meta's advertising revenue?", "ground_truth": "$34 billion", "company": "META"},
    {"query": "What is Meta's total revenue?", "ground_truth": "$40 billion", "company": "META"},
    {"query": "What was Meta's operating margin?", "ground_truth": "41%", "company": "META"},
    {"query": "How many daily active users does Meta have?", "ground_truth": "3.2 billion", "company": "META"},
    {"query": "What is Meta's Reality Labs revenue?", "ground_truth": "$270 million", "company": "META"},
    {"query": "What was Meta's Reality Labs operating loss?", "ground_truth": "$4 billion", "company": "META"},
    {"query": "What is Meta's earnings per share?", "ground_truth": "$5.16", "company": "META"},
    {"query": "How much did Meta spend on R&D?", "ground_truth": "$10 billion", "company": "META"},
    {"query": "What is Meta's average revenue per user?", "ground_truth": "$11.50", "company": "META"},
    {"query": "How is Meta's AI advertising performing?", "ground_truth": "20% improvement in conversions", "company": "META"},

    # ==================== ALPHABET/GOOGLE (10 questions) ====================
    {"query": "What was Google Cloud's revenue?", "ground_truth": "$9 billion", "company": "Alphabet"},
    {"query": "What is Alphabet's total revenue?", "ground_truth": "$86 billion", "company": "Alphabet"},
    {"query": "What was Google Search revenue?", "ground_truth": "$48 billion", "company": "Alphabet"},
    {"query": "What is YouTube's advertising revenue?", "ground_truth": "$8 billion", "company": "Alphabet"},
    {"query": "What was Google Cloud's operating income?", "ground_truth": "$900 million", "company": "Alphabet"},
    {"query": "What is Alphabet's operating margin?", "ground_truth": "30%", "company": "Alphabet"},
    {"query": "How much did Alphabet return to shareholders?", "ground_truth": "$15 billion", "company": "Alphabet"},
    {"query": "What was Google Network revenue?", "ground_truth": "$8 billion", "company": "Alphabet"},
    {"query": "What is Alphabet's earnings per share?", "ground_truth": "$1.89", "company": "Alphabet"},
    {"query": "How is Google Cloud growth trending?", "ground_truth": "28% year over year", "company": "Alphabet"},

    # ==================== OTHER COMPANIES (15 questions) ====================
    {"query": "How did Walmart's e-commerce sales perform?", "ground_truth": "24% growth", "company": "Walmart"},
    {"query": "What was Walmart's total revenue?", "ground_truth": "$173 billion", "company": "Walmart"},
    {"query": "What is Walmart's US comparable sales growth?", "ground_truth": "4.9%", "company": "Walmart"},
    {"query": "What was Costco's membership revenue?", "ground_truth": "$1.1 billion", "company": "Costco"},
    {"query": "What is Costco's membership renewal rate?", "ground_truth": "93%", "company": "Costco"},
    {"query": "What was Nike's revenue?", "ground_truth": "$12.4 billion", "company": "Nike"},
    {"query": "What is Nike's direct-to-consumer revenue?", "ground_truth": "$5.4 billion", "company": "Nike"},
    {"query": "What was Adobe's revenue?", "ground_truth": "$5.4 billion", "company": "Adobe"},
    {"query": "What is Adobe's recurring revenue percentage?", "ground_truth": "94%", "company": "Adobe"},
    {"query": "What was Oracle's cloud revenue?", "ground_truth": "$5.6 billion", "company": "Oracle"},
    {"query": "What is Oracle's cloud growth rate?", "ground_truth": "25%", "company": "Oracle"},
    {"query": "What was Salesforce's revenue?", "ground_truth": "$9 billion", "company": "SalesForce"},
    {"query": "What is IBM's AI revenue?", "ground_truth": "$2 billion", "company": "IBM"},
    {"query": "What was JPMorgan's net interest income?", "ground_truth": "$23 billion", "company": "JPM"},
    {"query": "What is Accenture's revenue growth?", "ground_truth": "3%", "company": "Accenture"},
]


def run_test_bench():
    """Run evaluation on 100 test queries."""
    
    print("=" * 70)
    print("FinVERIFY TEST BENCH - 100 Questions")
    print("F1 Score & Exact Match Evaluation")
    print("=" * 70)
    
    # Initialize RAG
    print("\nLoading FinVERIFY RAG pipeline...")
    rag = FinVerifyRAG()
    
    # Results
    results = []
    total_em = 0
    total_f1 = 0
    total_partial = 0
    citation_matches = 0
    
    print(f"\nEvaluating {len(TEST_SET)} questions...\n")
    print("-" * 70)
    
    for i, test in enumerate(TEST_SET):
        query = test["query"]
        ground_truth = test["ground_truth"]
        expected_company = test["company"]
        
        # Get prediction
        response = rag.answer(query, top_k=3)
        prediction = response["answer"]
        
        # Calculate scores
        em = exact_match_score(prediction, ground_truth)
        f1 = f1_score(prediction, ground_truth)
        partial = partial_match_score(prediction, ground_truth)
        
        total_em += em
        total_f1 += f1
        total_partial += partial
        
        # Citation accuracy
        top_citation = response["citations"][0]["source"] if response["citations"] else "None"
        citation_match = expected_company.lower() in top_citation.lower()
        if citation_match:
            citation_matches += 1
        
        result = {
            "id": i + 1,
            "query": query,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "em": em,
            "f1": round(f1, 4),
            "partial": round(partial, 4),
            "citation": top_citation,
            "citation_match": citation_match,
            "company": expected_company
        }
        results.append(result)
        
        # Progress every 10 questions
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(TEST_SET)} queries...")
    
    # Summary by company
    print("\n" + "=" * 70)
    print("RESULTS BY COMPANY")
    print("=" * 70)
    
    companies = set(t["company"] for t in TEST_SET)
    company_stats = {}
    
    for company in sorted(companies):
        company_results = [r for r in results if r["company"] == company]
        if company_results:
            avg_f1 = sum(r["f1"] for r in company_results) / len(company_results)
            avg_em = sum(r["em"] for r in company_results) / len(company_results)
            avg_partial = sum(r["partial"] for r in company_results) / len(company_results)
            cite_acc = sum(1 for r in company_results if r["citation_match"]) / len(company_results)
            
            company_stats[company] = {
                "count": len(company_results),
                "f1": avg_f1,
                "em": avg_em,
                "partial": avg_partial,
                "citation_acc": cite_acc
            }
            print(f"  {company:15} | N={len(company_results):2} | F1={avg_f1:.3f} | EM={avg_em:.2%} | Cite={cite_acc:.2%}")
    
    # Overall summary
    avg_em = total_em / len(TEST_SET)
    avg_f1 = total_f1 / len(TEST_SET)
    avg_partial = total_partial / len(TEST_SET)
    citation_acc = citation_matches / len(TEST_SET)
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Total Queries:       {len(TEST_SET)}")
    print(f"  Exact Match (EM):    {avg_em:.2%} ({total_em}/{len(TEST_SET)})")
    print(f"  Average F1:          {avg_f1:.4f}")
    print(f"  Partial Match:       {avg_partial:.2%}")
    print(f"  Citation Accuracy:   {citation_acc:.2%} ({citation_matches}/{len(TEST_SET)})")
    print("=" * 70)
    
    # Sample outputs
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS (First 5)")
    print("=" * 70)
    for r in results[:5]:
        print(f"\n  Q: {r['query']}")
        print(f"  GT: {r['ground_truth']}")
        print(f"  Pred: {r['prediction']}")
        print(f"  F1={r['f1']:.3f} | EM={r['em']} | Citation: {r['citation']}")
    
    # Save results
    output = {
        "metrics": {
            "exact_match": avg_em,
            "f1_score": avg_f1,
            "partial_match": avg_partial,
            "citation_accuracy": citation_acc,
            "num_queries": len(TEST_SET)
        },
        "company_stats": company_stats,
        "results": results
    }
    
    output_file = "/scratch/sd5957/finverify_2/data/outputs/test_bench_100_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    run_test_bench()
