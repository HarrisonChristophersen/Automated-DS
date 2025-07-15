import argparse
from importing import load_and_inspect, detect_column_types
from preprocessing import handle_missing_values, feature_engineering
from modeling import run_h2o_automl
from explainability import (
    generate_lime_explanation,
    summarize_lime_explanations,
    generate_narrative,
    plot_feature_importance
)

def main():
    parser = argparse.ArgumentParser(description="Automated DS CLI")
    parser.add_argument("file", help="Path to CSV or XLSX")
    parser.add_argument("target", help="Target column name")
    parser.add_argument("--max_models", type=int, default=20, help="Max AutoML models")
    parser.add_argument("--max_secs", type=int, default=300, help="Max runtime seconds")
    parser.add_argument("--num_feats", type=int, default=5, help="LIME features per instance")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of instances to explain")
    args = parser.parse_args()

    df = load_and_inspect(args.file)
    types = detect_column_types(df)
    df = handle_missing_values(df, types)
    df = feature_engineering(df)

    model = run_h2o_automl(df, args.target, args.max_models, args.max_secs)
    exps = generate_lime_explanation(df, model, args.target, args.num_feats, args.num_samples)
    summary = summarize_lime_explanations(exps)
    narrative = generate_narrative(summary, args.target)

    print("\n=== Narrative ===")
    print(narrative)
    plot_feature_importance(summary, args.target)

if __name__ == "__main__":
    main()
